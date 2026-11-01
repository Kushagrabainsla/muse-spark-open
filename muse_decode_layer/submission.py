# okc: problem=muse_decode_layer
"""SGLang's decoder layer for Muse Spark Open (see SGLang's MuseSpark1xDecoderLayer.forward) with two of its
module groups replaced by our kernels; everything else is SGLang's, as in baseline.py:

  experts      NVFP4 grouped GEMMs in CUDA (as submitted to muse_moe_w4a16): only the routed (token, local
               expert) pairs, each touched expert's weights streamed once, SiLU and the post-expert norm
               and weighted sum fused, FP32 split-K accumulation.
  projections  qkv_gate, o, pooler and unpooler (and the router above 8 tokens, on the BF16 hi + lo split
               SGLang uses there) as cluster split-K skinny GEMMs in CUDA up to 32 tokens (as submitted to
               muse_bf16_proj), cuBLAS above, as SGLang.
  attention    as submitted to muse_attention_decode: on sliding decode steps and sliding verify blocks of up to
               16 requests one CUDA launch (prep, split-KV attention over the paged FP8 cache, the chunks merged
               in a cluster, the gated output norm); on full decode steps a fused Triton prep, the CUDA attention
               and a Triton combine with the gated norm; elsewhere the fused prep, SGLang's FA3 and norm-gate.
Up to 8 tokens the router keeps SGLang's FP32 input and weights.
"""

from __future__ import annotations


import math
from typing import TypedDict

import triton
import triton.language as tl
from torch.utils.cpp_extension import load_inline

import types

import torch
import torch.nn.functional as F
from sgl_kernel.flash_attn import flash_attn_with_kvcache
from sgl_kernel.scalar_type import scalar_types
from torch import nn

from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.kvcache.kvcache import store_cache
from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
from sglang.kernels.ops.muse_spark_v12.fused_muse_spark_routing import FusedMuseSparkRouting
from sglang.kernels.ops.muse_spark_v12.fused_norm_gate import FusedRMSNormGate
from sglang.kernels.ops.muse_spark_v12.fused_norm_residual import FusedPostNormResidualPreNorm
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
    fused_post_expert_norm_weighted_sum,
)
from sglang.kernels.ops.muse_spark_v12.fused_qk_norm import FusedQKNorm
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
from sglang.srt.layers.muse_spark_layers import (
    MuseSpark1xNormPreservingResidualConnection,
    MuseSpark1xRMSNorm,
    MuseSpark1xRMSNormWithInputScale,
)
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin
from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl



# ---- attention (muse_attention_decode) ----

import math

import torch
import triton
import triton.language as tl
from sgl_kernel.flash_attn import flash_attn_with_kvcache
from torch.utils.cpp_extension import load_inline

from sglang.kernels.ops.muse_spark_v12.norm_gate import rms_norm_gate


_attn_RMS_EPS = 1e-5
_attn_FP8 = torch.float8_e4m3fn
_attn_LOG2E = 1.4426950408889634
# The CUDA kernels run decode steps (q_len 1), where they beat FA3 at every batch, and sliding verify blocks of at
# most _attn_CUDA_VERIFY_MAX_BATCH requests (multi-draw sweeps on an H200); other verify blocks run SGLang's FA3 path with
# the fused prep (the wgmma kernel is behind FA3 on full layers: its serial per-stage chain and the 64 B-row gather
# rate, see EXPERIMENTS.md A9). _attn_CHUNK overrides the keys per CTA (None: by shape).
_attn_CUDA_VERIFY_MAX_BATCH = 16
_attn_CHUNK = {"sliding": None, "full": None}
_attn_STAGE_KEYS = {"sliding": 64, "full": 64}
_attn_MMA8 = {1: False, 8: True}  # per q_len: E4M3 MMAs (attn8_kernel), else FP16 MMAs on converted fragments
_attn_WGMMA = True  # verify blocks (q_len 8, 8 query heads per KV head): the wgmma kernel (attnwg_kernel) instead of attn8
_attn_VERIFY_HEADS = {"sliding": 1, "full": 1}  # KV heads per CTA of the wgmma kernel
_attn_VERIFY_PC = {"sliding": True, "full": True}  # the warp-specialized wgmma kernel (attnpc_kernel) instead
_attn_FUSED = {"sliding": True, "full": False}  # decode steps in one launch (attnc), when the chunks divide 16
_attn_FUSED_VERIFY = True  # sliding verify blocks in one launch (attnpcc)


def _attn_chunk(window: int, batch: int, length: int) -> int:
    """Keys per CTA: sliding layers aim at ~128 CTAs (128 keys at least), full layers grow chunks with the batch;
    verify blocks (one CTA per KV head) aim at ~300 CTAs on sliding layers and keep 4 CTAs per SM on full ones."""
    if _attn_CHUNK["sliding" if window else "full"]:
        return _attn_CHUNK["sliding" if window else "full"]
    if length > 1:
        if window:
            return 64 if batch <= 4 else 512  # bs1 1.45x at 64, bs16 1.08x at 512 (5 draws, the attnpc kernel)
        return 512 if batch <= 2 else 2048
    if window:
        return min(2048, max(128, triton.next_power_of_2(triton.cdiv(window * batch, 128))))
    return 512 if batch <= 2 else 1024 if batch <= 8 else 2048
_attn_STAGES = 4


@triton.jit
def _attn_rms_rope(x, pos, theta_log2, HAS_ROPE: tl.constexpr, R: tl.constexpr, D: tl.constexpr):
    """x [R, D] FP32: RMSNorm over each row (head), then interleaved RoPE on (2i, 2i + 1)."""
    x = x * tl.rsqrt(tl.sum(x * x, axis=1) / D + 1e-5)[:, None]
    if HAS_ROPE:
        xe, xo = tl.split(tl.reshape(x, (R, D // 2, 2)))
        i = tl.arange(0, D // 2).to(tl.float32)
        ang = pos.to(tl.float32) * tl.exp2(-(2.0 * i / D) * theta_log2)
        c = tl.cos(ang)
        s = tl.sin(ang)
        x = tl.reshape(tl.join(xe * c[None, :] - xo * s[None, :], xe * s[None, :] + xo * c[None, :]), (R, D))
    return x


@triton.jit
def _attn_prep(qkv_ptr, pos_ptr, loc_ptr, ks_ptr, vs_ptr, q16_ptr, kc_ptr, vc_ptr, theta_log2, q_mult,
          HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, QKV_W: tl.constexpr, HAS_ROPE: tl.constexpr):
    t = tl.program_id(0)
    pos = tl.load(pos_ptr + t)
    dn = tl.arange(0, D)
    hq = tl.arange(0, HQ)
    q = tl.load(qkv_ptr + t * QKV_W + hq[:, None] * D + dn[None, :]).to(tl.float32)
    q = _attn_rms_rope(q, pos, theta_log2, HAS_ROPE, HQ, D) * q_mult
    tl.store(q16_ptr + t * HQ * D + hq[:, None] * D + dn[None, :], q.to(tl.float16))
    hk = tl.arange(0, HKV)
    k = tl.load(qkv_ptr + t * QKV_W + (HQ + hk)[:, None] * D + dn[None, :]).to(tl.float32)
    k = _attn_rms_rope(k, pos, theta_log2, HAS_ROPE, HKV, D)
    v = tl.load(qkv_ptr + t * QKV_W + (HQ + HKV + hk)[:, None] * D + dn[None, :]).to(tl.float32)
    row = tl.load(loc_ptr + t) * HKV
    tl.store(kc_ptr + (row + hk)[:, None] * D + dn[None, :], (k / tl.load(ks_ptr)).to(tl.float8e4nv))
    tl.store(vc_ptr + (row + hk)[:, None] * D + dn[None, :], (v / tl.load(vs_ptr)).to(tl.float8e4nv))


@triton.jit
def _attn_attn_combine(qkv_ptr, out_ptr, ws_o, ws_m, ws_l, NCH,
                  HKV: tl.constexpr, KVPC: tl.constexpr, G: tl.constexpr, L: tl.constexpr, D: tl.constexpr,
                  RH: tl.constexpr, QKV_W: tl.constexpr, HQ: tl.constexpr, CT: tl.constexpr):
    """One query row: merges its chunks (CT at a time), then RMSNorm over the head times sigmoid(gate)."""
    R: tl.constexpr = KVPC * RH
    pid = tl.program_id(0)
    bg = pid // R
    row = pid % R
    r = row % RH
    if r >= G * L:
        return
    b = bg // (HKV // KVPC)
    h = (bg % (HKV // KVPC)) * KVPC + row // RH
    j = r // G
    g = r % G
    dn = tl.arange(0, D)
    cc = tl.arange(0, CT)
    m = float("-inf")
    for c0 in range(0, NCH, CT):
        idx = ((bg * NCH + c0 + cc) * R + row)
        m = tl.maximum(m, tl.max(tl.load(ws_m + idx, mask=c0 + cc < NCH, other=float("-inf"))))
    m_safe = tl.where(m == float("-inf"), 0.0, m)
    l = 0.0
    acc = tl.zeros((D,), tl.float32)
    for c0 in range(0, NCH, CT):
        idx = ((bg * NCH + c0 + cc) * R + row)
        mc = tl.load(ws_m + idx, mask=c0 + cc < NCH, other=float("-inf"))
        live = mc != float("-inf")  # chunks past a request's keys left their outputs unwritten
        w = tl.where(live, tl.exp2(mc - m_safe), 0.0)
        l += tl.sum(w * tl.load(ws_l + idx, mask=live, other=0.0))
        acc += tl.sum(w[:, None] * tl.load(ws_o + idx[:, None] * D + dn[None, :], mask=live[:, None], other=0.0),
                      axis=0)
    o = acc / tl.where(l > 0, l, 1.0)
    o = o.to(tl.bfloat16).to(tl.float32)  # the attention output is BF16 before the norm, as in SGLang
    rr = tl.rsqrt(tl.sum(o * o) / D + 1e-5)
    tok = b * L + j
    head = h * G + g
    gate = tl.load(qkv_ptr + tok * QKV_W + (HQ + 2 * HKV) * D + head * D + dn).to(tl.float32)
    tl.store(out_ptr + tok * (HQ * D) + head * D + dn, (o * rr * tl.sigmoid(gate)).to(tl.bfloat16))


@triton.jit
def _attn_qkv_prep(qkv_ptr, pos_ptr, loc_ptr, ks_ptr, vs_ptr, q8_ptr, kc_ptr, vc_ptr, theta_log2,
              HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, QKV_W: tl.constexpr, HAS_ROPE: tl.constexpr):
    t = tl.program_id(0)
    pos = tl.load(pos_ptr + t)
    dn = tl.arange(0, D)
    hq = tl.arange(0, HQ)
    q = tl.load(qkv_ptr + t * QKV_W + hq[:, None] * D + dn[None, :]).to(tl.float32)
    q = _attn_rms_rope(q, pos, theta_log2, HAS_ROPE, HQ, D)
    tl.store(q8_ptr + t * HQ * D + hq[:, None] * D + dn[None, :], q.to(tl.float8e4nv))
    hk = tl.arange(0, HKV)
    k = tl.load(qkv_ptr + t * QKV_W + (HQ + hk)[:, None] * D + dn[None, :]).to(tl.float32)
    k = _attn_rms_rope(k, pos, theta_log2, HAS_ROPE, HKV, D)
    v = tl.load(qkv_ptr + t * QKV_W + (HQ + HKV + hk)[:, None] * D + dn[None, :]).to(tl.float32)
    row = tl.load(loc_ptr + t) * HKV
    tl.store(kc_ptr + (row + hk)[:, None] * D + dn[None, :], (k / tl.load(ks_ptr)).to(tl.float8e4nv))
    tl.store(vc_ptr + (row + hk)[:, None] * D + dn[None, :], (v / tl.load(vs_ptr)).to(tl.float8e4nv))


def _attn_kernel_fa3(data: Inputs) -> torch.Tensor:
    """SGLang's FA3 path with the fused prep (the A3 submission)."""
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    tokens = batch * length
    qkv = data["qkv"].view(tokens, -1)
    theta = cfg["rope_theta"]
    q8 = torch.empty((tokens, hq, d), dtype=_attn_FP8, device=qkv.device)
    k_cache, v_cache = data["k_cache"], data["v_cache"]
    _attn_qkv_prep[(tokens,)](qkv, data["positions"], data["cache_loc"], data["k_scale"], data["v_scale"], q8, k_cache,
                         v_cache, math.log2(theta) if theta else 0.0, HQ=hq, HKV=hkv, D=d, QKV_W=qkv.shape[1],
                         HAS_ROPE=bool(theta), num_warps=4)
    window = cfg["window"]
    descale = {}
    if cfg["kv_scale"] != 1.0:
        descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(batch, hkv),
                       v_descale=data["v_scale"].reshape(1, 1).expand(batch, hkv))
    out = flash_attn_with_kvcache(
        q=q8,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=data["page_table"],
        cache_seqlens=data["cache_seqlens"],
        cu_seqlens_q=data["cu_seqlens_q"],
        max_seqlen_q=length,
        # The 5.473 query factor is folded into the softmax scale, as SGLang does.
        softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"],
        causal=True,
        window_size=(window - 1, 0) if window else (-1, -1),
        scheduler_metadata=data["scheduler_metadata"],
        **descale,
    )
    if isinstance(out, tuple):
        out = out[0]
    gate = qkv[:, (hq + 2 * hkv) * d:]
    o = rms_norm_gate(out.reshape(tokens, hq * d).to(torch.bfloat16), gate, None, head_dim=d, eps=_attn_RMS_EPS,
                      has_norm=True)
    return o.view(batch, length, hq * d)


def _attn_wgmma_fn(n: int) -> str:
    """CUDA source of wgmma_<n>: one wgmma.mma_async m64n<n>k32 f32 += e4m3 (A in registers) x e4m3 (B by descriptor).
    The instruction reads the A registers asynchronously: the caller keeps them live (pinned with an empty asm) past
    the wgmma.wait_group that completes it, so the compiler does not reuse them in between. (In-out constraints on
    them instead make ptxas serialize consecutive wgmmas, C7513.)"""
    nreg = n // 2
    d = ",".join(f"%{i}" for i in range(nreg))
    a = ",".join(f"%{nreg + i}" for i in range(4))
    outs = ", ".join(f'"+f"(d[{i}])' for i in range(nreg))
    return f"""
__device__ __forceinline__ void wgmma_{n}(float* d, const uint32_t* a, uint64_t bdesc, int scale_d) {{
  asm volatile("{{ .reg .pred p; setp.ne.b32 p, %{nreg + 4}, 0; "
               "wgmma.mma_async.sync.aligned.m64n{n}k32.f32.e4m3.e4m3 {{{d}}}, {{{a}}}, %{nreg + 5}, p, 1, 1; }}"
               : {outs}
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(scale_d), "l"(bdesc));
}}"""


_attn_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>

__device__ __forceinline__ uint32_t e4m3x2_f16x2(uint32_t v) {  // the low 2 bytes of v
  uint32_t r;
  asm("{ .reg .b16 lo, hi; mov.b32 {lo, hi}, %1; cvt.rn.f16x2.e4m3x2 %0, lo; }" : "=r"(r) : "r"(v));
  return r;
}

__device__ __forceinline__ void mma_f16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                        uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t pack_f16(float lo, float hi) {
  __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&h);
}

__device__ __forceinline__ void cp16(void* dst, const void* src, bool valid) {  // zero-fills when !valid
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"((uint32_t)__cvta_generic_to_shared(dst)),
               "l"(src), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;"); }
template <int N>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N));
}

// Stage tile: SK keys x 128 B of keys (both heads), then SK x 128 B of values. Keys: 16-byte chunk c of key row k
// at chunk c ^ ((k & 1) << 2). Values: chunk c at c ^ (((k >> 1) & 3) << 1).
template <int L, int STG, int SK>
__global__ void __launch_bounds__(L == 1 ? 128 : 256) attn_kernel(
    const __half* __restrict__ q16, const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,
    const int* __restrict__ pt, const int* __restrict__ seqlens, float* __restrict__ ws_o, float* __restrict__ ws_m,
    float* __restrict__ ws_l, int HKV, int max_pages, int page, int window, int CH, int NCH) {
  constexpr int G = 8, D = 64, RH = G * L < 16 ? 16 : G * L, MT = RH / 16;
  constexpr int NT = L == 1 ? 128 : 256, NWARP = NT / 32;
  constexpr int KS = NWARP / (2 * MT);  // warps splitting each stage's keys: 2 (L = 1) or 1 (L = 8)
  constexpr int KPW = SK / KS;          // keys per warp per stage
  constexpr int TILE = SK * 256;
  constexpr int NTS = KPW / 8, KSP = KPW / 16;
  extern __shared__ uint4 smem4[];
  uint8_t* smem = reinterpret_cast<uint8_t*>(smem4);

  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, q = lane & 3;
  const int ngroups = HKV / 2;
  const int bg = blockIdx.x, c = blockIdx.y;
  const int b = bg / ngroups, h0 = (bg % ngroups) * 2;
  const int seq = seqlens[b], hi = seq + L;
  const int lo = window > 0 ? max(seq - (window - 1), 0) : 0;
  const int cs = lo + c * CH, ce = min(cs + CH, hi);
  constexpr int R = 2 * RH;
  const size_t row_base = ((size_t)bg * NCH + c) * R;
  const int unit = warp % (2 * MT), ks = warp / (2 * MT);
  const int hh = unit / MT, mt = unit % MT;
  if (cs >= hi) {
    for (int i = threadIdx.x; i < R; i += NT) {
      ws_m[row_base + i] = -INFINITY;
      ws_l[row_base + i] = 0.f;
    }
    return;
  }

  // This warp's query rows (m-tile mt of head h0 + hh): r = mt * 16 + g and + 8; row r is token r / G, query head
  // (h0 + hh) * G + r % G. Lane q, k-step s holds dims 16q + 4s .. + 3 (the same permutation as its keys).
  const int ra = mt * 16 + g, rb = ra + 8;
  uint32_t qa[4][4];
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    uint2 v0 = make_uint2(0, 0), v1 = make_uint2(0, 0);
    if (ra < G * L)
      v0 = *reinterpret_cast<const uint2*>(q16 + ((size_t)(b * L + ra / G) * (HKV * G) + (h0 + hh) * G + ra % G) * D +
                                           16 * q + 4 * s);
    if (rb < G * L)
      v1 = *reinterpret_cast<const uint2*>(q16 + ((size_t)(b * L + rb / G) * (HKV * G) + (h0 + hh) * G + rb % G) * D +
                                           16 * q + 4 * s);
    qa[s][0] = v0.x;
    qa[s][1] = v1.x;
    qa[s][2] = v0.y;
    qa[s][3] = v1.y;
  }
  const int qpa = seq + ra / G, qpb = seq + rb / G;
  const bool va = ra < G * L, vb = rb < G * L;

  const int nsteps = (ce - cs + SK - 1) / SK;
  // The chunk's cache rows (from the page table) go to shared memory first, all loads in flight at once, so the
  // copy ring never waits on the page table. Each thread then copies 16 SK / NT 16-byte pieces per stage.
  int* rows = reinterpret_cast<int*>(smem + STG * TILE);
  for (int i = threadIdx.x; i < ce - cs; i += NT) {
    const int pos = cs + i;
    rows[i] = pt[(size_t)b * max_pages + pos / page] * page + pos % page;
  }
  __syncthreads();
  constexpr int PER = 16 * SK / NT;
  auto issue = [&](int stage, int kb) {
    uint8_t* tile = smem + stage * TILE;
#pragma unroll
    for (int i = 0; i < PER; ++i) {
      const int idx = threadIdx.x + i * NT;
      const int key = idx >> 4, part = (idx >> 3) & 1, ch = idx & 7;
      const bool ok = kb + key < ce;
      const int row = ok ? rows[kb + key - cs] : 0;
      const uint8_t* src = (part ? vc : kc) + ((long long)row * HKV + h0) * D + ch * 16;
      const int sw = part ? (ch ^ (((key >> 1) & 3) << 1)) : (ch ^ ((key & 1) << 2));
      cp16(tile + part * (SK * 128) + key * 128 + sw * 16, src, ok);
    }
  };
#pragma unroll
  for (int s = 0; s < STG - 1; ++s) {
    if (s < nsteps) issue(s, cs + SK * s);
    cp_commit();
  }

  float ma = -INFINITY, mb = -INFINITY, la = 0.f, lb = 0.f;
  float acc[8][4];
#pragma unroll
  for (int i = 0; i < 8; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;

  for (int it = 0; it < nsteps; ++it) {
    cp_wait<STG - 2>();
    __syncthreads();
    {
      const int nx = it + STG - 1;
      if (nx < nsteps) issue(nx % STG, cs + SK * nx);
      cp_commit();
    }
    const uint8_t* Kt = smem + (it % STG) * TILE;
    const uint8_t* Vt = Kt + SK * 128;
    const int kbs = cs + SK * it, kb = kbs + ks * KPW;  // the stage's and this warp's first key

    float sc[NTS][4];
#pragma unroll
    for (int t = 0; t < NTS; ++t) {
#pragma unroll
      for (int j = 0; j < 4; ++j) sc[t][j] = 0.f;
      const int key = ks * KPW + 8 * t + g;
      const uint4 kv = *reinterpret_cast<const uint4*>(Kt + key * 128 + (((4 * hh + q) ^ ((key & 1) << 2)) * 16));
      const uint32_t w[4] = {kv.x, kv.y, kv.z, kv.w};
#pragma unroll
      for (int s = 0; s < 4; ++s)
        mma_f16(sc[t], qa[s][0], qa[s][1], qa[s][2], qa[s][3], e4m3x2_f16x2(w[s]), e4m3x2_f16x2(w[s] >> 16));
    }
    // Masks: element (row, key kb + 8t + 2q + e). Only stages holding keys past the chunk's end, keys after the
    // block's first query, or (sliding verify) keys before its last query's window need them; padding rows of the
    // other stages see unmasked scores, which never reach the output.
    const bool edge = kbs + SK > ce || kbs + SK > hi - L + 1 || (window > 0 && kbs < lo + L - 1);
    if (edge) {
#pragma unroll
      for (int t = 0; t < NTS; ++t)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int pos = kb + 8 * t + 2 * q + e;
          const bool okk = pos < ce;
          const bool oka = okk && va && pos <= qpa && (window == 0 || pos > qpa - window);
          const bool okb = okk && vb && pos <= qpb && (window == 0 || pos > qpb - window);
          sc[t][e] = oka ? sc[t][e] : -INFINITY;
          sc[t][2 + e] = okb ? sc[t][2 + e] : -INFINITY;
        }
    }
    if (!vb) {  // padding rows (q_len 1): masked everywhere, unmasked they slow the kernel ~1.6x
#pragma unroll
      for (int t = 0; t < NTS; ++t) sc[t][2] = sc[t][3] = -INFINITY;
    }
    float mxa = -INFINITY, mxb = -INFINITY;
#pragma unroll
    for (int t = 0; t < NTS; ++t) {
      mxa = fmaxf(mxa, fmaxf(sc[t][0], sc[t][1]));
      mxb = fmaxf(mxb, fmaxf(sc[t][2], sc[t][3]));
    }
#pragma unroll
    for (int o = 1; o <= 2; o <<= 1) {
      mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, o));
      mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, o));
    }
    const float mna = fmaxf(ma, mxa), mnb = fmaxf(mb, mxb);
    const float sa = mna == -INFINITY ? 0.f : mna, sb = mnb == -INFINITY ? 0.f : mnb;
    const float aa = exp2f(ma - sa), ab = exp2f(mb - sb);
    ma = mna;
    mb = mnb;
    la *= aa;
    lb *= ab;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      acc[i][0] *= aa;
      acc[i][1] *= aa;
      acc[i][2] *= ab;
      acc[i][3] *= ab;
    }
    uint32_t pa[NTS], pb[NTS];
#pragma unroll
    for (int t = 0; t < NTS; ++t) {
      const float p0 = exp2f(sc[t][0] - sa), p1 = exp2f(sc[t][1] - sa);
      const float p2 = exp2f(sc[t][2] - sb), p3 = exp2f(sc[t][3] - sb);
      la += p0 + p1;
      lb += p2 + p3;
      pa[t] = pack_f16(p0, p1);
      pb[t] = pack_f16(p2, p3);
    }
    // O += P V. k-step u: keys 16u .. 16u + 15 of this warp; lane (g, q) supplies dims 8g + nt of keys 2q, 2q + 1,
    // 2q + 8, 2q + 9 (head hh: bytes 64 hh + 8g .. + 7 of each value row).
#pragma unroll
    for (int u = 0; u < KSP; ++u) {
      uint2 vr[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int key = ks * KPW + 16 * u + 2 * q + (e & 1) + (e >> 1) * 8;
        const int unit8 = (8 * hh + g) ^ (((key >> 1) & 3) << 2);
        vr[e] = *reinterpret_cast<const uint2*>(Vt + key * 128 + unit8 * 8);
      }
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        const uint32_t x0 = nt < 4 ? vr[0].x : vr[0].y, x1 = nt < 4 ? vr[1].x : vr[1].y;
        const uint32_t x2 = nt < 4 ? vr[2].x : vr[2].y, x3 = nt < 4 ? vr[3].x : vr[3].y;
        const uint32_t sel = (nt & 3) | (((nt & 3) + 4) << 4);
        const uint32_t b0 = e4m3x2_f16x2(__byte_perm(x0, x1, sel));
        const uint32_t b1 = e4m3x2_f16x2(__byte_perm(x2, x3, sel));
        mma_f16(acc[nt], pa[2 * u], pb[2 * u], pa[2 * u + 1], pb[2 * u + 1], b0, b1);
      }
    }
  }
  cp_wait<0>();

#pragma unroll
  for (int o = 1; o <= 2; o <<= 1) {
    la += __shfl_xor_sync(0xffffffffu, la, o);
    lb += __shfl_xor_sync(0xffffffffu, lb, o);
  }
  if constexpr (KS > 1) {
    // Warps of the second key half hand their state to the first through shared memory.
    __syncthreads();
    float* buf = reinterpret_cast<float*>(smem) + (unit * 32 + lane) * 36;
    if (ks == 1) {
      buf[0] = ma;
      buf[1] = mb;
      buf[2] = la;
      buf[3] = lb;
#pragma unroll
      for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) buf[4 + i * 4 + j] = acc[i][j];
    }
    __syncthreads();
    if (ks == 1) return;
    const float m2a = buf[0], m2b = buf[1];
    const float Ma = fmaxf(ma, m2a), Mb = fmaxf(mb, m2b);
    const float Sa = Ma == -INFINITY ? 0.f : Ma, Sb = Mb == -INFINITY ? 0.f : Mb;
    const float w1a = exp2f(ma - Sa), w2a = exp2f(m2a - Sa), w1b = exp2f(mb - Sb), w2b = exp2f(m2b - Sb);
    la = la * w1a + buf[2] * w2a;
    lb = lb * w1b + buf[3] * w2b;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      acc[i][0] = acc[i][0] * w1a + buf[4 + i * 4 + 0] * w2a;
      acc[i][1] = acc[i][1] * w1a + buf[4 + i * 4 + 1] * w2a;
      acc[i][2] = acc[i][2] * w1b + buf[4 + i * 4 + 2] * w2b;
      acc[i][3] = acc[i][3] * w1b + buf[4 + i * 4 + 3] * w2b;
    }
    ma = Ma;
    mb = Mb;
  }
  // Rows hh * RH + r. Output columns: c0 of n-tile nt is dim 16q + nt, c1 is 16q + 8 + nt.
  const size_t rowa = row_base + hh * RH + ra, rowb = row_base + hh * RH + rb;
  if (q == 0) {
    ws_m[rowa] = va ? ma : -INFINITY;
    ws_l[rowa] = va ? la : 0.f;
    ws_m[rowb] = vb ? mb : -INFINITY;
    ws_l[rowb] = vb ? lb : 0.f;
  }
#pragma unroll
  for (int nt = 0; nt < 8; ++nt) {
    ws_o[rowa * D + 16 * q + nt] = acc[nt][0];
    ws_o[rowa * D + 16 * q + 8 + nt] = acc[nt][1];
    ws_o[rowb * D + 16 * q + nt] = acc[nt][2];
    ws_o[rowb * D + 16 * q + 8 + nt] = acc[nt][3];
  }
}

void attn(torch::Tensor q16, torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor seqlens,
          torch::Tensor ws_o, torch::Tensor ws_m, torch::Tensor ws_l, int64_t L, int64_t HKV, int64_t page,
          int64_t window, int64_t CH, int64_t NCH, int64_t batch, int64_t SK) {
  constexpr int STG = 4;
  const size_t smem = STG * 256 * SK + 4 * CH;  // stage ring, then the chunk's cache rows
  TORCH_CHECK(smem <= 200 * 1024, "attn: chunk too large");
  dim3 grid(batch * (HKV / 2), NCH);
  auto stream = at::cuda::getCurrentCUDAStream();
#define LAUNCH(LL, SKK)                                                                                        \
  {                                                                                                           \
    auto k = attn_kernel<LL, STG, SKK>;                                                                           \
    static bool set = false;                                                                                   \
    if (!set) {                                                                                                \
      cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024);                    \
      set = true;                                                                                              \
    }                                                                                                          \
    k<<<grid, LL == 1 ? 128 : 256, smem, stream>>>(                                                            \
        reinterpret_cast<const __half*>(q16.data_ptr()), reinterpret_cast<const uint8_t*>(kc.data_ptr()),      \
        reinterpret_cast<const uint8_t*>(vc.data_ptr()), pt.data_ptr<int>(), seqlens.data_ptr<int>(),           \
        ws_o.data_ptr<float>(), ws_m.data_ptr<float>(), ws_l.data_ptr<float>(), (int)HKV, (int)pt.size(1),     \
        (int)page, (int)window, (int)CH, (int)NCH);                                                            \
  }
  TORCH_CHECK((L == 1 || L == 8) && (SK == 32 || SK == 64), "attn: q_len must be 1 or 8, stage 32 or 64 keys");
  if (L == 1 && SK == 32) LAUNCH(1, 32) else if (L == 1) LAUNCH(1, 64) else if (SK == 32) LAUNCH(8, 32) else LAUNCH(8, 64)
}
__device__ __forceinline__ void mma_e4m3(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t pack_e4m3x4(float a, float b, float c, float d) {  // bytes a, b, c, d from low
  uint32_t r;
  asm("{ .reg .b16 lo, hi; cvt.rn.satfinite.e4m3x2.f32 lo, %2, %1; cvt.rn.satfinite.e4m3x2.f32 hi, %4, %3;"
      " mov.b32 %0, {lo, hi}; }"
      : "=r"(r) : "f"(a), "f"(b), "f"(c), "f"(d));
  return r;
}

// Bytes nt of x0..x3 -> word nt (byte e from x_e), nt = 0..3.
__device__ __forceinline__ void transpose4(uint32_t x0, uint32_t x1, uint32_t x2, uint32_t x3, uint32_t* o) {
  const uint32_t a = __byte_perm(x0, x1, 0x5140), b = __byte_perm(x0, x1, 0x7362);
  const uint32_t c = __byte_perm(x2, x3, 0x5140), d = __byte_perm(x2, x3, 0x7362);
  o[0] = __byte_perm(a, c, 0x5410);
  o[1] = __byte_perm(a, c, 0x7632);
  o[2] = __byte_perm(b, d, 0x5410);
  o[3] = __byte_perm(b, d, 0x7632);
}

// The same CTA layout, tiles and ring as attn_kernel, with E4M3 MMAs (m16n8k32): E4M3 queries (as FA3) and keys go
// in as they are, and the probabilities are rounded to E4M3 for P V (as FA3's _attn_FP8 kernel does). Lane q, k-step s
// holds query/key dims 16q + 8s .. + 7 (one 16-byte read per key). S n-tile t column c is key 8t + c; P V k-step u
// position 4q + e is key 32u + 2q + (e & 1) + 8 (e >> 1) (+ 16 for positions 16 + 4q + e), so the score fragments
// are the P fragments. Each stage's values are transposed once, by all threads, into VT: per head and dim one row of
// SK bytes in position order, 8-byte unit 4u + q holding lane q's positions (b0 then b1), units XOR-swizzled by bit 1
// of the dim. V n-tile nt column n is dim 8nt + n.
template <int L, int STG, int SK>
__global__ void __launch_bounds__(L == 1 ? 128 : 256, L == 1 ? 4 : 2) attn8_kernel(
    const uint8_t* __restrict__ q8, const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,
    const int* __restrict__ pt, const int* __restrict__ seqlens, const float* __restrict__ kscale,
    const float* __restrict__ vscale, float* __restrict__ ws_o, float* __restrict__ ws_m, float* __restrict__ ws_l,
    float qscale, int HKV, int max_pages, int page, int window, int CH, int NCH) {
  constexpr int G = 8, D = 64, RH = G * L < 16 ? 16 : G * L, MT = RH / 16;
  constexpr int NT = L == 1 ? 128 : 256, NWARP = NT / 32;
  constexpr int KS = NWARP / (2 * MT);
  constexpr int KPW = SK / KS;
  static_assert(KPW % 32 == 0, "a warp's keys per stage must fill k32 steps");
  constexpr int TILE = SK * 256;
  constexpr int NTS = KPW / 8, KSP = KPW / 32;
  extern __shared__ uint4 smem4[];
  uint8_t* smem = reinterpret_cast<uint8_t*>(smem4);

  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, q = lane & 3;
  const int ngroups = HKV / 2;
  const int bg = blockIdx.x, c = blockIdx.y;
  const int b = bg / ngroups, h0 = (bg % ngroups) * 2;
  const int seq = seqlens[b], hi = seq + L;
  const int lo = window > 0 ? max(seq - (window - 1), 0) : 0;
  const int cs = lo + c * CH, ce = min(cs + CH, hi);
  constexpr int R = 2 * RH;
  const size_t row_base = ((size_t)bg * NCH + c) * R;
  const int unit = warp % (2 * MT), ks = warp / (2 * MT);
  const int hh = unit / MT, mt = unit % MT;
  if (cs >= hi) {
    for (int i = threadIdx.x; i < R; i += NT) {
      ws_m[row_base + i] = -INFINITY;
      ws_l[row_base + i] = 0.f;
    }
    return;
  }

  const int ra = mt * 16 + g, rb = ra + 8;
  const bool va = ra < G * L, vb = rb < G * L;
  uint4 qx = make_uint4(0, 0, 0, 0), qy = make_uint4(0, 0, 0, 0);
  if (va) qx = *reinterpret_cast<const uint4*>(q8 + ((size_t)(b * L + ra / G) * (HKV * G) + (h0 + hh) * G + ra % G) * D + 16 * q);
  if (vb) qy = *reinterpret_cast<const uint4*>(q8 + ((size_t)(b * L + rb / G) * (HKV * G) + (h0 + hh) * G + rb % G) * D + 16 * q);
  const int qpa = seq + ra / G, qpb = seq + rb / G;
  const float sc_mul = qscale * kscale[0];  // scores -> base-2 logits

  const int nsteps = (ce - cs + SK - 1) / SK;
  uint8_t* VT = smem + STG * TILE;
  int* rows = reinterpret_cast<int*>(VT + SK * 128);
  for (int i = threadIdx.x; i < ce - cs; i += NT) {
    const int pos = cs + i;
    rows[i] = pt[(size_t)b * max_pages + pos / page] * page + pos % page;
  }
  __syncthreads();
  constexpr int PER = 16 * SK / NT;
  auto issue = [&](int stage, int kb) {
    uint8_t* tile = smem + stage * TILE;
#pragma unroll
    for (int i = 0; i < PER; ++i) {
      const int idx = threadIdx.x + i * NT;
      const int key = idx >> 4, part = (idx >> 3) & 1, ch = idx & 7;
      const bool ok = kb + key < ce;
      const int row = ok ? rows[kb + key - cs] : 0;
      const uint8_t* src = (part ? vc : kc) + ((long long)row * HKV + h0) * D + ch * 16;
      const int sw = part ? (ch ^ (((key >> 1) & 3) << 1)) : (ch ^ ((key & 1) << 2));
      cp16(tile + part * (SK * 128) + key * 128 + sw * 16, src, ok);
    }
  };
#pragma unroll
  for (int s = 0; s < STG - 1; ++s) {
    if (s < nsteps) issue(s, cs + SK * s);
    cp_commit();
  }

  float ma = -INFINITY, mb = -INFINITY, la = 0.f, lb = 0.f;
  float acc[8][4];
#pragma unroll
  for (int i = 0; i < 8; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;

  for (int it = 0; it < nsteps; ++it) {
    cp_wait<STG - 2>();
    __syncthreads();  // also: every warp is done with the last stage's VT
    {
      const int nx = it + STG - 1;
      if (nx < nsteps) issue(nx % STG, cs + SK * nx);
      cp_commit();
    }
    const uint8_t* Kt = smem + (it % STG) * TILE;
    {
      // VT: block (head, u, half, q, dim group) = 4 dims x positions 4q .. + 3 of half `half` of k-step u.
      const uint8_t* Vt = Kt + SK * 128;
#pragma unroll
      for (int blk = threadIdx.x; blk < SK * 8; blk += NT) {
        const int dg = blk & 15, qq = (blk >> 4) & 3, hf = (blk >> 6) & 1, rest = blk >> 7;
        const int u = rest % (SK / 32), hd = rest / (SK / 32);
        const int byte = 64 * hd + 4 * dg;
        uint32_t x[4], o[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int key = 32 * u + 16 * hf + 2 * qq + (e & 1) + 8 * (e >> 1);
          x[e] = *reinterpret_cast<const uint32_t*>(Vt + key * 128 + (((byte >> 4) ^ (((key >> 1) & 3) << 1)) << 4) +
                                                    (byte & 15));
        }
        transpose4(x[0], x[1], x[2], x[3], o);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int dim = 4 * dg + j;
          const int un = (4 * u + qq) ^ (((dim >> 1) & 1) << 2);
          *reinterpret_cast<uint32_t*>(VT + (hd * 64 + dim) * SK + un * 8 + 4 * hf) = o[j];
        }
      }
    }
    __syncthreads();
    const int kbs = cs + SK * it, kb = kbs + ks * KPW;

    float sc[NTS][4];
#pragma unroll
    for (int t = 0; t < NTS; ++t) {
#pragma unroll
      for (int j = 0; j < 4; ++j) sc[t][j] = 0.f;
      const int key = ks * KPW + 8 * t + g;
      const uint4 kv = *reinterpret_cast<const uint4*>(Kt + key * 128 + (((4 * hh + q) ^ ((key & 1) << 2)) * 16));
      mma_e4m3(sc[t], qx.x, qy.x, qx.y, qy.y, kv.x, kv.y);
      mma_e4m3(sc[t], qx.z, qy.z, qx.w, qy.w, kv.z, kv.w);
#pragma unroll
      for (int j = 0; j < 4; ++j) sc[t][j] *= sc_mul;
    }
    const bool edge = kbs + SK > ce || kbs + SK > hi - L + 1 || (window > 0 && kbs < lo + L - 1);
    if (edge) {
#pragma unroll
      for (int t = 0; t < NTS; ++t)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int pos = kb + 8 * t + 2 * q + e;
          const bool okk = pos < ce;
          const bool oka = okk && va && pos <= qpa && (window == 0 || pos > qpa - window);
          const bool okb = okk && vb && pos <= qpb && (window == 0 || pos > qpb - window);
          sc[t][e] = oka ? sc[t][e] : -INFINITY;
          sc[t][2 + e] = okb ? sc[t][2 + e] : -INFINITY;
        }
    }
    if (!vb) {  // padding rows (q_len 1): masked everywhere, unmasked they slow the kernel
#pragma unroll
      for (int t = 0; t < NTS; ++t) sc[t][2] = sc[t][3] = -INFINITY;
    }
    float mxa = -INFINITY, mxb = -INFINITY;
#pragma unroll
    for (int t = 0; t < NTS; ++t) {
      mxa = fmaxf(mxa, fmaxf(sc[t][0], sc[t][1]));
      mxb = fmaxf(mxb, fmaxf(sc[t][2], sc[t][3]));
    }
#pragma unroll
    for (int o = 1; o <= 2; o <<= 1) {
      mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, o));
      mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, o));
    }
    const float mna = fmaxf(ma, mxa), mnb = fmaxf(mb, mxb);
    const float sa = mna == -INFINITY ? 0.f : mna, sb = mnb == -INFINITY ? 0.f : mnb;
    const float aa = exp2f(ma - sa), ab = exp2f(mb - sb);
    ma = mna;
    mb = mnb;
    la *= aa;
    lb *= ab;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      acc[i][0] *= aa;
      acc[i][1] *= aa;
      acc[i][2] *= ab;
      acc[i][3] *= ab;
    }
#pragma unroll
    for (int t = 0; t < NTS; ++t) {
      sc[t][0] = exp2f(sc[t][0] - sa);
      sc[t][1] = exp2f(sc[t][1] - sa);
      sc[t][2] = exp2f(sc[t][2] - sb);
      sc[t][3] = exp2f(sc[t][3] - sb);
      la += sc[t][0] + sc[t][1];
      lb += sc[t][2] + sc[t][3];
    }
#pragma unroll
    for (int u = 0; u < KSP; ++u) {
      const float(*pt4)[4] = sc + 4 * u;
      const uint32_t a0 = pack_e4m3x4(pt4[0][0], pt4[0][1], pt4[1][0], pt4[1][1]);
      const uint32_t a1 = pack_e4m3x4(pt4[0][2], pt4[0][3], pt4[1][2], pt4[1][3]);
      const uint32_t a2 = pack_e4m3x4(pt4[2][0], pt4[2][1], pt4[3][0], pt4[3][1]);
      const uint32_t a3 = pack_e4m3x4(pt4[2][2], pt4[2][3], pt4[3][2], pt4[3][3]);
      const int uw = (ks * KPW) / 32 + u;  // the stage's k-step
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        const int dim = 8 * nt + g;
        const uint2 bv = *reinterpret_cast<const uint2*>(VT + (hh * 64 + dim) * SK +
                                                         (((4 * uw + q) ^ (((dim >> 1) & 1) << 2)) * 8));
        mma_e4m3(acc[nt], a0, a1, a2, a3, bv.x, bv.y);
      }
    }
  }
  cp_wait<0>();

#pragma unroll
  for (int o = 1; o <= 2; o <<= 1) {
    la += __shfl_xor_sync(0xffffffffu, la, o);
    lb += __shfl_xor_sync(0xffffffffu, lb, o);
  }
  if constexpr (KS > 1) {
    __syncthreads();
    float* buf = reinterpret_cast<float*>(smem) + (unit * 32 + lane) * 36;
    if (ks == 1) {
      buf[0] = ma;
      buf[1] = mb;
      buf[2] = la;
      buf[3] = lb;
#pragma unroll
      for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) buf[4 + i * 4 + j] = acc[i][j];
    }
    __syncthreads();
    if (ks == 1) return;
    const float m2a = buf[0], m2b = buf[1];
    const float Ma = fmaxf(ma, m2a), Mb = fmaxf(mb, m2b);
    const float Sa = Ma == -INFINITY ? 0.f : Ma, Sb = Mb == -INFINITY ? 0.f : Mb;
    const float w1a = exp2f(ma - Sa), w2a = exp2f(m2a - Sa), w1b = exp2f(mb - Sb), w2b = exp2f(m2b - Sb);
    la = la * w1a + buf[2] * w2a;
    lb = lb * w1b + buf[3] * w2b;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      acc[i][0] = acc[i][0] * w1a + buf[4 + i * 4 + 0] * w2a;
      acc[i][1] = acc[i][1] * w1a + buf[4 + i * 4 + 1] * w2a;
      acc[i][2] = acc[i][2] * w1b + buf[4 + i * 4 + 2] * w2b;
      acc[i][3] = acc[i][3] * w1b + buf[4 + i * 4 + 3] * w2b;
    }
    ma = Ma;
    mb = Mb;
  }
  const float vs = vscale[0];
  const size_t rowa = row_base + hh * RH + ra, rowb = row_base + hh * RH + rb;
  if (q == 0) {
    ws_m[rowa] = va ? ma : -INFINITY;
    ws_l[rowa] = va ? la : 0.f;
    ws_m[rowb] = vb ? mb : -INFINITY;
    ws_l[rowb] = vb ? lb : 0.f;
  }
  // Output columns: n-tile nt holds dims 8nt + 2q, + 1.
#pragma unroll
  for (int nt = 0; nt < 8; ++nt) {
    *reinterpret_cast<float2*>(ws_o + rowa * D + 8 * nt + 2 * q) = make_float2(acc[nt][0] * vs, acc[nt][1] * vs);
    *reinterpret_cast<float2*>(ws_o + rowb * D + 8 * nt + 2 * q) = make_float2(acc[nt][2] * vs, acc[nt][3] * vs);
  }
}

void attn8(torch::Tensor q8, torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor seqlens,
           torch::Tensor kscale, torch::Tensor vscale, torch::Tensor ws_o, torch::Tensor ws_m, torch::Tensor ws_l,
           double qscale, int64_t L, int64_t HKV, int64_t page, int64_t window, int64_t CH, int64_t NCH,
           int64_t batch) {
  constexpr int STG = 4, SK = 64;
  const size_t smem = STG * 256 * SK + 128 * SK + 4 * CH;  // stage ring, VT, the chunk's cache rows
  TORCH_CHECK(smem <= 200 * 1024, "attn8: chunk too large");
  dim3 grid(batch * (HKV / 2), NCH);
  auto stream = at::cuda::getCurrentCUDAStream();
#define LAUNCH8(LL)                                                                                            \
  {                                                                                                           \
    auto k = attn8_kernel<LL, STG, SK>;                                                                        \
    static bool set = false;                                                                                   \
    if (!set) {                                                                                                \
      cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024);                        \
      set = true;                                                                                              \
    }                                                                                                          \
    k<<<grid, LL == 1 ? 128 : 256, smem, stream>>>(                                                            \
        reinterpret_cast<const uint8_t*>(q8.data_ptr()), reinterpret_cast<const uint8_t*>(kc.data_ptr()),      \
        reinterpret_cast<const uint8_t*>(vc.data_ptr()), pt.data_ptr<int>(), seqlens.data_ptr<int>(),           \
        kscale.data_ptr<float>(), vscale.data_ptr<float>(), ws_o.data_ptr<float>(), ws_m.data_ptr<float>(),    \
        ws_l.data_ptr<float>(), (float)qscale, (int)HKV, (int)pt.size(1), (int)page, (int)window, (int)CH,     \
        (int)NCH);                                                                                             \
  }
  if (L == 1) LAUNCH8(1) else if (L == 8) LAUNCH8(8) else TORCH_CHECK(false, "attn8: q_len must be 1 or 8");
}

// ---- verify blocks (q_len 8): wgmma ----
__device__ __forceinline__ uint64_t make_desc(const void* smem, int layout, int sbo) {  // K-major, swizzled
  uint64_t d = 0;
  d |= (uint64_t)(((uint32_t)__cvta_generic_to_shared(smem) & 0x3FFFFu) >> 4);
  d |= (uint64_t)(sbo >> 4) << 32;
  d |= (uint64_t)layout << 62;  // 1: 128 B swizzle, 2: 64 B
  return d;
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void wg_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(N) : "memory");
}
__device__ __forceinline__ void fence_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void reg_fence(float* r) {  // pins the accumulators' reads and writes around a wgmma
#pragma unroll
  for (int i = 0; i < N; ++i) asm volatile("" : "+f"(r[i])::"memory");
}
template <int N>
__device__ __forceinline__ void reg_fence(uint32_t* r) {  // the A fragments: kept in place until the wgmma completes
#pragma unroll
  for (int i = 0; i < N; ++i) asm volatile("" : "+r"(r[i])::"memory");
}
"""
_attn_CUDA_SRC += _attn_wgmma_fn(64) + _attn_wgmma_fn(128) + r"""
// Verify blocks: one CTA per (request, HEADS KV heads, key chunk), one warpgroup per KV head, whose 64 query rows
// (8 tokens x 8 heads; row r = 8 j + g) are one m64 tile. Per stage of SK keys: S = Q K^T as m64nSKk32 E4M3 wgmmas
// (A = Q in registers; B = the stage's key tile: rows of 64 HEADS bytes, swizzled by their width, head hh at
// + 64 hh), softmax in registers, the values transposed once by all threads into VT (per head: 64 dim rows of SK
// bytes, swizzled by 64 B (128 B at SK = 128), keys in the P-fragment slot order: slot 32u + 16h + 4q + e <-> key
// 32u + 16h + 2q + (e & 1) + 8 (e >> 1)), then O += P V as m64n64k32 wgmmas with P (E4M3) from registers.
// HEADS = 2 reads whole 128 B cache rows (the faster gather) with two CTAs per SM; HEADS = 1 reads 64 B rows (the
// other head's CTA reads the same lines from L2) with four CTAs per SM, which suits the short sliding windows.
template <int STG, int SK, int HEADS>
__global__ void __launch_bounds__(128 * HEADS, HEADS == 1 ? 3 : 1) attnwg_kernel(
    const uint8_t* __restrict__ q8, const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,
    const int* __restrict__ pt, const int* __restrict__ seqlens, const float* __restrict__ kscale,
    const float* __restrict__ vscale, float* __restrict__ ws_o, float* __restrict__ ws_m, float* __restrict__ ws_l,
    float qscale, int HKV, int max_pages, int page, int window, int CH, int NCH) {
  constexpr int L = 8, G = 8, D = 64, RH = 64, NT = 128 * HEADS, RB = 64 * HEADS, CPR = RB / 16;  // row bytes, chunks
  constexpr int TILE = SK * 2 * RB, KSP = SK / 32, NS = SK / 2;  // NS: S accumulator registers
  static_assert(SK == 64 || SK == 128, "stage of 64 or 128 keys");
  constexpr int KLAYOUT = HEADS == 1 ? 2 : 1, KSBO = HEADS == 1 ? 512 : 1024;
  constexpr int VLAYOUT = SK == 64 ? 2 : 1, VSBO = SK == 64 ? 512 : 1024;
  extern __shared__ uint4 smem4[];
  uint8_t* smem = reinterpret_cast<uint8_t*>(smem4);
  smem += (1024 - ((uint32_t)__cvta_generic_to_shared(smem) & 1023)) & 1023;
  uint8_t* VT = smem + STG * TILE;  // two buffers per head: the values of stage it + 1 are transposed during stage it
  int* rows = reinterpret_cast<int*>(VT + HEADS * 2 * 64 * SK);

  const int wq = HEADS == 1 ? threadIdx.x >> 5 : (threadIdx.x >> 5) & 3, lane = threadIdx.x & 31, g = lane >> 2, q = lane & 3;
  const int hl = HEADS == 1 ? 0 : threadIdx.x >> 7;  // this warpgroup's head within the CTA
  const int bh = blockIdx.x, c = blockIdx.y;
  const int b = bh / (HKV / HEADS), hbase = (bh % (HKV / HEADS)) * HEADS, h = hbase + hl;
  const int hh = HEADS == 1 ? (h & 1) : hl;
  const int seq = seqlens[b], hi = seq + L;
  const int lo = window > 0 ? max(seq - (window - 1), 0) : 0;
  const int cs = lo + c * CH, ce = min(cs + CH, hi);
  constexpr int R = 2 * RH;
  const size_t row_base = ((size_t)(b * (HKV / 2) + h / 2) * NCH + c) * R + hh * RH;  // this head's 64 rows
  if (cs >= hi) {
    for (int i = HEADS == 1 ? threadIdx.x : threadIdx.x & 127; i < RH; i += 128) {
      ws_m[row_base + i] = -INFINITY;
      ws_l[row_base + i] = 0.f;
    }
    return;
  }

  // Rows ra = 16 wq + g (token 2 wq, head g) and rb = ra + 8 (token 2 wq + 1). Q fragments: k-step s, lane q holds
  // dims 32 s + 4 q .. + 3 (a0 / a1) and 32 s + 16 + 4 q .. (a2 / a3).
  const int ra = 16 * wq + g, rb = ra + 8;
  const uint8_t* qa_p = q8 + ((size_t)(b * L + ra / G) * (HKV * G) + h * G + ra % G) * D + 4 * q;
  const uint8_t* qb_p = q8 + ((size_t)(b * L + rb / G) * (HKV * G) + h * G + rb % G) * D + 4 * q;
  const int qpa = seq + ra / G, qpb = seq + rb / G;
  const float sc_mul = qscale * kscale[0];

  const int nsteps = (ce - cs + SK - 1) / SK;
  // The chunk's cache rows go to shared memory in two parts: the first STG - 1 stages' before their copies are
  // issued, the rest while those copies are in flight (eight page-table loads in flight per thread).
  auto load_rows = [&](int from, int to) {
    for (int i0 = from + threadIdx.x; i0 < to; i0 += 8 * NT) {
      int v[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int pos = cs + i0 + j * NT;
        v[j] = pos < ce ? pt[(size_t)b * max_pages + pos / page] * page + pos % page : 0;
      }
#pragma unroll
      for (int j = 0; j < 8; ++j)
        if (i0 + j * NT < to) rows[i0 + j * NT] = v[j];
    }
  };
  const int first = min(ce - cs, SK * (STG - 1));
  load_rows(0, first);
  __syncthreads();
  constexpr int PER = SK / 16;
  // Key rows: 16 B chunk ch of row k at ch ^ ((k >> 1) & 3) (64 B rows, the 64 B swizzle) or ch ^ (k & 7) (128 B
  // rows, the 128 B swizzle). Value rows (read by the transpose only): ch ^ ((k >> 1) & 3) or ch ^ (((k >> 1) & 3) << 1).
  auto issue = [&](int stage, int kb) {
    uint8_t* tile = smem + stage * TILE;
#pragma unroll
    for (int i = 0; i < PER; ++i) {
      const int idx = threadIdx.x + i * NT;
      const int key = idx / (2 * CPR), part = (idx / CPR) & 1, ch = idx % CPR;
      const bool ok = kb + key < ce;
      const int row = ok ? rows[kb + key - cs] : 0;
      const uint8_t* src = (part ? vc : kc) + ((long long)row * HKV + hbase) * D + ch * 16;
      const int sw = HEADS == 1 ? (ch ^ ((key >> 1) & 3)) : part ? (ch ^ (((key >> 1) & 3) << 1)) : (ch ^ (key & 7));
      cp16(tile + part * (SK * RB) + key * RB + (sw << 4), src, ok);
    }
  };
#pragma unroll
  for (int s = 0; s < STG - 1; ++s) {
    if (s < nsteps) issue(s, cs + SK * s);
    cp_commit();
  }
  load_rows(first, ce - cs);  // visible to every thread after the barrier below
  // VT: block (u, half, q, dim group) = 4 dims x slots 4 q .. + 3 of half `half` of k-step u of stage `st`.
  auto transpose = [&](int st) {
    const uint8_t* Vt = smem + (st % STG) * TILE + SK * RB;
    uint8_t* vt = VT + (st & 1) * (HEADS * 64 * SK);
#pragma unroll
    // Lanes of a warp spread over (q, half, k-step) and two dim groups, so their 4-byte reads and writes land in
    // distinct banks (two-way at most) instead of 16 lanes hitting dims 256 B apart.
    for (int blk = threadIdx.x; blk < SK * 4 * HEADS; blk += NT) {
      const int qq = blk & 3, hf = (blk >> 2) & 1, u = (blk >> 3) % KSP, rest = (blk >> 3) / KSP;
      const int dg = rest & 15, hd = rest >> 4;
      const int byte = 64 * hd + 4 * dg;
      uint32_t x[4], o[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int key = 32 * u + 16 * hf + 2 * qq + (e & 1) + 8 * (e >> 1);
        const int sw = HEADS == 1 ? ((byte >> 4) ^ ((key >> 1) & 3)) : ((byte >> 4) ^ (((key >> 1) & 3) << 1));
        x[e] = *reinterpret_cast<const uint32_t*>(Vt + key * RB + (sw << 4) + (byte & 15));
      }
      transpose4(x[0], x[1], x[2], x[3], o);
      const int slot = 32 * u + 16 * hf + 4 * qq, chunk = slot >> 4;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int dim = 4 * dg + j;
        const int sw = SK == 64 ? (chunk ^ ((dim >> 1) & 3)) : (chunk ^ (dim & 7));
        *reinterpret_cast<uint32_t*>(vt + (hd * 64 + dim) * SK + sw * 16 + (slot & 15)) = o[j];
      }
    }
    fence_async();  // the writes above, for the wgmma reads after the next barrier
  };
  cp_wait<STG - 2>();  // stage 0
  fence_async();
  __syncthreads();
  transpose(0);

  float ma = -INFINITY, mb = -INFINITY, la = 0.f, lb = 0.f;
  float acc[32];
#pragma unroll
  for (int i = 0; i < 32; ++i) acc[i] = 0.f;
  float sc0[NS], sc1[NS];  // the scores of two consecutive stages: S of stage it + 1 is computed during stage it
  uint32_t pa[KSP][4];     // P of the stage whose P V is in flight
  uint32_t qa[2][4];       // Q, reloaded every stage (L1 hits) and kept live until its wgmma completes
  auto issue_s = [&](int st, float* sc) {
#pragma unroll
    for (int s = 0; s < 2; ++s) {
      qa[s][0] = __ldg(reinterpret_cast<const uint32_t*>(qa_p + 32 * s));
      qa[s][1] = __ldg(reinterpret_cast<const uint32_t*>(qb_p + 32 * s));
      qa[s][2] = __ldg(reinterpret_cast<const uint32_t*>(qa_p + 32 * s + 16));
      qa[s][3] = __ldg(reinterpret_cast<const uint32_t*>(qb_p + 32 * s + 16));
    }
    const uint8_t* Kt = smem + (st % STG) * TILE + 64 * hl;
    const uint64_t kd0 = make_desc(Kt, KLAYOUT, KSBO), kd1 = make_desc(Kt + 32, KLAYOUT, KSBO);  // before the fence
    reg_fence<NS>(sc);
    wg_fence();
    if constexpr (SK == 64) {
      wgmma_64(sc, qa[0], kd0, 0);
      wgmma_64(sc, qa[1], kd1, 1);
    } else {
      wgmma_128(sc, qa[0], kd0, 0);
      wgmma_128(sc, qa[1], kd1, 1);
    }
    wg_commit();
  };
  issue_s(0, sc0);

  // Stage it: its scores are in `sc` (complete), its values in VT; the next stage's scores are computed and its
  // values transposed meanwhile, and its own P V runs into the next stage. One barrier per stage: after it, stage
  // it + 1 has landed (for its transpose and S), VT of stage it is visible, and the ring slot of stage it + STG - 1
  // (last read by stage it - 1) is free.
  auto stage = [&](int it, float* sc, float* nxt) {
    cp_wait<STG - 3>();
    fence_async();
    __syncthreads();
    wg_wait<0>();  // S of this stage, and the previous stage's P V (acc, qa and pa free)
    reg_fence<NS>(sc);
    reg_fence<32>(acc);
    reg_fence<4>(qa[0]);
    reg_fence<4>(qa[1]);
#pragma unroll
    for (int u = 0; u < KSP; ++u) reg_fence<4>(pa[u]);
    {
      const int nx = it + STG - 1;
      if (nx < nsteps) issue(nx % STG, cs + SK * nx);
      cp_commit();
    }
    if (it + 1 < nsteps) {
      issue_s(it + 1, nxt);
      transpose(it + 1);
    }
    const uint8_t* vt = VT + ((it & 1) * HEADS + hl) * (64 * SK);
    const int kbs = cs + SK * it;
    // Element (row, key kbs + 8 i + 2 q + e) is sc[4 i + e] (row ra) and sc[4 i + 2 + e] (row rb); the scale
    // sc_mul (> 0) is applied in the exponent.
    const bool edge = kbs + SK > ce || kbs + SK > hi - L + 1 || (window > 0 && kbs < lo + L - 1);
    if (edge) {
#pragma unroll
      for (int i = 0; i < NS / 4; ++i)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int pos = kbs + 8 * i + 2 * q + e;
          const bool okk = pos < ce;
          const bool oka = okk && pos <= qpa && (window == 0 || pos > qpa - window);
          const bool okb = okk && pos <= qpb && (window == 0 || pos > qpb - window);
          sc[4 * i + e] = oka ? sc[4 * i + e] : -INFINITY;
          sc[4 * i + 2 + e] = okb ? sc[4 * i + 2 + e] : -INFINITY;
        }
    }
    float mxa = -INFINITY, mxb = -INFINITY;
#pragma unroll
    for (int i = 0; i < NS / 4; ++i) {
      mxa = fmaxf(mxa, fmaxf(sc[4 * i], sc[4 * i + 1]));
      mxb = fmaxf(mxb, fmaxf(sc[4 * i + 2], sc[4 * i + 3]));
    }
#pragma unroll
    for (int o = 1; o <= 2; o <<= 1) {
      mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, o));
      mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, o));
    }
    const float mna = fmaxf(ma, mxa * sc_mul), mnb = fmaxf(mb, mxb * sc_mul);
    const float sa = mna == -INFINITY ? 0.f : mna, sb = mnb == -INFINITY ? 0.f : mnb;
    const float aa = exp2f(ma - sa), ab = exp2f(mb - sb);
    ma = mna;
    mb = mnb;
    la *= aa;
    lb *= ab;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      acc[4 * i] *= aa;
      acc[4 * i + 1] *= aa;
      acc[4 * i + 2] *= ab;
      acc[4 * i + 3] *= ab;
    }
#pragma unroll
    for (int i = 0; i < NS / 4; ++i) {
      sc[4 * i] = exp2f(fmaf(sc[4 * i], sc_mul, -sa));
      sc[4 * i + 1] = exp2f(fmaf(sc[4 * i + 1], sc_mul, -sa));
      sc[4 * i + 2] = exp2f(fmaf(sc[4 * i + 2], sc_mul, -sb));
      sc[4 * i + 3] = exp2f(fmaf(sc[4 * i + 3], sc_mul, -sb));
      la += sc[4 * i] + sc[4 * i + 1];
      lb += sc[4 * i + 2] + sc[4 * i + 3];
    }
#pragma unroll
    for (int u = 0; u < KSP; ++u) {
      const float* p4 = sc + 16 * u;
      pa[u][0] = pack_e4m3x4(p4[0], p4[1], p4[4], p4[5]);
      pa[u][1] = pack_e4m3x4(p4[2], p4[3], p4[6], p4[7]);
      pa[u][2] = pack_e4m3x4(p4[8], p4[9], p4[12], p4[13]);
      pa[u][3] = pack_e4m3x4(p4[10], p4[11], p4[14], p4[15]);
    }
    uint64_t vd[KSP];
#pragma unroll
    for (int u = 0; u < KSP; ++u) vd[u] = make_desc(vt + 32 * u, VLAYOUT, VSBO);
    reg_fence<32>(acc);
    wg_fence();
#pragma unroll
    for (int u = 0; u < KSP; ++u) wgmma_64(acc, pa[u], vd[u], 1);
    wg_commit();
  };
  for (int it = 0; it < nsteps; it += 2) {
    stage(it, sc0, sc1);
    if (it + 1 < nsteps) stage(it + 1, sc1, sc0);
  }
  wg_wait<0>();
  reg_fence<32>(acc);
  reg_fence<4>(qa[0]);
  reg_fence<4>(qa[1]);
#pragma unroll
  for (int u = 0; u < KSP; ++u) reg_fence<4>(pa[u]);
  cp_wait<0>();

#pragma unroll
  for (int o = 1; o <= 2; o <<= 1) {
    la += __shfl_xor_sync(0xffffffffu, la, o);
    lb += __shfl_xor_sync(0xffffffffu, lb, o);
  }
  const float vs = vscale[0];
  const size_t rowa = row_base + ra, rowb = row_base + rb;
  if (q == 0) {
    ws_m[rowa] = ma;
    ws_l[rowa] = la;
    ws_m[rowb] = mb;
    ws_l[rowb] = lb;
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    *reinterpret_cast<float2*>(ws_o + rowa * D + 8 * i + 2 * q) = make_float2(acc[4 * i] * vs, acc[4 * i + 1] * vs);
    *reinterpret_cast<float2*>(ws_o + rowb * D + 8 * i + 2 * q) = make_float2(acc[4 * i + 2] * vs, acc[4 * i + 3] * vs);
  }
}

void attnwg(torch::Tensor q8, torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor seqlens,
            torch::Tensor kscale, torch::Tensor vscale, torch::Tensor ws_o, torch::Tensor ws_m, torch::Tensor ws_l,
            double qscale, int64_t HKV, int64_t page, int64_t window, int64_t CH, int64_t NCH, int64_t batch,
            int64_t SK, int64_t HEADS) {
  constexpr int STG = 5;
  const size_t smem = STG * 128 * HEADS * SK + HEADS * 128 * SK + 4 * CH + 1024;  // ring, VT x 2, rows, alignment
  TORCH_CHECK(smem <= 200 * 1024, "attnwg: chunk too large");
  TORCH_CHECK(HEADS == 1 || HEADS == 2, "attnwg: 1 or 2 KV heads per CTA");
  dim3 grid(batch * (HKV / HEADS), NCH);
  auto stream = at::cuda::getCurrentCUDAStream();
#define LAUNCHWG(SKK, HH)                                                                                      \
  {                                                                                                           \
    auto k = attnwg_kernel<STG, SKK, HH>;                                                                      \
    static bool set = false;                                                                                   \
    if (!set) {                                                                                                \
      cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024);                        \
      set = true;                                                                                              \
    }                                                                                                          \
    k<<<grid, 128 * HH, smem, stream>>>(                                                                       \
        reinterpret_cast<const uint8_t*>(q8.data_ptr()), reinterpret_cast<const uint8_t*>(kc.data_ptr()),      \
        reinterpret_cast<const uint8_t*>(vc.data_ptr()), pt.data_ptr<int>(), seqlens.data_ptr<int>(),           \
        kscale.data_ptr<float>(), vscale.data_ptr<float>(), ws_o.data_ptr<float>(), ws_m.data_ptr<float>(),    \
        ws_l.data_ptr<float>(), (float)qscale, (int)HKV, (int)pt.size(1), (int)page, (int)window, (int)CH,     \
        (int)NCH);                                                                                             \
  }
  TORCH_CHECK(SK == 64 || SK == 128, "attnwg: stage 64 or 128 keys");
  if (SK == 64 && HEADS == 1) LAUNCHWG(64, 1) else if (SK == 64) LAUNCHWG(64, 2) else if (HEADS == 1) LAUNCHWG(128, 1) else LAUNCHWG(128, 2)
}

// Verify blocks, warp-specialized: one CTA per (request, KV head, key chunk) of a producer warpgroup (threads
// 128-255: cp.async copies of 64 B key/value rows and the value transposes, into three VT buffers) and a consumer
// warpgroup (threads 0-127: S = Q K^T wgmmas issued one stage ahead, softmax, P V wgmmas), as attnwg_kernel with
// HEADS = 1. One barrier per stage: after barrier it, stage it + 1 has landed, VT of stage it is visible, and the
// ring slot of stage it + STG - 1 is free. (setmaxnreg register moves fault with cudaErrorInvalidAddressSpace here;
// the consumer fits the launch's 128 registers without spills, so none are needed.)
template <int STG, int SK>
__global__ void __launch_bounds__(256, 2) attnpc_kernel(
    const uint8_t* __restrict__ q8, const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,
    const int* __restrict__ pt, const int* __restrict__ seqlens, const float* __restrict__ kscale,
    const float* __restrict__ vscale, float* __restrict__ ws_o, float* __restrict__ ws_m, float* __restrict__ ws_l,
    float qscale, int HKV, int max_pages, int page, int window, int CH, int NCH) {
  constexpr int L = 8, G = 8, D = 64, RH = 64, NT = 256, RB = 64, CPR = RB / 16;
  constexpr int TILE = SK * 2 * RB, KSP = SK / 32, NS = SK / 2;
  static_assert(SK == 64 || SK == 128, "stage of 64 or 128 keys");
  constexpr int KLAYOUT = 2, KSBO = 512;
  constexpr int VLAYOUT = SK == 64 ? 2 : 1, VSBO = SK == 64 ? 512 : 1024;
  extern __shared__ uint4 smem4[];
  uint8_t* smem = reinterpret_cast<uint8_t*>(smem4);
  smem += (1024 - ((uint32_t)__cvta_generic_to_shared(smem) & 1023)) & 1023;
  uint8_t* VT = smem + STG * TILE;  // three buffers: stage it + 1 is transposed while P V of stage it may run
  int* rows = reinterpret_cast<int*>(VT + 3 * 64 * SK);

  const bool producer = threadIdx.x >= 128;
  const int tid = threadIdx.x & 127, wq = tid >> 5, lane = tid & 31, g = lane >> 2, q = lane & 3;
  const int bh = blockIdx.x, c = blockIdx.y;
  const int b = bh / HKV, h = bh % HKV, hh = h & 1;
  const int seq = seqlens[b], hi = seq + L;
  const int lo = window > 0 ? max(seq - (window - 1), 0) : 0;
  const int cs = lo + c * CH, ce = min(cs + CH, hi);
  constexpr int R = 2 * RH;
  const size_t row_base = ((size_t)(b * (HKV / 2) + h / 2) * NCH + c) * R + hh * RH;
  if (cs >= hi) {
    for (int i = threadIdx.x; i < RH; i += NT) {
      ws_m[row_base + i] = -INFINITY;
      ws_l[row_base + i] = 0.f;
    }
    return;
  }
  const int nsteps = (ce - cs + SK - 1) / SK;
  auto load_rows = [&](int from, int to) {
    for (int i0 = from + threadIdx.x; i0 < to; i0 += 8 * NT) {
      int v[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int pos = cs + i0 + j * NT;
        v[j] = pos < ce ? pt[(size_t)b * max_pages + pos / page] * page + pos % page : 0;
      }
#pragma unroll
      for (int j = 0; j < 8; ++j)
        if (i0 + j * NT < to) rows[i0 + j * NT] = v[j];
    }
  };
  const int first = min(ce - cs, SK * (STG - 1));
  load_rows(0, first);
  __syncthreads();  // P1

  if (producer) {
    constexpr int PER = SK / 16;
    auto issue = [&](int stage, int kb) {
      uint8_t* tile = smem + stage * TILE;
#pragma unroll
      for (int i = 0; i < PER; ++i) {
        const int idx = tid + i * 128;
        const int key = idx / (2 * CPR), part = (idx / CPR) & 1, ch = idx % CPR;
        const bool ok = kb + key < ce;
        const int row = ok ? rows[kb + key - cs] : 0;
        const uint8_t* src = (part ? vc : kc) + ((long long)row * HKV + h) * D + ch * 16;
        cp16(tile + part * (SK * RB) + key * RB + ((ch ^ ((key >> 1) & 3)) << 4), src, ok);
      }
    };
    auto transpose = [&](int st) {
      const uint8_t* Vt = smem + (st % STG) * TILE + SK * RB;
      uint8_t* vt = VT + (st % 3) * (64 * SK);
#pragma unroll
      for (int blk = tid; blk < SK * 4; blk += 128) {
        const int qq = blk & 3, hf = (blk >> 2) & 1, u = (blk >> 3) % KSP, dg = ((blk >> 3) / KSP) & 15;
        const int byte = 4 * dg;
        uint32_t x[4], o[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int key = 32 * u + 16 * hf + 2 * qq + (e & 1) + 8 * (e >> 1);
          x[e] = *reinterpret_cast<const uint32_t*>(Vt + key * RB + (((byte >> 4) ^ ((key >> 1) & 3)) << 4) + (byte & 15));
        }
        transpose4(x[0], x[1], x[2], x[3], o);
        const int slot = 32 * u + 16 * hf + 4 * qq, chunk = slot >> 4;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int dim = 4 * dg + j;
          const int sw = SK == 64 ? (chunk ^ ((dim >> 1) & 3)) : (chunk ^ (dim & 7));
          *reinterpret_cast<uint32_t*>(vt + dim * SK + sw * 16 + (slot & 15)) = o[j];
        }
      }
    };
#pragma unroll
    for (int s = 0; s < STG - 1; ++s) {
      if (s < nsteps) issue(s, cs + SK * s);
      cp_commit();
    }
    load_rows(first, ce - cs);  // both warpgroups, before P2 (by the producer alone after P2 it delays the ring)
    cp_wait<STG - 3>();  // stages 0 and 1
    fence_async();
    __syncthreads();  // P2
    transpose(0);
    fence_async();
    for (int it = 0; it < nsteps; ++it) {
      __syncthreads();  // B(it)
      {
        const int nx = it + STG - 1;
        if (nx < nsteps) issue(nx % STG, cs + SK * nx);
        cp_commit();
      }
      cp_wait<STG - 3>();  // stage it + 2
      if (it + 1 < nsteps) transpose(it + 1);
      fence_async();
    }
    cp_wait<0>();
    return;
  }

  const int ra = 16 * wq + g, rb = ra + 8;
  const uint8_t* qa_p = q8 + ((size_t)(b * L + ra / G) * (HKV * G) + h * G + ra % G) * D + 4 * q;
  const uint8_t* qb_p = q8 + ((size_t)(b * L + rb / G) * (HKV * G) + h * G + rb % G) * D + 4 * q;
  const int qpa = seq + ra / G, qpb = seq + rb / G;
  const float sc_mul = qscale * kscale[0];
  float ma = -INFINITY, mb = -INFINITY, la = 0.f, lb = 0.f;
  float acc[32];
#pragma unroll
  for (int i = 0; i < 32; ++i) acc[i] = 0.f;
  float sc0[NS], sc1[NS];
  uint32_t pa[KSP][4];
  uint32_t qa[2][4];
  auto issue_s = [&](int st, float* sc) {
#pragma unroll
    for (int s = 0; s < 2; ++s) {
      qa[s][0] = __ldg(reinterpret_cast<const uint32_t*>(qa_p + 32 * s));
      qa[s][1] = __ldg(reinterpret_cast<const uint32_t*>(qb_p + 32 * s));
      qa[s][2] = __ldg(reinterpret_cast<const uint32_t*>(qa_p + 32 * s + 16));
      qa[s][3] = __ldg(reinterpret_cast<const uint32_t*>(qb_p + 32 * s + 16));
    }
    const uint8_t* Kt = smem + (st % STG) * TILE;
    const uint64_t kd0 = make_desc(Kt, KLAYOUT, KSBO), kd1 = make_desc(Kt + 32, KLAYOUT, KSBO);  // before the fence
    reg_fence<NS>(sc);
    wg_fence();
    if constexpr (SK == 64) {
      wgmma_64(sc, qa[0], kd0, 0);
      wgmma_64(sc, qa[1], kd1, 1);
    } else {
      wgmma_128(sc, qa[0], kd0, 0);
      wgmma_128(sc, qa[1], kd1, 1);
    }
    wg_commit();
  };
  load_rows(first, ce - cs);
  __syncthreads();  // P2: stages 0 and 1 landed
  issue_s(0, sc0);
  auto stage = [&](int it, float* sc, float* nxt) {
    __syncthreads();  // B(it)
    wg_wait<0>();
    reg_fence<NS>(sc);
    reg_fence<32>(acc);
    reg_fence<4>(qa[0]);
    reg_fence<4>(qa[1]);
#pragma unroll
    for (int u = 0; u < KSP; ++u) reg_fence<4>(pa[u]);
    if (it + 1 < nsteps) issue_s(it + 1, nxt);
    const uint8_t* vt = VT + (it % 3) * (64 * SK);
    const int kbs = cs + SK * it;
    const bool edge = kbs + SK > ce || kbs + SK > hi - L + 1 || (window > 0 && kbs < lo + L - 1);
    if (edge) {
#pragma unroll
      for (int i = 0; i < NS / 4; ++i)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int pos = kbs + 8 * i + 2 * q + e;
          const bool okk = pos < ce;
          const bool oka = okk && pos <= qpa && (window == 0 || pos > qpa - window);
          const bool okb = okk && pos <= qpb && (window == 0 || pos > qpb - window);
          sc[4 * i + e] = oka ? sc[4 * i + e] : -INFINITY;
          sc[4 * i + 2 + e] = okb ? sc[4 * i + 2 + e] : -INFINITY;
        }
    }
    float mxa = -INFINITY, mxb = -INFINITY;
#pragma unroll
    for (int i = 0; i < NS / 4; ++i) {
      mxa = fmaxf(mxa, fmaxf(sc[4 * i], sc[4 * i + 1]));
      mxb = fmaxf(mxb, fmaxf(sc[4 * i + 2], sc[4 * i + 3]));
    }
#pragma unroll
    for (int o = 1; o <= 2; o <<= 1) {
      mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, o));
      mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, o));
    }
    const float mna = fmaxf(ma, mxa * sc_mul), mnb = fmaxf(mb, mxb * sc_mul);
    const float sa = mna == -INFINITY ? 0.f : mna, sb = mnb == -INFINITY ? 0.f : mnb;
    const float aa = exp2f(ma - sa), ab = exp2f(mb - sb);
    ma = mna;
    mb = mnb;
    la *= aa;
    lb *= ab;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      acc[4 * i] *= aa;
      acc[4 * i + 1] *= aa;
      acc[4 * i + 2] *= ab;
      acc[4 * i + 3] *= ab;
    }
#pragma unroll
    for (int i = 0; i < NS / 4; ++i) {
      sc[4 * i] = exp2f(fmaf(sc[4 * i], sc_mul, -sa));
      sc[4 * i + 1] = exp2f(fmaf(sc[4 * i + 1], sc_mul, -sa));
      sc[4 * i + 2] = exp2f(fmaf(sc[4 * i + 2], sc_mul, -sb));
      sc[4 * i + 3] = exp2f(fmaf(sc[4 * i + 3], sc_mul, -sb));
      la += sc[4 * i] + sc[4 * i + 1];
      lb += sc[4 * i + 2] + sc[4 * i + 3];
    }
#pragma unroll
    for (int u = 0; u < KSP; ++u) {
      const float* p4 = sc + 16 * u;
      pa[u][0] = pack_e4m3x4(p4[0], p4[1], p4[4], p4[5]);
      pa[u][1] = pack_e4m3x4(p4[2], p4[3], p4[6], p4[7]);
      pa[u][2] = pack_e4m3x4(p4[8], p4[9], p4[12], p4[13]);
      pa[u][3] = pack_e4m3x4(p4[10], p4[11], p4[14], p4[15]);
    }
    uint64_t vd[KSP];
#pragma unroll
    for (int u = 0; u < KSP; ++u) vd[u] = make_desc(vt + 32 * u, VLAYOUT, VSBO);
    reg_fence<32>(acc);
    wg_fence();
#pragma unroll
    for (int u = 0; u < KSP; ++u) wgmma_64(acc, pa[u], vd[u], 1);
    wg_commit();
  };
  for (int it = 0; it < nsteps; it += 2) {
    stage(it, sc0, sc1);
    if (it + 1 < nsteps) stage(it + 1, sc1, sc0);
  }
  wg_wait<0>();
  reg_fence<32>(acc);
  reg_fence<4>(qa[0]);
  reg_fence<4>(qa[1]);
#pragma unroll
  for (int u = 0; u < KSP; ++u) reg_fence<4>(pa[u]);

#pragma unroll
  for (int o = 1; o <= 2; o <<= 1) {
    la += __shfl_xor_sync(0xffffffffu, la, o);
    lb += __shfl_xor_sync(0xffffffffu, lb, o);
  }
  const float vs = vscale[0];
  const size_t rowa = row_base + ra, rowb = row_base + rb;
  if (q == 0) {
    ws_m[rowa] = ma;
    ws_l[rowa] = la;
    ws_m[rowb] = mb;
    ws_l[rowb] = lb;
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    *reinterpret_cast<float2*>(ws_o + rowa * D + 8 * i + 2 * q) = make_float2(acc[4 * i] * vs, acc[4 * i + 1] * vs);
    *reinterpret_cast<float2*>(ws_o + rowb * D + 8 * i + 2 * q) = make_float2(acc[4 * i + 2] * vs, acc[4 * i + 3] * vs);
  }
}

void attnpc(torch::Tensor q8, torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor seqlens,
            torch::Tensor kscale, torch::Tensor vscale, torch::Tensor ws_o, torch::Tensor ws_m, torch::Tensor ws_l,
            double qscale, int64_t HKV, int64_t page, int64_t window, int64_t CH, int64_t NCH, int64_t batch,
            int64_t SK) {
  constexpr int STG = 5;
  const size_t smem = STG * 128 * SK + 3 * 64 * SK + 4 * CH + 1024;  // ring, VT x 3, rows, alignment
  TORCH_CHECK(smem <= 200 * 1024, "attnpc: chunk too large");
  dim3 grid(batch * HKV, NCH);
  auto stream = at::cuda::getCurrentCUDAStream();
#define LAUNCHPC(SKK)                                                                                          \
  {                                                                                                           \
    auto k = attnpc_kernel<STG, SKK>;                                                                          \
    static bool set = false;                                                                                   \
    if (!set) {                                                                                                \
      cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024);                        \
      set = true;                                                                                              \
    }                                                                                                         \
    k<<<grid, 256, smem, stream>>>(                                                                            \
        reinterpret_cast<const uint8_t*>(q8.data_ptr()), reinterpret_cast<const uint8_t*>(kc.data_ptr()),      \
        reinterpret_cast<const uint8_t*>(vc.data_ptr()), pt.data_ptr<int>(), seqlens.data_ptr<int>(),           \
        kscale.data_ptr<float>(), vscale.data_ptr<float>(), ws_o.data_ptr<float>(), ws_m.data_ptr<float>(),    \
        ws_l.data_ptr<float>(), (float)qscale, (int)HKV, (int)pt.size(1), (int)page, (int)window, (int)CH,     \
        (int)NCH);                                                                                             \
  }
  TORCH_CHECK(SK == 64, "attnpc: stage of 64 keys");
  LAUNCHPC(64)
}

"""
_attn_CUDA_SRC += r"""
#include <cuda_bf16.h>
// Helpers of the one-launch decode kernel (attnc_kernel below).
__device__ __forceinline__ void bf16x8(const uint4 v, float* x) {
  const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    x[2 * i] = __uint_as_float(w[i] << 16);
    x[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
  }
}
__device__ __forceinline__ uint32_t e4m3x4(float a, float b, float c, float d) {  // bytes a, b, c, d (low first)
  uint16_t lo, hi;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(lo) : "f"(b), "f"(a));
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(hi) : "f"(d), "f"(c));
  return (uint32_t)lo | ((uint32_t)hi << 16);
}
// The RoPE rotations of dims d0 .. d0 + 7 at position pos (pairs (2i, 2i + 1)), and RMSNorm + RoPE with them.
__device__ __forceinline__ void rope_cs(int d0, float pos, float theta_log2, float* sn, float* cn) {
#pragma unroll
  for (int p = 0; p < 4; ++p) sincosf(pos * exp2f(-(2.f * (float)(d0 / 2 + p) / 64.f) * theta_log2), &sn[p], &cn[p]);
}
__device__ __forceinline__ void rms_rope8c(float* x, const float* sn, const float* cn, bool rope) {
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) ss += x[i] * x[i];
  ss += __shfl_xor_sync(0xffffffffu, ss, 1);
  ss += __shfl_xor_sync(0xffffffffu, ss, 2);
  ss += __shfl_xor_sync(0xffffffffu, ss, 4);
  const float r = rsqrtf(ss / 64.f + 1e-5f);
#pragma unroll
  for (int i = 0; i < 8; ++i) x[i] *= r;
  if (rope) {
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      const float xe = x[2 * p], xo = x[2 * p + 1];
      x[2 * p] = xe * cn[p] - xo * sn[p];
      x[2 * p + 1] = xe * sn[p] + xo * cn[p];
    }
  }
}

"""
_attn_CUDA_SRC += r"""
#include <cooperative_groups.h>
// Sliding decode steps in one launch (attn + _attn_prep + _attn_attn_combine), the chunks of each (request, KV-head pair) one
// thread-block cluster (NCH | 16 CTAs): every CTA normalizes and rotates its 16 query heads into shared memory; the
// owner of the new token puts its keys and values straight into its stage tile and stores them to the cache without
// waiting; partials stay in shared memory, and after a cluster barrier CTA c merges rows 16c / NCH .. of all chunks
// over DSMEM, then RMSNorm(o) * sigmoid(gate). A second barrier keeps every CTA's shared memory alive until the
// merges are done.
__device__ __forceinline__ float ex2_approx(float x) {  // 2^x (MUFU), 0 at -inf
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float rcp_approx(float x) {
  float y;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
// dims d0 .. d0 + 7 of one output row, the row's 8 lanes in `mask`: o / l * vscale rounded to BF16 (as SGLang's
// attention output), RMSNorm over the 64 dims, times sigmoid(gate), stored as BF16.
__device__ __forceinline__ void gated_norm_store(float* o, float l, float vsc, const __nv_bfloat16* gate,
                                                 __nv_bfloat16* dst, unsigned mask, bool store = true) {
  const float inv = l > 0.f ? vsc / l : 0.f;
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    o[i] = __bfloat162float(__float2bfloat16(o[i] * inv));
    ss += o[i] * o[i];
  }
  ss += __shfl_xor_sync(mask, ss, 1);
  ss += __shfl_xor_sync(mask, ss, 2);
  ss += __shfl_xor_sync(mask, ss, 4);
  const float rr = rsqrtf(ss / 64.f + 1e-5f);
  float gt[8];
  bf16x8(*reinterpret_cast<const uint4*>(gate), gt);
  uint32_t ow[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const __nv_bfloat162 v =
        __floats2bfloat162_rn(o[2 * i] * rr * rcp_approx(1.f + ex2_approx(-1.4426950408889634f * gt[2 * i])),
                              o[2 * i + 1] * rr * rcp_approx(1.f + ex2_approx(-1.4426950408889634f * gt[2 * i + 1])));
    ow[i] = *reinterpret_cast<const uint32_t*>(&v);
  }
  if (store) *reinterpret_cast<uint4*>(dst) = make_uint4(ow[0], ow[1], ow[2], ow[3]);
}
__device__ __forceinline__ uint32_t dsmem_addr(const void* p, int rank) {  // p in CTA `rank` of the cluster
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"((uint32_t)__cvta_generic_to_shared(p)), "r"(rank));
  return r;
}
__device__ __forceinline__ void st_dsmem2(uint32_t a, float x, float y) {
  asm volatile("st.shared::cluster.v2.f32 [%0], {%1, %2};" ::"r"(a), "f"(x), "f"(y) : "memory");
}
__device__ __forceinline__ void st_dsmem4(uint32_t a, float x, float y, float z, float w) {
  asm volatile("st.shared::cluster.v4.f32 [%0], {%1, %2, %3, %4};" ::"r"(a), "f"(x), "f"(y), "f"(z), "f"(w) : "memory");
}
template <int STG, int SK>
__global__ void __launch_bounds__(128) attnc_kernel(
    const __nv_bfloat16* __restrict__ qkv, const long long* __restrict__ positions,
    const long long* __restrict__ cache_loc, const float* __restrict__ kscale, const float* __restrict__ vscale,
    uint8_t* kc, uint8_t* vc, const int* __restrict__ pt, const int* __restrict__ seqlens,
    __nv_bfloat16* __restrict__ out, float theta_log2, int rope, float q_mult, int HKV, int max_pages, int page,
    int window, int CH, int NCH, int QKV_W) {
  constexpr int G = 8, D = 64, NT = 128, KS = 2, KPW = SK / KS;
  constexpr int TILE = SK * 256, NTS = KPW / 8, KSP = KPW / 16;
  extern __shared__ uint4 smem4[];
  uint8_t* smem = reinterpret_cast<uint8_t*>(smem4);
  uint8_t* base = smem + STG * TILE;
  __half* qs = reinterpret_cast<__half*>(base);                      // 16 query heads x 64, FP16, pre-scaled
  __nv_bfloat16* gs = reinterpret_cast<__nv_bfloat16*>(base + 2048);  // gate logits of the rows this CTA merges
  // Partials pushed here by every chunk's CTA for the rows this CTA merges: slot 2 (jr NCH + chunk) + key half,
  // 68 floats each (o[64], m, l).
  float* rcv = reinterpret_cast<float*>(base + 4096);
  uint8_t* nkv = base + 13056;                                         // the new key row, then value row (128 B)
  int* rows = reinterpret_cast<int*>(base + 13312);                    // the chunk's cache rows
  cooperative_groups::cluster_group cl = cooperative_groups::this_cluster();

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, q = lane & 3;
  const int ngroups = HKV / 2, HQ = HKV * G;
  const int bg = blockIdx.x, c = blockIdx.y;
  const int b = bg / ngroups, h0 = (bg % ngroups) * 2;
  const int RPC = 16 / NCH;  // rows merged per CTA
  const __nv_bfloat16* tok = qkv + (size_t)b * QKV_W;
  const int jh = tid >> 3, d0 = (tid & 7) * 8;  // thread: dims d0 .. d0 + 7 of query head h0 G + jh
  if (tid < RPC * 8) cp16(gs + tid * 8, tok + (HQ + 2 * HKV + h0 * G + c * RPC + jh) * D + d0, true);
  cp_commit();
  // Everything that does not depend on the request's length is read at once, up front: the queries, the new keys
  // and values (warp 0; only the owner uses them), the scalars (pinned in registers: left to the compiler, the
  // scale loads sink to their uses, a cold DRAM read each at the end of the kernel).
  const uint4 qraw = *reinterpret_cast<const uint4*>(tok + (h0 * G + jh) * D + d0);
  uint4 kvraw = make_uint4(0, 0, 0, 0);
  if (warp == 0)  // lanes 0-15 the new keys, 16-31 the new values (head h0 + bit 3 of the lane)
    kvraw = *reinterpret_cast<const uint4*>(tok + (HQ + (lane >> 4) * HKV + h0 + ((lane >> 3) & 1)) * D + d0);
  float pos = (float)positions[b], ksc = kscale[0], vsc = vscale[0];
  long long loc = cache_loc[b];
  int seq = seqlens[b];
  asm volatile("" : "+f"(pos), "+f"(ksc), "+f"(vsc), "+l"(loc), "+r"(seq));
  const int hi = seq + 1;
  const int lo = window > 0 ? max(seq - (window - 1), 0) : 0;
  const int cs = lo + c * CH, ce = min(cs + CH, hi);
  const bool empty = cs >= hi, owner = !empty && seq < ce;
  const int hh = warp & 1, ks = warp >> 1;

  if (!empty) {
    for (int i = tid; i < ce - cs; i += NT) {
      const int pos = cs + i;
      rows[i] = pt[(size_t)b * max_pages + pos / page] * page + pos % page;
    }
    float x[8], sn[4], cn[4];
    if (rope) rope_cs(d0, pos, theta_log2, sn, cn);
    bf16x8(qraw, x);
    rms_rope8c(x, sn, cn, rope);
    const float qm = q_mult * ksc;
    uint4 qh;
    qh.x = pack_f16(x[0] * qm, x[1] * qm);
    qh.y = pack_f16(x[2] * qm, x[3] * qm);
    qh.z = pack_f16(x[4] * qm, x[5] * qm);
    qh.w = pack_f16(x[6] * qm, x[7] * qm);
    *reinterpret_cast<uint4*>(qs + jh * D + d0) = qh;
    if (owner && warp == 0) {
      const bool isk = lane < 16;
      bf16x8(kvraw, x);
      float y[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) y[i] = x[i];
      rms_rope8c(y, sn, cn, rope);  // all lanes shuffle; the value lanes drop theirs (same dims: same angles)
      const float s = isk ? ksc : vsc;
#pragma unroll
      for (int i = 0; i < 8; ++i) x[i] = (isk ? y[i] : x[i]) / s;
      const uint2 pk = make_uint2(e4m3x4(x[0], x[1], x[2], x[3]), e4m3x4(x[4], x[5], x[6], x[7]));
      const int hb = ((lane >> 3) & 1) * 64 + d0;
      *reinterpret_cast<uint2*>(nkv + (lane >> 4) * 128 + hb) = pk;
      *reinterpret_cast<uint2*>((isk ? kc : vc) + ((size_t)loc * HKV + h0) * D + hb) = pk;
    }
  }
  __syncthreads();

  if (!empty) {
    uint32_t qa[4][2];  // row g of head h0 + hh (rows 8 .. 15 are padding: zero); lane q, k-step s: dims 16q + 4s
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const uint2 v0 = *reinterpret_cast<const uint2*>(qs + (hh * G + g) * D + 16 * q + 4 * s);
      qa[s][0] = v0.x;
      qa[s][1] = v0.y;
    }
    const int nsteps = (ce - cs + SK - 1) / SK;
    constexpr int PER = 16 * SK / NT;
    auto issue = [&](int stage, int kb) {
      uint8_t* tile = smem + stage * TILE;
#pragma unroll
      for (int i = 0; i < PER; ++i) {
        const int idx = tid + i * NT;
        const int key = idx >> 4, part = (idx >> 3) & 1, ch = idx & 7;
        const bool ok = kb + key < ce;
        const int row = ok ? rows[kb + key - cs] : 0;
        const uint8_t* src = (part ? vc : kc) + ((long long)row * HKV + h0) * D + ch * 16;
        const int sw = part ? (ch ^ (((key >> 1) & 3) << 1)) : (ch ^ ((key & 1) << 2));
        uint8_t* dst = tile + part * (SK * 128) + key * 128 + sw * 16;
        if (ok && kb + key == seq)
          *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(nkv + part * 128 + ch * 16);
        else
          cp16(dst, src, ok);
      }
    };
#pragma unroll
    for (int s = 0; s < STG - 1; ++s) {
      if (s < nsteps) issue(s, cs + SK * s);
      cp_commit();
    }
    float ma = -INFINITY, la = 0.f;
    float acc[8][4];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
      for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;

    for (int it = 0; it < nsteps; ++it) {
      cp_wait<STG - 2>();
      __syncthreads();
      {
        const int nx = it + STG - 1;
        if (nx < nsteps) issue(nx % STG, cs + SK * nx);
        cp_commit();
      }
      const uint8_t* Kt = smem + (it % STG) * TILE;
      const uint8_t* Vt = Kt + SK * 128;
      const int kbs = cs + SK * it, kb = kbs + ks * KPW;
      float sc[NTS][4];
#pragma unroll
      for (int t = 0; t < NTS; ++t) {
#pragma unroll
        for (int j = 0; j < 4; ++j) sc[t][j] = 0.f;
        const int key = ks * KPW + 8 * t + g;
        const uint4 kv = *reinterpret_cast<const uint4*>(Kt + key * 128 + (((4 * hh + q) ^ ((key & 1) << 2)) * 16));
        const uint32_t w[4] = {kv.x, kv.y, kv.z, kv.w};
#pragma unroll
        for (int s = 0; s < 4; ++s)
          mma_f16(sc[t], qa[s][0], 0u, qa[s][1], 0u, e4m3x2_f16x2(w[s]), e4m3x2_f16x2(w[s] >> 16));
      }
      if (kbs + SK > ce) {  // every key in [lo, hi) is visible to the query: only keys past the chunk are masked
#pragma unroll
        for (int t = 0; t < NTS; ++t)
#pragma unroll
          for (int e = 0; e < 2; ++e)
            if (kb + 8 * t + 2 * q + e >= ce) sc[t][e] = -INFINITY;
      }
      float mxa = -INFINITY;
#pragma unroll
      for (int t = 0; t < NTS; ++t) mxa = fmaxf(mxa, fmaxf(sc[t][0], sc[t][1]));
      mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 1));
      mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 2));
      const float mna = fmaxf(ma, mxa);
      const float sa = mna == -INFINITY ? 0.f : mna;
      const float aa = exp2f(ma - sa);
      ma = mna;
      la *= aa;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        acc[i][0] *= aa;
        acc[i][1] *= aa;
      }
      uint32_t pa[NTS];
#pragma unroll
      for (int t = 0; t < NTS; ++t) {
        const float p0 = exp2f(sc[t][0] - sa), p1 = exp2f(sc[t][1] - sa);
        la += p0 + p1;
        pa[t] = pack_f16(p0, p1);
      }
#pragma unroll
      for (int u = 0; u < KSP; ++u) {
        uint2 vr[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int key = ks * KPW + 16 * u + 2 * q + (e & 1) + (e >> 1) * 8;
          const int unit8 = (8 * hh + g) ^ (((key >> 1) & 3) << 2);
          vr[e] = *reinterpret_cast<const uint2*>(Vt + key * 128 + unit8 * 8);
        }
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
          const uint32_t x0 = nt < 4 ? vr[0].x : vr[0].y, x1 = nt < 4 ? vr[1].x : vr[1].y;
          const uint32_t x2 = nt < 4 ? vr[2].x : vr[2].y, x3 = nt < 4 ? vr[3].x : vr[3].y;
          const uint32_t sel = (nt & 3) | (((nt & 3) + 4) << 4);
          const uint32_t b0 = e4m3x2_f16x2(__byte_perm(x0, x1, sel));
          const uint32_t b1 = e4m3x2_f16x2(__byte_perm(x2, x3, sel));
          mma_f16(acc[nt], pa[2 * u], 0u, pa[2 * u + 1], 0u, b0, b1);
        }
      }
    }
    cp_wait<0>();
    la += __shfl_xor_sync(0xffffffffu, la, 1);
    la += __shfl_xor_sync(0xffffffffu, la, 2);
    // Row hh * 8 + g (query head h0 G + 8 hh + g) goes to the CTA merging it (DSMEM stores: no wait). Lane q holds
    // dims 16q + nt (acc[nt][0]) and 16q + 8 + nt (acc[nt][1]).
    const int j = hh * G + g;
    const uint32_t d = dsmem_addr(rcv + (2 * ((j % RPC) * NCH + c) + ks) * 68, j / RPC);
    st_dsmem4(d + 4 * 16 * q, acc[0][0], acc[1][0], acc[2][0], acc[3][0]);
    st_dsmem4(d + 4 * (16 * q + 4), acc[4][0], acc[5][0], acc[6][0], acc[7][0]);
    st_dsmem4(d + 4 * (16 * q + 8), acc[0][1], acc[1][1], acc[2][1], acc[3][1]);
    st_dsmem4(d + 4 * (16 * q + 12), acc[4][1], acc[5][1], acc[6][1], acc[7][1]);
    if (q == 0) st_dsmem2(d + 4 * 64, ma, la);
  } else {  // an empty chunk: zero partials (m = -inf) for both key halves of every row
    const int j = tid >> 3, part = tid & 7;
    const uint32_t d = dsmem_addr(rcv + 2 * ((j % RPC) * NCH + c) * 68, j / RPC);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      st_dsmem2(d + 4 * (8 * part + 2 * i), 0.f, 0.f);
      st_dsmem2(d + 4 * (68 + 8 * part + 2 * i), 0.f, 0.f);
    }
    if (part == 0) {
      st_dsmem2(d + 4 * 64, -INFINITY, 0.f);
      st_dsmem2(d + 4 * (68 + 64), -INFINITY, 0.f);
    }
  }
  cp_wait<0>();  // the gate logits
  cl.sync();  // every chunk's partials have landed

  // Thread tid < 8 RPC: dims d0 .. d0 + 7 of row jh (query head h0 G + c RPC + jh) over every chunk and key half.
  if (tid < RPC * 8) {
    const float* rr0 = rcv + jh * 2 * NCH * 68;
    float m = -INFINITY;
    for (int k = 0; k < 2 * NCH; ++k) m = fmaxf(m, rr0[k * 68 + 64]);
    const float ms = m == -INFINITY ? 0.f : m;
    float l = 0.f, o[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) o[i] = 0.f;
#pragma unroll 2
    for (int k = 0; k < 2 * NCH; ++k) {
      const float* rs = rr0 + k * 68;
      const float w = ex2_approx(rs[64] - ms);  // 0 for empty chunks (their outputs are zero)
      const float4 a = *reinterpret_cast<const float4*>(rs + d0);
      const float4 e = *reinterpret_cast<const float4*>(rs + d0 + 4);
      l = fmaf(w, rs[65], l);
      o[0] = fmaf(w, a.x, o[0]);
      o[1] = fmaf(w, a.y, o[1]);
      o[2] = fmaf(w, a.z, o[2]);
      o[3] = fmaf(w, a.w, o[3]);
      o[4] = fmaf(w, e.x, o[4]);
      o[5] = fmaf(w, e.y, o[5]);
      o[6] = fmaf(w, e.z, o[6]);
      o[7] = fmaf(w, e.w, o[7]);
    }
    gated_norm_store(o, l, vsc, gs + jh * D + d0, out + (size_t)b * HQ * D + (h0 * G + c * RPC + jh) * D + d0,
                     0xffu << (lane & 24));
  }
}

void attnc(torch::Tensor qkv, torch::Tensor positions, torch::Tensor cache_loc, torch::Tensor kscale,
           torch::Tensor vscale, torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor seqlens,
           torch::Tensor out, double theta_log2, int64_t rope, double q_mult, int64_t HKV, int64_t page,
           int64_t window, int64_t CH, int64_t NCH, int64_t batch, int64_t SK) {
  constexpr int STG = 4;
  const size_t smem = STG * 256 * SK + 13312 + 4 * CH;  // ring, queries, gates, partials, new row, cache rows
  TORCH_CHECK(smem <= 200 * 1024, "attnc: chunk too large");
  TORCH_CHECK(NCH >= 1 && NCH <= 16 && 16 % NCH == 0, "attnc: the chunks must divide 16");
  TORCH_CHECK(positions.scalar_type() == torch::kInt64 && cache_loc.scalar_type() == torch::kInt64,
              "attnc: int64 positions and cache rows");
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(batch * (HKV / 2), NCH);
  cfg.blockDim = dim3(128);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = 1;
  attr[0].val.clusterDim.y = NCH;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
#define LAUNCHC(SKK)                                                                                           \
  {                                                                                                           \
    auto k = attnc_kernel<STG, SKK>;                                                                           \
    static bool set = false;                                                                                   \
    if (!set) {                                                                                                \
      cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024);                        \
      cudaFuncSetAttribute(k, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);                              \
      set = true;                                                                                              \
    }                                                                                                         \
    const cudaError_t err = cudaLaunchKernelEx(                                                                         \
        &cfg, k, reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr()),                                       \
        reinterpret_cast<const long long*>(positions.data_ptr()),                                              \
        reinterpret_cast<const long long*>(cache_loc.data_ptr()), kscale.data_ptr<float>(),                    \
        vscale.data_ptr<float>(), reinterpret_cast<uint8_t*>(kc.data_ptr()),                                   \
        reinterpret_cast<uint8_t*>(vc.data_ptr()), pt.data_ptr<int>(), seqlens.data_ptr<int>(),                \
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), (float)theta_log2, (int)rope, (float)q_mult,         \
        (int)HKV, (int)pt.size(1), (int)page, (int)window, (int)CH, (int)NCH, (int)qkv.size(1));               \
    TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));                                               \
  }
  TORCH_CHECK(SK == 32 || SK == 64, "attnc: stage of 32 or 64 keys");
  if (SK == 32) LAUNCHC(32) else LAUNCHC(64)
}
"""
_attn_CUDA_SRC += r"""
// Sliding verify blocks in one launch (_attn_qkv_prep + attnpc + _attn_attn_combine): attnpc's CTA (one KV head; a producer
// and a consumer warpgroup) with the chunks of each (request, KV head) one cluster (NCH <= 16 CTAs). Every CTA puts
// its 64 query rows (RMSNorm, RoPE, E4M3; row r = 8 j + g) in shared memory, which the consumer reads every stage;
// the CTAs whose chunk holds new tokens compute their keys and values, which the producer copies into the stage
// tiles from shared memory (their cache stores go out without waiting). The consumer's partials go to the idle ring
// and after a cluster barrier each warp merges rows of all chunks over DSMEM, then RMSNorm(o) * sigmoid(gate).
// The merge of attnpcc_kernel: lane (dims 8 (t & 7) .. + 7) of thread t's warp merges row jr = 4 (t >> 5) + bits 3-4
// of t (query row c RPC + jr) over every chunk, from rcv (slot jr NCH + chunk: o[64], m, l). Not inlined: the
// producer runs it early with store = false so its code is in the instruction cache when the real merge runs.
__device__ __noinline__ void verify_merge(const float* rcv, const __nv_bfloat16* gs, __nv_bfloat16* out, float vsc,
                                          int t, int b, int h, int c, int RPC, int NCH, int HQ, bool store) {
  constexpr int L = 8, G = 8, D = 64;
  const int jr = 4 * (t >> 5) + ((t >> 3) & 3), j = c * RPC + jr, s8 = (t & 7) * 8;
  if (jr >= RPC || j >= 64) return;
  const float* rr0 = rcv + jr * NCH * 68;
  float m = -INFINITY;
  for (int k = 0; k < NCH; ++k) m = fmaxf(m, rr0[k * 68 + 64]);
  const float ms = m == -INFINITY ? 0.f : m;
  float l = 0.f, o[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) o[i] = 0.f;
#pragma unroll 2
  for (int k = 0; k < NCH; ++k) {
    const float* rs = rr0 + k * 68;
    const float w = ex2_approx(rs[64] - ms);  // 0 for empty chunks (their outputs are zero)
    const float4 a = *reinterpret_cast<const float4*>(rs + s8);
    const float4 e = *reinterpret_cast<const float4*>(rs + s8 + 4);
    l = fmaf(w, rs[65], l);
    o[0] = fmaf(w, a.x, o[0]);
    o[1] = fmaf(w, a.y, o[1]);
    o[2] = fmaf(w, a.z, o[2]);
    o[3] = fmaf(w, a.w, o[3]);
    o[4] = fmaf(w, e.x, o[4]);
    o[5] = fmaf(w, e.y, o[5]);
    o[6] = fmaf(w, e.z, o[6]);
    o[7] = fmaf(w, e.w, o[7]);
  }
  gated_norm_store(o, l, vsc, gs + jr * D + s8, out + ((size_t)b * L + (j >> 3)) * HQ * D + (h * G + (j & 7)) * D + s8,
                   0xffu << (t & 24), store);
}

template <int STG, int SK>
__global__ void __launch_bounds__(256, 2) attnpcc_kernel(
    const __nv_bfloat16* __restrict__ qkv, const long long* __restrict__ positions,
    const long long* __restrict__ cache_loc, const float* __restrict__ kscale, const float* __restrict__ vscale,
    uint8_t* kc, uint8_t* vc, const int* __restrict__ pt, const int* __restrict__ seqlens,
    __nv_bfloat16* __restrict__ out, float theta_log2, int rope, float qscale, int HKV, int max_pages, int page,
    int window, int CH, int NCH, int QKV_W) {
  constexpr int L = 8, G = 8, D = 64, NT = 256, RB = 64, CPR = RB / 16;
  constexpr int TILE = SK * 2 * RB, KSP = SK / 32, NS = SK / 2;
  static_assert(SK == 64, "stage of 64 keys");
  constexpr int KLAYOUT = 2, KSBO = 512, VLAYOUT = 2, VSBO = 512;
  extern __shared__ uint4 smem4[];
  uint8_t* smem = reinterpret_cast<uint8_t*>(smem4);
  smem += (1024 - ((uint32_t)__cvta_generic_to_shared(smem) & 1023)) & 1023;
  uint8_t* VT = smem + STG * TILE;                                      // three transposed value buffers
  uint8_t* qs = VT + 3 * 64 * SK;                                       // 64 query rows x 64 B, E4M3
  uint8_t* nkv = qs + 64 * 64;                                          // new tokens: [8][keys 64 B | values 64 B]
  __nv_bfloat16* gs = reinterpret_cast<__nv_bfloat16*>(nkv + 8 * 128);  // gate logits of the rows merged here
  float2* rt = reinterpret_cast<float2*>(gs + 16 * D);                  // RoPE (sin, cos): [8 tokens][32 pairs]
  // Partials pushed here by every chunk's CTA for the rows this CTA merges: slot jr NCH + chunk, 68 floats each
  // (o[64], m, l).
  float* rcv = reinterpret_cast<float*>(rt + 256);
  int* rows = reinterpret_cast<int*>(rcv + 80 * 68);                    // the chunk's cache rows
  cooperative_groups::cluster_group cl = cooperative_groups::this_cluster();

  const bool producer = threadIdx.x >= 128;
  const int tid = threadIdx.x & 127, wq = tid >> 5, lane = tid & 31, g = lane >> 2, q = lane & 3;
  const int bh = blockIdx.x, c = blockIdx.y;
  const int b = bh / HKV, h = bh % HKV, HQ = HKV * G;
  const int RPC = (64 + NCH - 1) / NCH;  // rows merged per CTA: c RPC ..
  const __nv_bfloat16* tok = qkv + (size_t)b * L * QKV_W;
  // Up front, independent of the request's length: the queries (two (row, 8-dim) slices per thread), the new keys and
  // values (threads 0-127: k for 0-63, v for 64-127; token (t >> 3) & 7), the scalars (pinned in registers).
  const int s8 = (threadIdx.x & 7) * 8;
  uint4 qraw[2];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int r = (threadIdx.x >> 3) + 32 * i, j = r >> 3;
    qraw[i] = *reinterpret_cast<const uint4*>(tok + (size_t)j * QKV_W + (h * G + (r & 7)) * D + s8);
  }
  const int kj = (threadIdx.x >> 3) & 7, kpart = (threadIdx.x >> 6) & 1;
  uint4 kvraw = make_uint4(0, 0, 0, 0);
  if (threadIdx.x < 128)
    kvraw = *reinterpret_cast<const uint4*>(tok + (size_t)kj * QKV_W + (HQ + kpart * HKV + h) * D + s8);
  float ksc = kscale[0], vsc = vscale[0], tpos = (float)positions[b * L + (threadIdx.x >> 5)];
  long long kloc = cache_loc[b * L + kj];
  int seq = seqlens[b];
  asm volatile("" : "+f"(ksc), "+f"(vsc), "+f"(tpos), "+l"(kloc), "+r"(seq));
  if (threadIdx.x < RPC * 8 && c * RPC + (threadIdx.x >> 3) < 64) {
    const int j = c * RPC + (threadIdx.x >> 3);
    cp16(gs + threadIdx.x * 8, tok + (size_t)(j >> 3) * QKV_W + (HQ + 2 * HKV + h * G + (j & 7)) * D + s8, true);
  }
  cp_commit();
  const int hi = seq + L;
  const int lo = window > 0 ? max(seq - (window - 1), 0) : 0;
  const int cs = lo + c * CH, ce = min(cs + CH, hi);
  const bool empty = cs >= hi, owner = !empty && ce > seq;

  if (!empty) {
    const int nsteps = (ce - cs + SK - 1) / SK;
    auto load_rows = [&](int from, int to) {
      for (int i0 = from + threadIdx.x; i0 < to; i0 += 8 * NT) {
        int v[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int pos = cs + i0 + j * NT;
          v[j] = pos < ce ? pt[(size_t)b * max_pages + pos / page] * page + pos % page : 0;
        }
#pragma unroll
        for (int j = 0; j < 8; ++j)
          if (i0 + j * NT < to) rows[i0 + j * NT] = v[j];
      }
    };
    const int first = min(ce - cs, SK * (STG - 1));  // <= NT: one page-table read per thread, stored below
    const int prow = threadIdx.x < first ? pt[(size_t)b * max_pages + (cs + threadIdx.x) / page] * page +
                                               (cs + threadIdx.x) % page : 0;
    if (rope) {  // thread: token threadIdx.x / 32, pair threadIdx.x % 32
      float sn, cn;
      sincosf(tpos * exp2f(-(2.f * (float)(threadIdx.x & 31) / 64.f) * theta_log2), &sn, &cn);
      rt[threadIdx.x] = make_float2(sn, cn);
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int r = (threadIdx.x >> 3) + 32 * i;
      float x[8], sn[4], cn[4];
#pragma unroll
      for (int p = 0; p < 4; ++p) {
        const float2 v = rt[(r >> 3) * 32 + s8 / 2 + p];
        sn[p] = v.x;
        cn[p] = v.y;
      }
      bf16x8(qraw[i], x);
      rms_rope8c(x, sn, cn, rope);
      *reinterpret_cast<uint2*>(qs + r * 64 + s8) = make_uint2(e4m3x4(x[0], x[1], x[2], x[3]), e4m3x4(x[4], x[5], x[6], x[7]));
    }
    if (owner && threadIdx.x < 128) {
      const bool isk = kpart == 0;
      float x[8], y[8], sn[4], cn[4];
#pragma unroll
      for (int p = 0; p < 4; ++p) {
        const float2 v = rt[kj * 32 + s8 / 2 + p];
        sn[p] = v.x;
        cn[p] = v.y;
      }
      bf16x8(kvraw, x);
#pragma unroll
      for (int i = 0; i < 8; ++i) y[i] = x[i];
      rms_rope8c(y, sn, cn, rope);  // all lanes shuffle; the value lanes drop theirs
      const float s = isk ? ksc : vsc;
#pragma unroll
      for (int i = 0; i < 8; ++i) x[i] = (isk ? y[i] : x[i]) / s;
      const uint2 pk = make_uint2(e4m3x4(x[0], x[1], x[2], x[3]), e4m3x4(x[4], x[5], x[6], x[7]));
      *reinterpret_cast<uint2*>(nkv + kj * 128 + kpart * 64 + s8) = pk;
      if (seq + kj >= cs && seq + kj < ce)
        *reinterpret_cast<uint2*>((isk ? kc : vc) + ((size_t)kloc * HKV + h) * D + s8) = pk;
    }
    if (threadIdx.x < first) rows[threadIdx.x] = prow;
    __syncthreads();  // P1

    if (producer) {
      constexpr int PER = SK / 16;
      auto issue = [&](int stage, int kb) {
        uint8_t* tile = smem + stage * TILE;
#pragma unroll
        for (int i = 0; i < PER; ++i) {
          const int idx = tid + i * 128;
          const int key = idx / (2 * CPR), part = (idx / CPR) & 1, ch = idx % CPR;
          const bool ok = kb + key < ce;
          const int row = ok ? rows[kb + key - cs] : 0;
          const uint8_t* src = (part ? vc : kc) + ((long long)row * HKV + h) * D + ch * 16;
          uint8_t* dst = tile + part * (SK * RB) + key * RB + ((ch ^ ((key >> 1) & 3)) << 4);
          if (ok && kb + key >= seq)
            *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(nkv + (kb + key - seq) * 128 + part * 64 + ch * 16);
          else
            cp16(dst, src, ok);
        }
      };
      auto transpose = [&](int st) {
        const uint8_t* Vt = smem + (st % STG) * TILE + SK * RB;
        uint8_t* vt = VT + (st % 3) * (64 * SK);
#pragma unroll
        for (int blk = tid; blk < SK * 4; blk += 128) {
          const int qq = blk & 3, hf = (blk >> 2) & 1, u = (blk >> 3) % KSP, dg = ((blk >> 3) / KSP) & 15;
          const int byte = 4 * dg;
          uint32_t x[4], o[4];
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int key = 32 * u + 16 * hf + 2 * qq + (e & 1) + 8 * (e >> 1);
            x[e] = *reinterpret_cast<const uint32_t*>(Vt + key * RB + (((byte >> 4) ^ ((key >> 1) & 3)) << 4) + (byte & 15));
          }
          transpose4(x[0], x[1], x[2], x[3], o);
          const int slot = 32 * u + 16 * hf + 4 * qq, chunk = slot >> 4;
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const int dim = 4 * dg + j;
            *reinterpret_cast<uint32_t*>(vt + dim * SK + ((chunk ^ ((dim >> 1) & 3)) << 4) + (slot & 15)) = o[j];
          }
        }
      };
#pragma unroll
      for (int s = 0; s < STG - 1; ++s) {
        if (s < nsteps) issue(s, cs + SK * s);
        cp_commit();
      }
      load_rows(first, ce - cs);
      cp_wait<STG - 3>();  // stages 0 and 1
      fence_async();
      __syncthreads();  // P2
      transpose(0);
      fence_async();
      for (int it = 0; it < nsteps; ++it) {
        __syncthreads();  // B(it)
        {
          const int nx = it + STG - 1;
          if (nx < nsteps) issue(nx % STG, cs + SK * nx);
          cp_commit();
        }
        cp_wait<STG - 3>();
        if (it + 1 < nsteps) transpose(it + 1);
        fence_async();
      }
      cp_wait<0>();
      // Idle until the merge: run its code once (nothing stored) so the real merge finds it in the instruction cache
      // (every replay starts with a cold L2: the first fetch of that code is a DRAM read per line).
      verify_merge(rcv, gs, out, vsc, tid, b, h, c, RPC, NCH, HQ, false);
    } else {
      const int ra = 16 * wq + g, rb = ra + 8;
      const uint8_t* qa_p = qs + ra * 64 + 4 * q;
      const uint8_t* qb_p = qs + rb * 64 + 4 * q;
      const int qpa = seq + ra / G, qpb = seq + rb / G;
      const float sc_mul = qscale * ksc;
      float ma = -INFINITY, mb = -INFINITY, la = 0.f, lb = 0.f;
      float acc[32];
#pragma unroll
      for (int i = 0; i < 32; ++i) acc[i] = 0.f;
      float sc0[NS], sc1[NS];
      uint32_t pa[KSP][4];
      uint32_t qa[2][4];
      auto issue_s = [&](int st, float* sc) {
#pragma unroll
        for (int s = 0; s < 2; ++s) {
          qa[s][0] = *reinterpret_cast<const uint32_t*>(qa_p + 32 * s);
          qa[s][1] = *reinterpret_cast<const uint32_t*>(qb_p + 32 * s);
          qa[s][2] = *reinterpret_cast<const uint32_t*>(qa_p + 32 * s + 16);
          qa[s][3] = *reinterpret_cast<const uint32_t*>(qb_p + 32 * s + 16);
        }
        const uint8_t* Kt = smem + (st % STG) * TILE;
        const uint64_t kd0 = make_desc(Kt, KLAYOUT, KSBO), kd1 = make_desc(Kt + 32, KLAYOUT, KSBO);
        reg_fence<NS>(sc);
        wg_fence();
        wgmma_64(sc, qa[0], kd0, 0);
        wgmma_64(sc, qa[1], kd1, 1);
        wg_commit();
      };
      load_rows(first, ce - cs);
      __syncthreads();  // P2: stages 0 and 1 landed
      issue_s(0, sc0);
      auto stage = [&](int it, float* sc, float* nxt) {
        __syncthreads();  // B(it)
        wg_wait<0>();
        reg_fence<NS>(sc);
        reg_fence<32>(acc);
        reg_fence<4>(qa[0]);
        reg_fence<4>(qa[1]);
#pragma unroll
        for (int u = 0; u < KSP; ++u) reg_fence<4>(pa[u]);
        if (it + 1 < nsteps) issue_s(it + 1, nxt);
        const uint8_t* vt = VT + (it % 3) * (64 * SK);
        const int kbs = cs + SK * it;
        const bool edge = kbs + SK > ce || kbs + SK > hi - L + 1 || (window > 0 && kbs < lo + L - 1);
        if (edge) {
#pragma unroll
          for (int i = 0; i < NS / 4; ++i)
#pragma unroll
            for (int e = 0; e < 2; ++e) {
              const int pos = kbs + 8 * i + 2 * q + e;
              const bool okk = pos < ce;
              const bool oka = okk && pos <= qpa && (window == 0 || pos > qpa - window);
              const bool okb = okk && pos <= qpb && (window == 0 || pos > qpb - window);
              sc[4 * i + e] = oka ? sc[4 * i + e] : -INFINITY;
              sc[4 * i + 2 + e] = okb ? sc[4 * i + 2 + e] : -INFINITY;
            }
        }
        float mxa = -INFINITY, mxb = -INFINITY;
#pragma unroll
        for (int i = 0; i < NS / 4; ++i) {
          mxa = fmaxf(mxa, fmaxf(sc[4 * i], sc[4 * i + 1]));
          mxb = fmaxf(mxb, fmaxf(sc[4 * i + 2], sc[4 * i + 3]));
        }
#pragma unroll
        for (int o = 1; o <= 2; o <<= 1) {
          mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, o));
          mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, o));
        }
        const float mna = fmaxf(ma, mxa * sc_mul), mnb = fmaxf(mb, mxb * sc_mul);
        const float sa = mna == -INFINITY ? 0.f : mna, sb = mnb == -INFINITY ? 0.f : mnb;
        const float aa = exp2f(ma - sa), ab = exp2f(mb - sb);
        ma = mna;
        mb = mnb;
        la *= aa;
        lb *= ab;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          acc[4 * i] *= aa;
          acc[4 * i + 1] *= aa;
          acc[4 * i + 2] *= ab;
          acc[4 * i + 3] *= ab;
        }
#pragma unroll
        for (int i = 0; i < NS / 4; ++i) {
          sc[4 * i] = exp2f(fmaf(sc[4 * i], sc_mul, -sa));
          sc[4 * i + 1] = exp2f(fmaf(sc[4 * i + 1], sc_mul, -sa));
          sc[4 * i + 2] = exp2f(fmaf(sc[4 * i + 2], sc_mul, -sb));
          sc[4 * i + 3] = exp2f(fmaf(sc[4 * i + 3], sc_mul, -sb));
          la += sc[4 * i] + sc[4 * i + 1];
          lb += sc[4 * i + 2] + sc[4 * i + 3];
        }
#pragma unroll
        for (int u = 0; u < KSP; ++u) {
          const float* p4 = sc + 16 * u;
          pa[u][0] = pack_e4m3x4(p4[0], p4[1], p4[4], p4[5]);
          pa[u][1] = pack_e4m3x4(p4[2], p4[3], p4[6], p4[7]);
          pa[u][2] = pack_e4m3x4(p4[8], p4[9], p4[12], p4[13]);
          pa[u][3] = pack_e4m3x4(p4[10], p4[11], p4[14], p4[15]);
        }
        uint64_t vd[KSP];
#pragma unroll
        for (int u = 0; u < KSP; ++u) vd[u] = make_desc(vt + 32 * u, VLAYOUT, VSBO);
        reg_fence<32>(acc);
        wg_fence();
#pragma unroll
        for (int u = 0; u < KSP; ++u) wgmma_64(acc, pa[u], vd[u], 1);
        wg_commit();
      };
      for (int it = 0; it < nsteps; it += 2) {
        stage(it, sc0, sc1);
        if (it + 1 < nsteps) stage(it + 1, sc1, sc0);
      }
      wg_wait<0>();
      reg_fence<32>(acc);
      reg_fence<4>(qa[0]);
      reg_fence<4>(qa[1]);
#pragma unroll
      for (int u = 0; u < KSP; ++u) reg_fence<4>(pa[u]);
#pragma unroll
      for (int o = 1; o <= 2; o <<= 1) {
        la += __shfl_xor_sync(0xffffffffu, la, o);
        lb += __shfl_xor_sync(0xffffffffu, lb, o);
      }
      // Rows ra and rb go to the CTAs merging them (DSMEM stores: no wait).
      const uint32_t da = dsmem_addr(rcv + ((ra % RPC) * NCH + c) * 68, ra / RPC);
      const uint32_t db = dsmem_addr(rcv + ((rb % RPC) * NCH + c) * 68, rb / RPC);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        st_dsmem2(da + 4 * (8 * i + 2 * q), acc[4 * i], acc[4 * i + 1]);
        st_dsmem2(db + 4 * (8 * i + 2 * q), acc[4 * i + 2], acc[4 * i + 3]);
      }
      if (q == 0) {
        st_dsmem2(da + 4 * 64, ma, la);
        st_dsmem2(db + 4 * 64, mb, lb);
      }
    }
  } else {  // an empty chunk: zero partials (m = -inf) for every row
    const int r = threadIdx.x >> 2, part = threadIdx.x & 3;
    const uint32_t d = dsmem_addr(rcv + ((r % RPC) * NCH + c) * 68, r / RPC);
#pragma unroll
    for (int i = 0; i < 8; ++i) st_dsmem2(d + 4 * (16 * part + 2 * i), 0.f, 0.f);
    if (part == 0) st_dsmem2(d + 4 * 64, -INFINITY, 0.f);
  }
  cp_wait<0>();  // the gate logits
  cl.sync();      // every chunk's partials have landed

  verify_merge(rcv, gs, out, vsc, threadIdx.x, b, h, c, RPC, NCH, HQ, true);
}

void attnpcc(torch::Tensor qkv, torch::Tensor positions, torch::Tensor cache_loc, torch::Tensor kscale,
             torch::Tensor vscale, torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor seqlens,
             torch::Tensor out, double theta_log2, int64_t rope, double qscale, int64_t HKV, int64_t page,
             int64_t window, int64_t CH, int64_t NCH, int64_t batch) {
  constexpr int STG = 5, SK = 64;
  // ring, VT x 3, queries, new tokens, gates, RoPE table, received partials, the chunk's cache rows, alignment
  const size_t smem = STG * 128 * SK + 3 * 64 * SK + 64 * 64 + 8 * 128 + 16 * 64 * 2 + 2048 + 80 * 68 * 4 + 4 * CH + 1024;
  TORCH_CHECK(smem <= 200 * 1024, "attnpcc: chunk too large");
  TORCH_CHECK(NCH >= 1 && NCH <= 16, "attnpcc: at most 16 chunks (one cluster)");
  TORCH_CHECK((64 + NCH - 1) / NCH <= 16, "attnpcc: at most 16 merged rows per CTA");
  TORCH_CHECK(positions.scalar_type() == torch::kInt64 && cache_loc.scalar_type() == torch::kInt64,
              "attnpcc: int64 positions and cache rows");
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(batch * HKV, NCH);
  cfg.blockDim = dim3(256);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = 1;
  attr[0].val.clusterDim.y = NCH;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  auto k = attnpcc_kernel<STG, SK>;
  static bool set = false;
  if (!set) {
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024);
    cudaFuncSetAttribute(k, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    set = true;
  }
  const cudaError_t err = cudaLaunchKernelEx(
      &cfg, k, reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr()),
      reinterpret_cast<const long long*>(positions.data_ptr()), reinterpret_cast<const long long*>(cache_loc.data_ptr()),
      kscale.data_ptr<float>(), vscale.data_ptr<float>(), reinterpret_cast<uint8_t*>(kc.data_ptr()),
      reinterpret_cast<uint8_t*>(vc.data_ptr()), pt.data_ptr<int>(), seqlens.data_ptr<int>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), (float)theta_log2, (int)rope, (float)qscale, (int)HKV,
      (int)pt.size(1), (int)page, (int)window, (int)CH, (int)NCH, (int)qkv.size(1));
  TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));
}
"""
_attn_CPP_SRC = ("void attn(torch::Tensor q16, torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor seqlens, "
           "torch::Tensor ws_o, torch::Tensor ws_m, torch::Tensor ws_l, int64_t L, int64_t HKV, int64_t page, "
           "int64_t window, int64_t CH, int64_t NCH, int64_t batch, int64_t SK);\n"
           "void attn8(torch::Tensor q8, torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor seqlens, "
           "torch::Tensor kscale, torch::Tensor vscale, torch::Tensor ws_o, torch::Tensor ws_m, torch::Tensor ws_l, "
           "double qscale, int64_t L, int64_t HKV, int64_t page, int64_t window, int64_t CH, int64_t NCH, "
           "int64_t batch);\n"
           "void attnwg(torch::Tensor q8, torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor seqlens, "
           "torch::Tensor kscale, torch::Tensor vscale, torch::Tensor ws_o, torch::Tensor ws_m, torch::Tensor ws_l, "
           "double qscale, int64_t HKV, int64_t page, int64_t window, int64_t CH, int64_t NCH, int64_t batch, "
           "int64_t SK, int64_t HEADS);\n"
           "void attnpc(torch::Tensor q8, torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor seqlens, "
           "torch::Tensor kscale, torch::Tensor vscale, torch::Tensor ws_o, torch::Tensor ws_m, torch::Tensor ws_l, "
           "double qscale, int64_t HKV, int64_t page, int64_t window, int64_t CH, int64_t NCH, int64_t batch, "
           "int64_t SK);\n"
           "void attnc(torch::Tensor qkv, torch::Tensor positions, torch::Tensor cache_loc, torch::Tensor kscale, "
           "torch::Tensor vscale, torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor seqlens, "
           "torch::Tensor out, double theta_log2, int64_t rope, double q_mult, int64_t HKV, int64_t page, "
           "int64_t window, int64_t CH, int64_t NCH, int64_t batch, int64_t SK);\n"
           "void attnpcc(torch::Tensor qkv, torch::Tensor positions, torch::Tensor cache_loc, torch::Tensor kscale, "
           "torch::Tensor vscale, torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor seqlens, "
           "torch::Tensor out, double theta_log2, int64_t rope, double qscale, int64_t HKV, int64_t page, "
           "int64_t window, int64_t CH, int64_t NCH, int64_t batch);")
_attn_mod = None


def _attn_module():
    global _attn_mod
    if _attn_mod is None:
        _attn_mod = load_inline("muse_layer_attn_decode", _attn_CPP_SRC, cuda_sources=_attn_CUDA_SRC, functions=["attn", "attn8", "attnwg", "attnpc", "attnc", "attnpcc"],
                           extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
    return _attn_mod


def _attn_kernel_fn(data: Inputs) -> torch.Tensor:
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    group = hq // hkv
    use_cuda = group == 8 and hkv % 2 == 0 and d == 64 and (
        length == 1 or (length == 8 and cfg["window"] > 0 and batch <= _attn_CUDA_VERIFY_MAX_BATCH))
    if not use_cuda:
        return _attn_kernel_fa3(data)
    return _attn_kernel_fn_cuda(data)


def _attn_kernel_fn_cuda(data: Inputs) -> torch.Tensor:
    """The prep + CUDA attention + combine path (q_len 1 or 8, 8 query heads per KV head, even KV heads)."""
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    group = hq // hkv
    window = cfg["window"]
    tokens = batch * length
    qkv = data["qkv"].view(tokens, -1)
    dev = qkv.device
    theta = cfg["rope_theta"]
    k_cache, v_cache = data["k_cache"], data["v_cache"]
    if length == 8 and window and _attn_FUSED_VERIFY and triton.cdiv(window - 1 + length, _attn_vchunk(batch)) <= 16:
        return _attn_kernel_verify_cluster(data)
    fused = _attn_FUSED["sliding" if window else "full"] if length == 1 else None
    if fused and 16 % triton.cdiv(window or _attn_pt_span(data), _attn_chunk(window, batch, 1)) == 0:
        return _attn_kernel_cluster(data)
    if _attn_MMA8[length]:
        return _attn_kernel_cuda8(data)
    q16 = torch.empty((tokens, hq, d), dtype=torch.float16, device=dev)
    _attn_prep[(tokens,)](qkv, data["positions"], data["cache_loc"], data["k_scale"], data["v_scale"], q16, k_cache, v_cache,
                     math.log2(theta) if theta else 0.0, cfg["qk_scale_factor"] * cfg["sm_scale"] * _attn_LOG2E,
                     HQ=hq, HKV=hkv, D=d, QKV_W=qkv.shape[1], HAS_ROPE=bool(theta), num_warps=4)
    page = cfg["page_size"]
    pt = data["page_table"]
    ch = _attn_chunk(window, batch, length)
    span = (window - 1 + length) if window else pt.shape[1] * page  # keys a block can see, at most
    nch = triton.cdiv(span, ch)
    rh = max(16, group * length)
    rows = 2 * rh
    groups = batch * (hkv // 2)
    ws_o = torch.empty((groups * nch * rows, d), dtype=torch.float32, device=dev)
    ws_m = torch.empty((groups * nch * rows,), dtype=torch.float32, device=dev)
    ws_l = torch.empty_like(ws_m)
    # The kernel reads every key from the cache, the block's own included (prep stored them first). Descaling by
    # the cache scales happens on the output: scores scale by k_scale (folded into the queries) and values by v_scale.
    if cfg["kv_scale"] != 1.0:
        q16.mul_(data["k_scale"])
    _attn_module().attn(q16, k_cache.view(torch.uint8), v_cache.view(torch.uint8), pt, data["seq_lens"], ws_o, ws_m, ws_l,
                   length, hkv, page, window, ch, nch, batch, _attn_STAGE_KEYS["sliding" if window else "full"])
    if cfg["kv_scale"] != 1.0:
        ws_o.mul_(data["v_scale"])
    out = torch.empty((batch, length, hq * d), dtype=torch.bfloat16, device=dev)
    _attn_attn_combine[(groups * rows,)](qkv, out, ws_o, ws_m, ws_l, nch, HKV=hkv, KVPC=2, G=group, L=length, D=d,
                                    RH=rh, QKV_W=qkv.shape[1], HQ=hq, CT=64, num_warps=2)
    return out


def _attn_pt_span(data: Inputs) -> int:
    return data["page_table"].shape[1] * data["config"]["page_size"]


def _attn_kernel_cluster(data: Inputs) -> torch.Tensor:
    """Decode steps in one launch, each request's chunks one cluster (attnc): the prep, the attention, the merge."""
    cfg = data["config"]
    batch, hq, hkv, d = cfg["batch"], cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    window, theta, page = cfg["window"], cfg["rope_theta"], cfg["page_size"]
    qkv = data["qkv"].view(batch, -1)
    ch = _attn_chunk(window, batch, 1)
    nch = triton.cdiv(window or _attn_pt_span(data), ch)
    out = torch.empty((batch, 1, hq * d), dtype=torch.bfloat16, device=qkv.device)
    _attn_module().attnc(qkv, data["positions"], data["cache_loc"], data["k_scale"], data["v_scale"],
                    data["k_cache"].view(torch.uint8), data["v_cache"].view(torch.uint8), data["page_table"],
                    data["seq_lens"], out, math.log2(theta) if theta else 0.0, int(bool(theta)),
                    cfg["qk_scale_factor"] * cfg["sm_scale"] * _attn_LOG2E, hkv, page, window, ch, nch, batch,
                    _attn_STAGE_KEYS["sliding" if window else "full"] if _attn_CHUNK["sliding" if window else "full"] else
                    32 if ch <= 128 else 64)
    return out


def _attn_vchunk(batch: int) -> int:
    """Keys per CTA of attnpcc: at most 16 chunks (one cluster) of the 2,055 keys a sliding verify block sees."""
    return _attn_CHUNK["sliding"] or (136 if batch <= 4 else 512)


def _attn_kernel_verify_cluster(data: Inputs) -> torch.Tensor:
    """Sliding verify blocks in one launch, each (request, KV head)'s chunks one cluster (attnpcc)."""
    cfg = data["config"]
    batch, length, hq, hkv, d = cfg["batch"], cfg["q_len"], cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    window, theta, page = cfg["window"], cfg["rope_theta"], cfg["page_size"]
    qkv = data["qkv"].view(batch * length, -1)
    ch = _attn_vchunk(batch)
    out = torch.empty((batch, length, hq * d), dtype=torch.bfloat16, device=qkv.device)
    _attn_module().attnpcc(qkv, data["positions"], data["cache_loc"], data["k_scale"], data["v_scale"],
                      data["k_cache"].view(torch.uint8), data["v_cache"].view(torch.uint8), data["page_table"],
                      data["seq_lens"], out, math.log2(theta) if theta else 0.0, int(bool(theta)),
                      cfg["qk_scale_factor"] * cfg["sm_scale"] * _attn_LOG2E, hkv, page, window, ch,
                      triton.cdiv(window - 1 + length, ch), batch)
    return out


def _attn_kernel_cuda8(data: Inputs) -> torch.Tensor:
    """kernel_fn_cuda with E4M3 MMAs: the FA3 path's prep (E4M3 queries), attn8, combine."""
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    group = hq // hkv
    window = cfg["window"]
    tokens = batch * length
    qkv = data["qkv"].view(tokens, -1)
    dev = qkv.device
    theta = cfg["rope_theta"]
    k_cache, v_cache = data["k_cache"], data["v_cache"]
    q8 = torch.empty((tokens, hq, d), dtype=_attn_FP8, device=dev)
    _attn_qkv_prep[(tokens,)](qkv, data["positions"], data["cache_loc"], data["k_scale"], data["v_scale"], q8, k_cache,
                         v_cache, math.log2(theta) if theta else 0.0, HQ=hq, HKV=hkv, D=d, QKV_W=qkv.shape[1],
                         HAS_ROPE=bool(theta), num_warps=4)
    pt = data["page_table"]
    page = cfg["page_size"]
    ch = _attn_chunk(window, batch, length)
    span = (window - 1 + length) if window else pt.shape[1] * page
    nch = triton.cdiv(span, ch)
    rh = max(16, group * length)
    rows = 2 * rh
    groups = batch * (hkv // 2)
    ws_o = torch.empty((groups * nch * rows, d), dtype=torch.float32, device=dev)
    ws_m = torch.empty((groups * nch * rows,), dtype=torch.float32, device=dev)
    ws_l = torch.empty_like(ws_m)
    if _attn_WGMMA and length == 8 and _attn_VERIFY_PC["sliding" if window else "full"]:
        _attn_module().attnpc(q8.view(torch.uint8), k_cache.view(torch.uint8), v_cache.view(torch.uint8), pt, data["seq_lens"],
                         data["k_scale"], data["v_scale"], ws_o, ws_m, ws_l, cfg["qk_scale_factor"] * cfg["sm_scale"] * _attn_LOG2E,
                         hkv, page, window, ch, nch, batch, 64)
    elif _attn_WGMMA and length == 8:
        _attn_module().attnwg(q8.view(torch.uint8), k_cache.view(torch.uint8), v_cache.view(torch.uint8), pt, data["seq_lens"],
                         data["k_scale"], data["v_scale"], ws_o, ws_m, ws_l, cfg["qk_scale_factor"] * cfg["sm_scale"] * _attn_LOG2E,
                         hkv, page, window, ch, nch, batch, _attn_STAGE_KEYS["sliding" if window else "full"],
                         _attn_VERIFY_HEADS["sliding" if window else "full"])
    else:
        _attn_module().attn8(q8.view(torch.uint8), k_cache.view(torch.uint8), v_cache.view(torch.uint8), pt, data["seq_lens"],
                        data["k_scale"], data["v_scale"], ws_o, ws_m, ws_l, cfg["qk_scale_factor"] * cfg["sm_scale"] * _attn_LOG2E,
                        length, hkv, page, window, ch, nch, batch)
    out = torch.empty((batch, length, hq * d), dtype=torch.bfloat16, device=dev)
    _attn_attn_combine[(groups * rows,)](qkv, out, ws_o, ws_m, ws_l, nch, HKV=hkv, KVPC=2, G=group, L=length, D=d,
                                    RH=rh, QKV_W=qkv.shape[1], HQ=hq, CT=64, num_warps=2)
    return out

# ---- projections (muse_bf16_proj) ----


import torch
import triton
import triton.language as tl
from torch.utils.cpp_extension import load_inline


_proj_SKINNY_MAX_M = 32
# The router's split-K GEMM above _proj_SKINNY_MAX_M: (BM, BN, BK, S, warps, stages), from a sweep at m = 128, 256.
_proj_ROUTER_GEMM = (64, 64, 128, 8, 4, 4)
_proj_ROUTER_GEMM_MAX_M = 4096


@triton.jit
def _proj_router_gemm(x_ptr, hi_ptr, lo_ptr, out_ptr, ws_ptr, cnt_ptr, M, N, K,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, S: tl.constexpr):
    """out[M, N] (FP32) = x @ hi^T + x @ lo^T, split S ways over K. Partials go to an FP32 workspace and the
    last program of each tile (a counter it resets, so graph replays stay correct) sums them in order."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_s = tl.program_id(2)
    kblocks = K // BK
    kb0 = pid_s * kblocks // S
    kb1 = (pid_s + 1) * kblocks // S
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = kb0 * BK + tl.arange(0, BK)
    m_mask = offs_m[:, None] < M
    x_ptrs = x_ptr + offs_m[:, None] * K + offs_k[None, :]
    hi_ptrs = hi_ptr + offs_n[None, :] * K + offs_k[:, None]
    lo_ptrs = lo_ptr + offs_n[None, :] * K + offs_k[:, None]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(kb0, kb1):
        a = tl.load(x_ptrs, mask=m_mask, other=0.0)
        acc = tl.dot(a, tl.load(hi_ptrs), acc)
        acc = tl.dot(a, tl.load(lo_ptrs), acc)
        x_ptrs += BK
        hi_ptrs += BK
        lo_ptrs += BK
    tile = pid_m * tl.num_programs(1) + pid_n
    ws = ws_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(ws + pid_s * M * N, acc, mask=m_mask)
    tl.debug_barrier()
    arrived = tl.atomic_add(cnt_ptr + tile, 1, sem="acq_rel")
    if arrived == S - 1:
        tl.debug_barrier()
        total = tl.zeros((BM, BN), dtype=tl.float32)
        for s in range(S):
            total += tl.load(ws + s * M * N, mask=m_mask, other=0.0, cache_modifier=".cg")
        tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], total, mask=m_mask)
        tl.atomic_xchg(cnt_ptr + tile, 0)


def _proj_router_gemm_launch(x, state):
    bm, bn, bk, s, warps, stages = _proj_ROUTER_GEMM
    m, k = x.shape
    n = state["router_hi"].shape[0]
    out = torch.empty((m, n), dtype=torch.float32, device=x.device)
    _proj_router_gemm[(triton.cdiv(m, bm), n // bn, s)](x, state["router_hi"], state["router_lo"], out, state["ws"],
                                                   state["cnt"], m, n, k, BM=bm, BN=bn, BK=bk, S=s,
                                                   num_warps=warps, num_stages=stages)
    return out

_proj_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__device__ __forceinline__ void mma_16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                          uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t dsmem_addr(const void* p, int rank) {  // p in CTA `rank` of the cluster
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"((uint32_t)__cvta_generic_to_shared(p)), "r"(rank));
  return r;
}
__device__ __forceinline__ void st_dsmem(uint32_t a, float v) {
  asm volatile("st.shared::cluster.f32 [%0], %1;" ::"r"(a), "f"(v) : "memory");
}

// Weights are read once: don't keep them in L1.
__device__ __forceinline__ uint4 ld_stream(const void* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}

// Writes one output (FP32, or rounded to BF16). With ssq (the decode layer's pooler), also adds each token's sum of
// squares of its rounded outputs to ssq[token]: the BN lanes holding one token's outputs (aligned, one warp) sum
// theirs by shuffles and the first adds them, so the next kernel can apply an RMSNorm without another pass.
template <bool OUT_F32, int BN>
__device__ __forceinline__ void store_out(void* out, float* ssq, bool ok, int tok, size_t o, float sum) {
  if constexpr (OUT_F32) {
    if (ok) reinterpret_cast<float*>(out)[o] = sum;
  } else {
    const __nv_bfloat16 v = __float2bfloat16(sum);
    if (ok) reinterpret_cast<__nv_bfloat16*>(out)[o] = v;
    if (ssq != nullptr) {
      const float f = __bfloat162float(v);
      float sq = ok ? f * f : 0.f;
#pragma unroll
      for (int x = 1; x < BN; x <<= 1) sq += __shfl_xor_sync(0xffffffffu, sq, x);
      if (ok && (threadIdx.x & (BN - 1)) == 0) atomicAdd(ssq + tok, sq);
    }
  }
}

// out[M, N] = x[M, K] @ W[N, K]^T (+ x @ Wlo^T). Grid: (N / BN) * S CTAs of 128 threads, clusters of S
// along x; CTA rank r, warp w covers k in [(4r + w) * 32 * NSTEPS * CH, +32 * NSTEPS * CH), as CH chunks of
// NSTEPS 32-wide steps. Each chunk's loads are all issued before its MMAs; other resident warps hide the wait.
template <int MT, int RT, int NSTEPS, int CH, bool LO, bool OUT_F32>
__global__ void __launch_bounds__(128) skinny_kernel(const __nv_bfloat16* __restrict__ x,
                                                     const __nv_bfloat16* __restrict__ w,
                                                     const __nv_bfloat16* __restrict__ wlo,
                                                     void* __restrict__ out, int M, int N, int K, int S,
                                                     float* __restrict__ ssq) {
  constexpr int BN = 16 * RT, TK = 8 * MT, ELEMS = BN * TK;
  static_assert(BN == 16 || BN == 32, "a token's BN outputs are BN consecutive lanes of one warp");
  __shared__ float red[4][ELEMS];
  // Element i (in blocks of 128) is reduced by CTA (i / 128) % S, at local index li = (i / 128 / S) * 128 + i % 128.
  // MT = 1: every CTA of the cluster pushes its partial there, slot rank * EPO + li (EPO <= ELEMS / S + 127); larger
  // tiles (more partials per thread) pull them after the barrier instead, which measured faster for them.
  constexpr bool PUSH = MT == 1;
  __shared__ float rcv[PUSH ? ELEMS + 128 * 16 : ELEMS];

  cg::cluster_group cluster = cg::this_cluster();
  const int rank = cluster.block_rank();
  const int n0 = (blockIdx.x / S) * BN;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, q = lane & 3;
  // W is packed so that each warp's loads are contiguous: [tile][rank][warp][chunk][s][rt][h][lane][8].
  const size_t wwarp = ((size_t)(blockIdx.x / S) * S + rank) * 4 + warp;

  float acc[RT][MT][4];
#pragma unroll
  for (int rt = 0; rt < RT; ++rt)
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[rt][mt][c] = 0.f;

#pragma unroll 1
  for (int ch = 0; ch < CH; ++ch) {
    const size_t wbase = wwarp * CH + ch;
    const int kw = (int)wbase % (S * 4 * CH) * (32 * NSTEPS) + 8 * q;

    uint4 wv[NSTEPS][RT][2];
    uint4 lv[LO ? NSTEPS : 1][RT][2];
#pragma unroll
    for (int s = 0; s < NSTEPS; ++s)
#pragma unroll
      for (int rt = 0; rt < RT; ++rt)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const size_t off = (((wbase * NSTEPS + s) * RT + rt) * 2 + h) * 256 + lane * 8;
          wv[s][rt][h] = ld_stream(w + off);
          if constexpr (LO) lv[s][rt][h] = ld_stream(wlo + off);
        }

    uint4 xv[NSTEPS][MT];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      const int tok = mt * 8 + g;
#pragma unroll
      for (int s = 0; s < NSTEPS; ++s)
        xv[s][mt] = tok < M ? __ldg(reinterpret_cast<const uint4*>(x + (size_t)tok * K + kw + 32 * s))
                            : make_uint4(0, 0, 0, 0);
    }

#pragma unroll
    for (int s = 0; s < NSTEPS; ++s)
#pragma unroll
      for (int rt = 0; rt < RT; ++rt)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          const uint4 a = wv[s][rt][0], b = wv[s][rt][1], v = xv[s][mt];
          mma_16816(acc[rt][mt], a.x, b.x, a.y, b.y, v.x, v.y);
          mma_16816(acc[rt][mt], a.z, b.z, a.w, b.w, v.z, v.w);
          if constexpr (LO) {
            const uint4 c = lv[s][rt][0], d = lv[s][rt][1];
            mma_16816(acc[rt][mt], c.x, d.x, c.y, d.y, v.x, v.y);
            mma_16816(acc[rt][mt], c.z, d.z, c.w, d.w, v.z, v.w);
          }
        }
  }

  // Partials as [token][row]: c0, c1 are rows g, tokens 2q, 2q + 1; c2, c3 the same at row g + 8.
#pragma unroll
  for (int rt = 0; rt < RT; ++rt)
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const int row = rt * 16 + g + (c >= 2 ? 8 : 0), tok = mt * 8 + 2 * q + (c & 1);
        red[warp][tok * BN + row] = acc[rt][mt][c];
      }
  __syncthreads();
  if constexpr (PUSH) {
    // The CTA's partial of element i goes straight into its reducer's shared memory (DSMEM stores need no wait), so
    // one cluster barrier suffices and the reduction reads only local memory.
    const int EPO = (ELEMS + 128 * S - 1) / (128 * S) * 128;
    for (int i = threadIdx.x; i < ELEMS; i += 128) {
      const int blk = i >> 7, li = blk / S * 128 + (i & 127);
      st_dsmem(dsmem_addr(rcv + rank * EPO + li, blk % S), red[0][i] + red[1][i] + red[2][i] + red[3][i]);
    }
    cluster.sync();  // every partial has landed; nothing reads another CTA's memory after this
    for (int li = threadIdx.x; li < EPO; li += 128) {  // every thread runs the same count (EPO % 128 == 0)
      const int i = ((li >> 7) * S + rank) * 128 + (li & 127);
      const int tok = i / BN, row = i % BN;
      const bool ok = i < ELEMS && tok < M;
      float sum = 0.f;
      if (ok)
        for (int r = 0; r < S; ++r) sum += rcv[r * EPO + li];
      store_out<OUT_F32, BN>(out, ssq, ok, tok, (size_t)tok * N + n0 + row, sum);
    }
  } else {
    for (int i = threadIdx.x; i < ELEMS; i += 128) rcv[i] = red[0][i] + red[1][i] + red[2][i] + red[3][i];
    cluster.sync();
    // Each CTA of the cluster reduces every S-th block of 128 elements across the cluster and writes it.
    for (int i = rank * 128 + threadIdx.x; i < ELEMS; i += S * 128) {  // the same count for every thread
      const int tok = i / BN, row = i % BN;
      const bool ok = tok < M;
      float sum = 0.f;
      if (ok)
        for (int r = 0; r < S; ++r) sum += cluster.map_shared_rank(rcv, r)[i];
      store_out<OUT_F32, BN>(out, ssq, ok, tok, (size_t)tok * N + n0 + row, sum);
    }
    cluster.sync();  // keep this CTA's shared memory alive until the others have read it
  }
}

template <int MT, int RT, int NSTEPS, int CH, bool LO>
void launch(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& wlo, torch::Tensor& out, int S,
            float* ssq) {
  const int M = x.size(0), K = x.size(1), N = out.size(1);
  auto kernel = skinny_kernel<MT, RT, NSTEPS, CH, LO, LO>;
  static bool attr_set = false;
  if (!attr_set) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    attr_set = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((N / (16 * RT)) * S);
  cfg.blockDim = dim3(128);
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = S;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  cudaLaunchKernelEx(&cfg, kernel, reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
                     reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()),
                     reinterpret_cast<const __nv_bfloat16*>(wlo.data_ptr()), out.data_ptr(), M, N, K, S, ssq);
}

void skinny(torch::Tensor x, torch::Tensor w, torch::Tensor wlo, torch::Tensor out, int64_t mt, int64_t rt,
            int64_t ns, int64_t ch, bool lo, int64_t s, torch::Tensor ssq) {
  float* sq = ssq.numel() ? ssq.data_ptr<float>() : nullptr;
  TORCH_CHECK(!sq || (!lo && ssq.numel() >= x.size(0)), "skinny: sum of squares only for BF16 outputs");
#define CASE(MT, RT, NS, CH, LO) \
  if (mt == MT && rt == RT && ns == NS && ch == CH && lo == LO) return launch<MT, RT, NS, CH, LO>(x, w, wlo, out, s, sq);
  CASES
  TORCH_CHECK(false, "no skinny kernel for this config");
}
"""

_proj_CPP_SRC = ("void skinny(torch::Tensor x, torch::Tensor w, torch::Tensor wlo, torch::Tensor out, int64_t mt, "
           "int64_t rt, int64_t ns, int64_t ch, bool lo, int64_t s, torch::Tensor ssq);")

_proj_mod = None


def _proj_module():
    global _proj_mod
    if _proj_mod is None:
        cases = " ".join(f"CASE({mt}, {rt}, {ns}, {ch}, {'true' if layer == 'router' else 'false'})"
                         for (layer, mt), (rt, ns, ch, _) in _proj_CONFIGS.items())
        _proj_mod = load_inline("muse_layer_bf16_skinny", _proj_CPP_SRC, cuda_sources=_proj_CUDA_SRC.replace("CASES", cases),
                           functions=["skinny"],
                           extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
    return _proj_mod


def _proj_mt(m: int) -> int:
    return 1 if m <= 8 else 2 if m <= 16 else 4


# (layer, MT) -> (RT, NSTEPS, CH, S): row tiles of 16 per warp, 32-wide k steps per chunk, chunks per warp,
# CTAs per cluster (K = 128 * NSTEPS * CH * S). From sweeps on an H200 under CUDA graphs with a cold L2;
# shapes not listed (o, the whole unpooler) are faster in cuBLAS.
_proj_CONFIGS = {
    ("qkv_gate", 1): (1, 16, 1, 4), ("qkv_gate", 2): (2, 4, 2, 8), ("qkv_gate", 4): (2, 4, 4, 4),
    ("pooler", 1): (2, 8, 2, 4), ("pooler", 2): (2, 4, 4, 4), ("pooler", 4): (2, 4, 8, 2),
    ("unpooler", 1): (1, 4, 1, 8),
    ("router", 1): (1, 2, 2, 16), ("router", 2): (1, 4, 1, 16), ("router", 4): (2, 4, 1, 16),
}


def _proj_config(layer: str, m: int):
    """(MT, RT, NSTEPS, CH, S) for the skinny kernel, or None for cuBLAS."""
    if m > _proj_SKINNY_MAX_M or (layer, _proj_mt(m)) not in _proj_CONFIGS:
        return None
    return (_proj_mt(m),) + _proj_CONFIGS[(layer, _proj_mt(m))]


def _proj_pack(w: torch.Tensor, rt: int, ns: int, ch: int, s: int) -> torch.Tensor:
    """W [N, K] reordered (losslessly) so each warp's loads are contiguous:
    [tile][rank][warp][chunk][step][rt][h][lane][8]."""
    n, k = w.shape
    return (w.view(n // (16 * rt), rt, 2, 8, s, 4, ch, ns, 4, 8).permute(0, 4, 5, 6, 7, 1, 2, 3, 8, 9)
            .contiguous().view(n, k))


def _proj_packed(state, name, rt, ns, ch, s):
    key = (name, rt, ns, ch, s)
    if key not in state["packed"]:
        state["packed"][key] = _proj_pack(state[name], rt, ns, ch, s)
    return state["packed"][key]


def _proj_launch(x, name, lo_name, cfg, out, state, ssq=None):
    mt, rt, ns, ch, s = cfg
    w = _proj_packed(state, name, rt, ns, ch, s)
    wlo = w if lo_name is None else _proj_packed(state, lo_name, rt, ns, ch, s)
    _proj_module().skinny(x, w, wlo, out, mt, rt, ns, ch, lo_name is not None, s, state["none"] if ssq is None else ssq)
    return out


def _proj_prepare(weights: Weights):
    state = {name: w.contiguous() for name, w in weights.items()}
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state["router_hi"] = hi.contiguous()
    state["router_lo"] = lo.contiguous()
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()
    state["packed"] = {}
    state["none"] = torch.empty(0, dtype=torch.float32, device=router.device)
    _proj_module()
    # Pack every configuration up front: packing is untimed and must not happen in a timed call.
    for (layer, _), (rt, ns, ch, s) in _proj_CONFIGS.items():
        for name in (("router_hi", "router_lo") if layer == "router" else (layer,)):
            _proj_packed(state, name, rt, ns, ch, s)
    bm, bn, _, splits, _, _ = _proj_ROUTER_GEMM
    state["ws"] = torch.empty(splits * _proj_ROUTER_GEMM_MAX_M * router.shape[0], dtype=torch.float32, device=router.device)
    state["cnt"] = torch.zeros(triton.cdiv(_proj_ROUTER_GEMM_MAX_M, bm) * (router.shape[0] // bn), dtype=torch.int32,
                               device=router.device)
    # Compile the router GEMM outside any graph capture (one compile covers every m).
    _proj_router_gemm_launch(torch.zeros((64, router.shape[1]), dtype=torch.bfloat16, device=router.device), state)
    torch.cuda.synchronize()
    return state


def _proj_kernel_fn(data: Inputs, state) -> torch.Tensor:
    x, cfg = data["x"], data["config"]
    m, n, layer = cfg["m"], cfg["n"], cfg["layer"]
    skinny = _proj_config(layer, m)
    if layer == "router":
        if skinny is not None:
            out = torch.empty((m, n), dtype=torch.float32, device=x.device)
            return _proj_launch(x, "router_hi", "router_lo", skinny, out, state)
        if m <= _proj_ROUTER_GEMM_MAX_M:
            return _proj_router_gemm_launch(x, state)
        acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
        acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
        return acc
    if skinny is not None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        # The decode layer passes "ssq" for the pooler: each token's sum of squares of the output is added to it.
        return _proj_launch(x, layer, None, skinny, out, state, data.get("ssq"))
    return x @ state[layer][:n].T

# ---- experts (muse_moe_w4a16) ----


import torch
from torch.utils.cpp_extension import load_inline


LOCAL_START, LOCAL_COUNT = 0, 32
_moe_TOP_K = 8
_moe_POST_NORM_EPS = 1e-8
# Split-K ways of the gate/up and down GEMMs, and rows per warp (16 * _moe_RT).
# (largest m, (split-K ways of gate/up, of down)). From sweeps on an H200: 16 up to 16 tokens (finer units balance
# the few active experts), 8 above.
_moe_SPLITS = ((16, (16, 16)), (1 << 30, (8, 8)))
# (largest m, MT): the token tile is 8 * MT; an expert with more tokens takes several groups.
_moe_MT_BY_M = ((16, 1), (1 << 30, 2))
_moe_RT = 2
# Tokens up to which the FP32 accumulators are kept in state; above, they are allocated per call.
_moe_ACC_MAX_M = 512
_moe_NW = 8  # warps per CTA, matching the kernel's _moe_NW
_moe_ROUTE_INTS = 34 + 32 * 4 + 3 * _moe_NW * 32  # the kernel's R_PAIRS

_moe_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;
typedef __nv_bfloat16 bf16;
constexpr int _moe_NW = 8;

// Programmatic dependent launch: a kernel launched with programmatic stream serialization may start before the
// previous one finishes; griddep_wait() waits for it (and its memory), griddep_launch() lets the next one start.
__device__ __forceinline__ void griddep_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void griddep_launch() { asm volatile("griddepcontrol.launch_dependents;"); }

__device__ __forceinline__ void mma0(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                     uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%10,%10,%10,%10};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(0.f));
}

__device__ __forceinline__ void mma_acc(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                        uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t bmul2(uint32_t a, uint32_t b) {
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&a), *reinterpret_cast<const __nv_bfloat162*>(&b));
  return *reinterpret_cast<uint32_t*>(&r);
}

__device__ __forceinline__ uint4 ld_stream(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}

// A packed word holds four BF16 pairs of E2M1 values, bit-permuted at _moe_prepare time so each comes out with
// shifts and a mask: each value's sign at bit 15 (31), e at bits 8-7 (24-23), m at bit 6 (22) of its half.
constexpr uint32_t E2M1_MASK = 0x81C081C0u;
__device__ __forceinline__ void e2m1x8(uint32_t q, uint32_t& o0, uint32_t& o1, uint32_t& o2, uint32_t& o3) {
  o0 = q & E2M1_MASK;
  o1 = (q << 3) & E2M1_MASK;
  o2 = (q << 6) & E2M1_MASK;
  o3 = ((q >> 4) & 0x00C000C0u) | ((q >> 5) & 0x01000100u) | ((q << 1) & 0x80008000u);
}

// Bytes j, j + 1 of the 4 E4M3 scales in word, times 2^100, as a BF16 pair. Exact either way.
// EXACT: the hardware conversion. Otherwise, valid only for positive normal scales (_moe_prepare() checks): the byte's
// exponent and mantissa bits moved into BF16's, and the bias difference (127 - 7) plus 100 added.
template <bool EXACT>
__device__ __forceinline__ uint32_t scale_pair(uint32_t word, int j) {
  if constexpr (EXACT) {
    __nv_fp8_e4m3 v0, v1;
    v0.__x = (__nv_fp8_storage_t)((word >> (8 * j)) & 0xff);
    v1.__x = (__nv_fp8_storage_t)((word >> (8 * j + 8)) & 0xff);
    const __nv_bfloat162 b =
        __floats2bfloat162_rn(float(v0) * 1.2676506002282294e30f, float(v1) * 1.2676506002282294e30f);
    return *reinterpret_cast<const uint32_t*>(&b);
  } else {
    return __byte_perm(word, 0, 0x4040 | ((j + 1) << 8) | j) * 16u + 0x6E006E00u;
  }
}

__device__ __forceinline__ float e4m3(uint32_t byte) {
  __nv_fp8_e4m3 v;
  v.__x = (__nv_fp8_storage_t)byte;
  return float(v);
}

// ---------------------------------------------------------------------------------------------------------
// Routing, computed by every CTA of both gemms from ids (identically, so their unit numbering agrees), into
// shared memory R: meta = R[0, 34) (active experts, total token groups, groups of TK before active expert a),
// act[a], cnt[e], off[e] (start of expert e's pairs in pairs[]), then scratch, then pairs[]: per local expert,
// its pairs (token * 8 + slot) in pair order. Thread t of a pass reads token t's 8 ids (distinct experts).
constexpr int R_ACT = 34, R_CNT = R_ACT + 32, R_OFF = R_CNT + 32, R_RUN = R_OFF + 32, R_WC = R_RUN + 32;
constexpr int R_BALL = R_WC + _moe_NW * 32, R_PRE = R_BALL + _moe_NW * 32, R_PAIRS = R_PRE + _moe_NW * 32;

__device__ __forceinline__ unsigned local_bits(const int* __restrict__ ids, int t, int M, int ls, int lc, int (&e)[8]) {
  unsigned bits = 0;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    e[j] = t < M ? __ldg(ids + t * 8 + j) - ls : -1;
    if ((unsigned)e[j] < (unsigned)lc)
      bits |= 1u << e[j];
    else
      e[j] = -1;
  }
  return bits;
}

// A warp's ballots for the local experts among its 32 tokens' ids (only those present: at most 8 per token):
// BALL[warp][x] = which lanes route to expert x, WC[warp][x] = how many. The entries must start at zero.
__device__ __forceinline__ void route_ballots(unsigned bits, int warp, int lane, int* R, bool count) {
  unsigned any = __reduce_or_sync(0xffffffffu, bits);
  while (any) {
    const int x = __ffs(any) - 1;
    any &= any - 1;
    const unsigned b = __ballot_sync(0xffffffffu, (bits >> x) & 1);
    if (lane == 0) {
      R[R_BALL + warp * 32 + x] = (int)b;
      R[R_WC + warp * 32 + x] = __popc(b);
      if (count) atomicAdd(&R[R_CNT + x], __popc(b));
    }
  }
}

template <int TK>
__device__ void route_cta(const int* __restrict__ ids, int M, int ls, int lc, int* R) {
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const unsigned lt = (1u << lane) - 1;
  const bool one_pass = M <= _moe_NW * 32;
  if (tid < 32) {
    R[R_CNT + tid] = 0;
    R[R_RUN + tid] = 0;
  }
  R[R_WC + tid] = 0;  // _moe_NW * 32 threads: one entry each
  R[R_BALL + tid] = 0;
  __syncthreads();
#pragma unroll 1
  for (int t0 = 0; t0 < M; t0 += _moe_NW * 32) {  // counts (and, for one pass, the ballots)
    if (t0 + warp * 32 < M) {
      int e[8];
      route_ballots(local_bits(ids, t0 + tid, M, ls, lc, e), warp, lane, R, true);
      if (!one_pass) {
        __syncwarp();
        R[R_WC + tid] = 0;
        R[R_BALL + tid] = 0;
      }
    }
  }
  __syncthreads();
  if (warp == 0) {
    const int c = lane < lc ? R[R_CNT + lane] : 0, g = (c + TK - 1) / TK;
    int ic = c, ig = g;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int vc = __shfl_up_sync(0xffffffffu, ic, o), vg = __shfl_up_sync(0xffffffffu, ig, o);
      if (lane >= o) {
        ic += vc;
        ig += vg;
      }
    }
    const unsigned am = __ballot_sync(0xffffffffu, c > 0);
    R[R_OFF + lane] = ic - c;
    if (c > 0) {
      const int a = __popc(am & lt);
      R[R_ACT + a] = lane;
      R[2 + a] = ig - g;
    }
    if (lane == 31) {
      R[0] = __popc(am);
      R[1] = ig;
    }
  }
#pragma unroll 1
  for (int t0 = 0; t0 < M; t0 += _moe_NW * 32) {  // positions
    int e[8];
    const unsigned bits = local_bits(ids, t0 + tid, M, ls, lc, e);
    if (!one_pass && t0 + warp * 32 < M) route_ballots(bits, warp, lane, R, false);
    __syncthreads();
    {  // thread (warp w, lane x): expert x's pairs before warp w (earlier passes included)
      int c = R[R_RUN + lane];
      for (int v = 0; v < warp; ++v) c += R[R_WC + v * 32 + lane];
      R[R_PRE + warp * 32 + lane] = c;
    }
    __syncthreads();
    const int t = t0 + tid;
#pragma unroll
    for (int j = 0; j < 8; ++j)
      if (e[j] >= 0)
        R[R_PAIRS + R[R_OFF + e[j]] + R[R_PRE + warp * 32 + e[j]] +
          __popc((unsigned)R[R_BALL + warp * 32 + e[j]] & lt)] = t * 8 + j;
    if (tid < 32) R[R_RUN + tid] = R[R_PRE + (_moe_NW - 1) * 32 + tid] + R[R_WC + (_moe_NW - 1) * 32 + tid];
    __syncthreads();
    if (!one_pass) {
      R[R_WC + tid] = 0;
      R[R_BALL + tid] = 0;
      __syncthreads();
    }
  }
}
// ---------------------------------------------------------------------------------------------------------
// acc[pair, row] += x[pair] . W[le][row] for every pair routed to an active expert, in FP32 atomics.
// A CTA tile is _moe_NW * 16 * _moe_RT rows (each warp 16 * _moe_RT of them); K is split S ways (KC = K / S, NST steps of
// 128). A unit is (rank, active expert, tile); CTA b takes a contiguous run of units, ordered rank-major
// then expert, so consecutive units reuse the staged x[tokens, KC] in shared memory (scaled by 2^100).
// Each warp prefetches its next step's weights while computing the current one.
// MODE 0: x rows are tokens (pair >> 3), BF16. MODE 1: x is silu(acc1[pair, :I]) * acc1[pair, I:], rounded to
// BF16 as SGLang's intermediate is.
constexpr int STG = 2;  // weight pipeline stages per warp, SUB steps each
constexpr int SUB = 2;  // 128-wide steps per loop iteration

__device__ __host__ __forceinline__ size_t ring_offset(int tk, int xs) {
  return ((size_t)tk * xs * 2 + tk * 4 + 15) / 16 * 16;  // after xs and stok
}

__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"((uint32_t)__cvta_generic_to_shared(dst)), "l"(src));
}
__device__ __forceinline__ void cp_async8(void* dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"((uint32_t)__cvta_generic_to_shared(dst)), "l"(src));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N));
}

// Four BF16 values times rs and their norm weights, rounded back to BF16 (as F.rms_norm on BF16 input).
__device__ __forceinline__ uint2 norm4(uint2 v, const float* __restrict__ w, float rs) {
  const float4 wv = __ldg(reinterpret_cast<const float4*>(w));
  const __nv_bfloat162 a = __floats2bfloat162_rn(__uint_as_float(v.x << 16) * rs * wv.x,
                                                 __uint_as_float(v.x & 0xffff0000u) * rs * wv.y);
  const __nv_bfloat162 b = __floats2bfloat162_rn(__uint_as_float(v.y << 16) * rs * wv.z,
                                                 __uint_as_float(v.y & 0xffff0000u) * rs * wv.w);
  return make_uint2(*reinterpret_cast<const uint32_t*>(&a), *reinterpret_cast<const uint32_t*>(&b));
}

__device__ __forceinline__ void red_add(float* p, float v) {
  asm volatile("red.global.add.f32 [%0], %1;" ::"l"(p), "f"(v) : "memory");
}

template <int MT, int _moe_RT, int MODE, bool EXACT, int KC>
__global__ void __launch_bounds__(_moe_NW * 32) moe_gemm(const bf16* __restrict__ x, const float* __restrict__ h1,
                                                     const uint4* __restrict__ wq, const uint2* __restrict__ wsf,
                                                     const float* __restrict__ gscale, const int* __restrict__ ids,
                                                     int local_start, int local_count, int* __restrict__ rglob, int M,
                                                     float* __restrict__ acc_out,
                                                     int out_stride, int K, int tiles, int S,
                                                     const float* __restrict__ pss, const float* __restrict__ pnw,
                                                     float peps) {
  constexpr int TK = 8 * MT, ROWS = _moe_NW * 16 * _moe_RT;
  constexpr int XS = KC + 32;  // row stride (bytes = 64 mod 128) keeps 16-byte reads conflict-free
  constexpr int nst = KC / 128;
  constexpr int nsg = nst / SUB;  // loop iterations per unit, SUB steps each
  extern __shared__ __align__(16) unsigned char smem[];
  bf16* xs = reinterpret_cast<bf16*>(smem);
  int* stok = reinterpret_cast<int*>(xs + TK * XS);

  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, q = lane & 3;
  // gemm1 waits for the previous kernel (the last call's norm zeroes h1). gemm2 routes and prefetches its first
  // weights, then waits for gemm1 (its x).
  if constexpr (MODE == 0) {
    griddep_wait();
    if (blockIdx.x != 0) griddep_launch();
  }
  int* R = reinterpret_cast<int*>(smem + ring_offset(TK, XS) + (size_t)_moe_NW * STG * SUB * _moe_RT * 2 * (32 * 16 + 8 * 8));
  if constexpr (MODE == 0) {
    route_cta<TK>(ids, M, local_start, local_count, R);
    // CTA 0 publishes the routing (meta, act, cnt, off, then the pairs) before it lets gemm2 launch.
    if (blockIdx.x == 0) {
      const int np = R[R_OFF + 31] + R[R_CNT + 31];
      for (int i = threadIdx.x; i < R_OFF + 32; i += _moe_NW * 32) rglob[i] = R[i];
      for (int i = threadIdx.x; i < np; i += _moe_NW * 32) rglob[R_PAIRS + i] = R[R_PAIRS + i];
      __threadfence();
      __syncthreads();
      griddep_launch();
    }
  } else {
    // gemm2 launches only after gemm1's CTA 0 published (L2 reads: this SM's L1 may hold the last call's).
    for (int i = threadIdx.x; i < R_OFF + 32; i += _moe_NW * 32) R[i] = __ldcg(rglob + i);
    __syncthreads();
  }
  const int* meta = R;
  const int* act = R + R_ACT;
  const int* cnt = R + R_CNT;
  const int na = meta[0];
  const int gt = meta[1];  // token groups over the active experts
  const int units = S * gt * tiles;
  const int u0 = (int)((long long)blockIdx.x * units / gridDim.x);
  const int u1 = (int)((long long)(blockIdx.x + 1) * units / gridDim.x);

  // Unit u = (rank * gt + groups before expert a + grp) * tiles + tile: one tile of one token group, so units are
  // equal work and a contiguous run of them balances. The CTA walks [u0, u1) as runs with the same (rank, expert,
  // token group), x staged once per run, then tile, then step. Incremental, so no divisions per step.
  struct It {
    int left, rank, a, le, tb, te, tile, grp, ngrp, st;
    size_t slot;  // the warp's (tile, rank, step) slot in the packed weights, kept incrementally
  };
  const size_t tile_step = ((size_t)S * _moe_NW * nst - (nst - SUB)) * _moe_RT * 2;  // slot (tile + 1, 0) - (tile, nst - SUB)
  auto set_run = [&](It& it) {
    it.le = act[it.a];
    it.ngrp = (cnt[it.le] + TK - 1) / TK;
    it.te = min(tiles, it.tb + it.left);
    it.tile = it.tb;
    it.st = 0;
    it.slot = (((((size_t)it.le * tiles + it.tile) * S + it.rank) * _moe_NW + warp) * nst) * _moe_RT * 2;
  };
  auto next = [&](It& it) {
    if (++it.st < nsg) {
      it.slot += SUB * _moe_RT * 2;
      return;
    }
    it.st = 0;
    if (++it.tile < it.te) {
      it.slot += tile_step;
      return;
    }
    it.left -= it.te - it.tb;
    if (it.left <= 0) return;
    it.tb = 0;
    if (++it.grp == it.ngrp) {
      it.grp = 0;
      if (++it.a == na) {
        it.a = 0;
        ++it.rank;
      }
    }
    set_run(it);
  };
  It cur;
  cur.left = u1 - u0;
  if (cur.left > 0) {
    cur.rank = u0 / (tiles * gt);
    const int gidx = (u0 / tiles) % gt;
    cur.a = 0;
    while (cur.a + 1 < na && meta[2 + cur.a + 1] <= gidx) ++cur.a;
    cur.grp = gidx - meta[2 + cur.a];
    cur.tb = u0 % tiles;
    set_run(cur);
  }

  // Each warp streams its weights through a ring of STG stages in shared memory (cp.async). A lane copies and
  // later reads only its own 16-byte chunks; each row's 8 scale bytes are copied once (by its q = 0 lane) and
  // read by the row's 4 lanes after a __syncwarp.
  uint4* wring = reinterpret_cast<uint4*>(smem + ring_offset(TK, XS)) + warp * STG * SUB * _moe_RT * 2 * 32;
  uint2* sring = reinterpret_cast<uint2*>(reinterpret_cast<uint4*>(smem + ring_offset(TK, XS)) +
                                          _moe_NW * STG * SUB * _moe_RT * 2 * 32) +
                 warp * STG * SUB * _moe_RT * 2 * 8;
  auto issue = [&](int stage, size_t slot0) {  // SUB consecutive steps: their slots are contiguous
#pragma unroll
    for (int j = 0; j < SUB * _moe_RT * 2; ++j) {
      const int i = stage * SUB * _moe_RT * 2 + j;
      cp_async16(wring + i * 32 + lane, wq + (slot0 + j) * 32 + lane);
    }
    // The steps' scale blocks (64 B each) are contiguous too: SUB * _moe_RT * 2 * 64 B as 16 B per lane.
    static_assert(SUB * _moe_RT * 2 * 64 == 32 * 16, "one 16-byte scale copy per lane per stage");
    cp_async16(reinterpret_cast<uint4*>(sring + stage * SUB * _moe_RT * 2 * 8) + lane,
               reinterpret_cast<const uint4*>(wsf + slot0 * 8) + lane);
  };

  float acc[_moe_RT][MT][4];
  // The loader runs STG - 1 steps ahead of the compute.
  It ld = cur;
#pragma unroll
  for (int i = 0; i < STG - 1; ++i) {
    if (ld.left > 0) {
      issue(i, ld.slot);
      next(ld);
    }
    cp_async_commit();
  }
  if constexpr (MODE == 1) {
    griddep_wait();
    griddep_launch();
  }
  int stage = 0;

  while (cur.left > 0) {
    const int le = cur.le, tile = cur.tile, rank = cur.rank, grp = cur.grp, st = cur.st;
    if (st == 0) {
      if (tile == cur.tb) {  // a new run (expert, token group): stage its x
        __syncthreads();  // everyone is done reading xs / stok
        const int c = cnt[le], t0 = grp * TK;
        if (threadIdx.x < TK) stok[threadIdx.x] = t0 + (int)threadIdx.x < c ? (MODE == 0 ? R[R_PAIRS + R[R_OFF + le] + t0 + threadIdx.x]
                                                                         : __ldcg(rglob + R_PAIRS + R[R_OFF + le] + t0 + threadIdx.x)) : -1;
        __syncthreads();
        // xs[r] holds, per 128-wide step and per (w, q), the 16 bytes x[16 (2w) + 4q + i], x[16 (2w + 1) + 4q + i]
        // interleaved over i = 0..3: the B operands of the two MMAs of word w, in one 16-byte read.
        constexpr int chunks = KC / 8;
        for (int i = threadIdx.x; i < TK * chunks; i += _moe_NW * 32) {
          const int r = i / chunks, ch = i - r * chunks;
          const int p = stok[r];
          uint4 v = make_uint4(0, 0, 0, 0);
          if (p >= 0) {
            const int k0 = rank * KC + (ch >> 4) * 128 + ((ch >> 2) & 3) * 32 + (ch & 3) * 4;  // block 2w; +16: 2w + 1
            uint2 lo, hi;
            if constexpr (MODE == 0) {
              const bf16* xr = x + (size_t)(p >> 3) * K + k0;
              lo = __ldg(reinterpret_cast<const uint2*>(xr));
              hi = __ldg(reinterpret_cast<const uint2*>(xr + 16));
              if (pss != nullptr) {  // the decode layer's pre-expert RMSNorm, folded in: x * rsqrt(ss / K + eps) * nw
                const float rs = rsqrtf(__ldcg(pss + (p >> 3)) / K + peps);
                lo = norm4(lo, pnw + k0, rs);
                hi = norm4(hi, pnw + k0 + 16, rs);
              }
            } else {
              const float* gp = h1 + (size_t)p * 2 * K + k0;
              auto act4 = [&](const float* gq) {
                const float4 gv = *reinterpret_cast<const float4*>(gq), uv = *reinterpret_cast<const float4*>(gq + K);
                const __nv_bfloat162 a = __floats2bfloat162_rn(gv.x / (1.f + __expf(-gv.x)) * uv.x,
                                                               gv.y / (1.f + __expf(-gv.y)) * uv.y);
                const __nv_bfloat162 b = __floats2bfloat162_rn(gv.z / (1.f + __expf(-gv.z)) * uv.z,
                                                               gv.w / (1.f + __expf(-gv.w)) * uv.w);
                return make_uint2(*reinterpret_cast<const uint32_t*>(&a), *reinterpret_cast<const uint32_t*>(&b));
              };
              lo = act4(gp);
              hi = act4(gp + 16);
            }
            v = make_uint4(__byte_perm(lo.x, hi.x, 0x5410), __byte_perm(lo.x, hi.x, 0x7632),
                           __byte_perm(lo.y, hi.y, 0x5410), __byte_perm(lo.y, hi.y, 0x7632));
          }
          *reinterpret_cast<uint4*>(xs + r * XS + ch * 8) = v;
        }
        __syncthreads();
      }
#pragma unroll
      for (int rt = 0; rt < _moe_RT; ++rt)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
          for (int i = 0; i < 4; ++i) acc[rt][mt][i] = 0.f;
    }

    cp_async_wait<STG - 2>();  // this step's stage has landed
    __syncwarp();              // including the scale bytes other lanes copied
    {
      const int ns = stage == 0 ? STG - 1 : stage - 1;  // the stage the previous step used
      if (ld.left > 0) {
        issue(ns, ld.slot);
        next(ld);
      }
      cp_async_commit();
    }
#pragma unroll
    for (int sub = 0; sub < SUB; ++sub) {
    uint4 wv[_moe_RT][2];
    uint2 sv[_moe_RT][2];
#pragma unroll
    for (int rt = 0; rt < _moe_RT; ++rt)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int i = (stage * SUB + sub) * _moe_RT * 2 + rt * 2 + h;
        wv[rt][h] = wring[i * 32 + lane];
        sv[rt][h] = sring[i * 8 + g];
      }

    const bf16* xq = xs + g * XS + (st * SUB + sub) * 128 + 8 * q;
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      uint4 xv[MT];
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) xv[mt] = *reinterpret_cast<const uint4*>(xq + mt * 8 * XS + 32 * w);
#pragma unroll
      for (int rt = 0; rt < _moe_RT; ++rt) {
        const uint32_t w0[4] = {wv[rt][0].x, wv[rt][0].y, wv[rt][0].z, wv[rt][0].w};
        const uint32_t w1[4] = {wv[rt][1].x, wv[rt][1].y, wv[rt][1].z, wv[rt][1].w};
        const uint32_t s0[2] = {sv[rt][0].x, sv[rt][0].y};
        const uint32_t s1[2] = {sv[rt][1].x, sv[rt][1].y};
        // Word w holds blocks 2w and 2w + 1 of row g (w0) and row g + 8 (w1): pair i is the values at position
        // 4q + i of the two blocks. Scaling each pair by (block 2w's, block 2w + 1's) scale (times 2^100) is exact
        // in BF16 (E2M1 x E4M3), and with the 2^-126 of the E2M1 placement the products stay normal. Pairs 0, 1
        // are the first MMA's k pairs, 2, 3 the second's.
        uint32_t r0[4], r1[4];
        e2m1x8(w0[w], r0[0], r0[1], r0[2], r0[3]);
        e2m1x8(w1[w], r1[0], r1[1], r1[2], r1[3]);
        const uint32_t f0 = scale_pair<EXACT>(s0[w >> 1], (w & 1) * 2);
        const uint32_t f1 = scale_pair<EXACT>(s1[w >> 1], (w & 1) * 2);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          r0[i] = bmul2(r0[i], f0);
          r1[i] = bmul2(r1[i], f1);
        }
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          mma_acc(acc[rt][mt], r0[0], r1[0], r0[1], r1[1], xv[mt].x, xv[mt].y);
          mma_acc(acc[rt][mt], r0[2], r1[2], r0[3], r1[3], xv[mt].z, xv[mt].w);
        }
      }
    }
    }  // sub

    if (st == nsg - 1) {
      const float gs = gscale[le] * 67108864.f;  // 2^26
      const int row0 = tile * ROWS + warp * 16 * _moe_RT + g;
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int p = stok[mt * 8 + 2 * q + (i & 1)];
          if (p < 0) continue;
#pragma unroll
          for (int rt = 0; rt < _moe_RT; ++rt)
            red_add(acc_out + (size_t)p * out_stride + row0 + rt * 16 + (i >= 2 ? 8 : 0), acc[rt][mt][i] * gs);
        }
    }
    next(cur);
    stage = stage == STG - 1 ? 0 : stage + 1;
  }
  cp_async_wait<0>();
}

// ---------------------------------------------------------------------------------------------------------
// Grid (M, 1 + ZC), 512 threads. blockIdx.y == 0: out[t] = sum over local slots s (in order) of
// weight[t, s] * rmsnorm(y[t * 8 + s] * nw), nw = 1 + post_norm_w, then zeroes those y rows. blockIdx.y >= 1:
// zeroes chunk y - 1 of ZC of token t's local gate/up rows h1. Both accumulators are zero again for the next call.
constexpr int NORM_ZC = 4;
__global__ void __launch_bounds__(512) norm_kernel(float* __restrict__ y, float* __restrict__ h1,
                                                   const int* __restrict__ ids, const float* __restrict__ tw,
                                                   const float* __restrict__ nw, int local_start, int local_count,
                                                   float eps, bf16* __restrict__ out, float* __restrict__ pss) {
  constexpr int H = 4096, I2 = 8192;
  const int t = blockIdx.x;
  const float4 zero = make_float4(0.f, 0.f, 0.f, 0.f);
  bool local[8];
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    const int e = ids[t * 8 + s] - local_start;
    local[s] = e >= 0 && e < local_count;
  }
  // The routing, weights and norm weight are inputs: read them before waiting for gemm2.
  if (blockIdx.y > 0) {
    griddep_wait();
    const int chunk = I2 / NORM_ZC, c = (blockIdx.y - 1) * chunk + threadIdx.x * 4;
#pragma unroll
    for (int s = 0; s < 8; ++s)
      if (local[s] && threadIdx.x * 4 < chunk) *reinterpret_cast<float4*>(h1 + (size_t)(t * 8 + s) * I2 + c) = zero;
    return;
  }
  __shared__ float wsum[16][8];
  const int c0 = threadIdx.x * 8;
  float nwv[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) nwv[i] = nw[c0 + i];
  float twv[8];
#pragma unroll
  for (int s = 0; s < 8; ++s) twv[s] = local[s] ? tw[t * 8 + s] : 0.f;
  griddep_wait();
  if (pss != nullptr && threadIdx.x == 0) pss[t] = 0.f;  // gemm1 has read it: zero again for the next call
  float v[8][8], ss[8];
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    float4 y0 = zero, y1 = zero;
    if (local[s]) {
      float4* yr = reinterpret_cast<float4*>(y + (size_t)(t * 8 + s) * H + c0);
      y0 = yr[0];
      y1 = yr[1];
      yr[0] = zero;
      yr[1] = zero;
    }
    const float yv[8] = {y0.x, y0.y, y0.z, y0.w, y1.x, y1.y, y1.z, y1.w};
    ss[s] = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      v[s][i] = yv[i] * nwv[i];
      ss[s] += v[s][i] * v[s][i];
    }
  }
#pragma unroll
  for (int s = 0; s < 8; ++s)
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss[s] += __shfl_xor_sync(0xffffffffu, ss[s], o);
  if ((threadIdx.x & 31) == 0)
#pragma unroll
    for (int s = 0; s < 8; ++s) wsum[threadIdx.x >> 5][s] = ss[s];
  __syncthreads();
  float acc[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) acc[i] = 0.f;
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    if (!local[s]) continue;
    float tot = 0.f;
#pragma unroll
    for (int w = 0; w < 16; ++w) tot += wsum[w][s];
    const float scale = twv[s] * rsqrtf(tot / H + eps);
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = fmaf(v[s][i], scale, acc[i]);
  }
  uint4 o;
  bf16* ov = reinterpret_cast<bf16*>(&o);
#pragma unroll
  for (int i = 0; i < 8; ++i) ov[i] = __float2bfloat16(acc[i]);
  *reinterpret_cast<uint4*>(out + (size_t)t * H + c0) = o;
}

// ---------------------------------------------------------------------------------------------------------
template <int MT, int _moe_RT, int MODE, bool EXACT, int KC>
void launch_gemm_kc(const torch::Tensor& x, const torch::Tensor& h1, const torch::Tensor& wq, const torch::Tensor& wsf,
                 const torch::Tensor& gs, const torch::Tensor& ids, int ls, int lc, const torch::Tensor& rglob, int M,
                 torch::Tensor& out, int K,
                 int tiles, int S, const float* pss, const float* pnw, float peps) {
  constexpr int TK = 8 * MT;
  auto kernel = moe_gemm<MT, _moe_RT, MODE, EXACT, KC>;
  const size_t smem = ring_offset(TK, KC + 32) + (size_t)_moe_NW * STG * SUB * _moe_RT * 2 * (32 * 16 + 8 * 8) +
                      4 * ((size_t)R_PAIRS + 8 * (size_t)M);
  TORCH_CHECK(smem <= 227 * 1024, "too many tokens for the in-CTA routing");
  // A persistent grid: as many CTAs as fit at once, computed once per (split, shared memory size).
  static int cache_s[64], cache_blocks[64], cache_n = 0;
  static size_t cache_smem[64], max_smem = 0;
  int blocks = 0;
  for (int i = 0; i < cache_n; ++i)
    if (cache_s[i] == S && cache_smem[i] == smem) blocks = cache_blocks[i];
  if (blocks == 0) {
    if (smem > max_smem) {
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
      max_smem = smem;
    }
    int per_sm = 0, sms = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, _moe_NW * 32, smem);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    blocks = (per_sm < 1 ? 1 : per_sm) * sms;
    if (cache_n < 64) {
      cache_s[cache_n] = S;
      cache_smem[cache_n] = smem;
      cache_blocks[cache_n++] = blocks;
    }
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(blocks);
  cfg.blockDim = dim3(_moe_NW * 32);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  cudaLaunchKernelEx(&cfg, kernel, reinterpret_cast<const bf16*>(x.data_ptr()), h1.data_ptr<float>(),
                     reinterpret_cast<const uint4*>(wq.data_ptr()), reinterpret_cast<const uint2*>(wsf.data_ptr()),
                     gs.data_ptr<float>(), ids.data_ptr<int>(), ls, lc, rglob.data_ptr<int>(), M, out.data_ptr<float>(), (int)out.size(1), K,
                     tiles, S, pss, pnw, peps);
}

// K per split (KC) is a compile-time constant of the kernel: 512 (split 8) or 256 (split 16) of K = 4096.
template <int MT, int _moe_RT, int MODE, bool EXACT>
void launch_gemm(const torch::Tensor& x, const torch::Tensor& h1, const torch::Tensor& wq, const torch::Tensor& wsf,
                 const torch::Tensor& gs, const torch::Tensor& ids, int ls, int lc, const torch::Tensor& rglob, int M,
                 torch::Tensor& out, int K,
                 int tiles, int S, const float* pss = nullptr, const float* pnw = nullptr, float peps = 0.f) {
  if (K / S == 512)
    return launch_gemm_kc<MT, _moe_RT, MODE, EXACT, 512>(x, h1, wq, wsf, gs, ids, ls, lc, rglob, M, out, K, tiles, S, pss, pnw, peps);
  if (K / S == 256)
    return launch_gemm_kc<MT, _moe_RT, MODE, EXACT, 256>(x, h1, wq, wsf, gs, ids, ls, lc, rglob, M, out, K, tiles, S, pss, pnw, peps);
  TORCH_CHECK(false, "moe_gemm is built for K / S = 512 or 256");
}

void moe_forward(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor w13q, torch::Tensor w13sf,
                 torch::Tensor w13gs, torch::Tensor w2q, torch::Tensor w2sf, torch::Tensor w2gs, torch::Tensor nw,
                 torch::Tensor rglob, torch::Tensor h1, torch::Tensor y, torch::Tensor out, int64_t mt, int64_t rt, int64_t S1, int64_t S2,
                 int64_t local_start, int64_t local_count, double eps, bool exact, torch::Tensor pre_ss,
                 torch::Tensor pre_w, double pre_eps) {
  const int M = x.size(0), H = x.size(1), I = h1.size(1) / 2;
  // pre_ss / pre_w non-empty (the decode layer): x is the pooler's output, pre_ss its per-token sum of squares, and
  // gemm1 applies the pre-expert RMSNorm (weight pre_w) as it stages x; the norm kernel zeroes pre_ss.
  float* pss = pre_ss.numel() ? pre_ss.data_ptr<float>() : nullptr;
  const float* pnw = pre_w.numel() ? pre_w.data_ptr<float>() : nullptr;
  TORCH_CHECK(!pss || (pnw && pre_w.numel() == H && pre_ss.numel() >= M), "moe: pre-norm buffers");
  auto stream = at::cuda::getCurrentCUDAStream();
  const int ls = (int)local_start, lc = (int)local_count;
  const int rows = _moe_NW * 16 * (int)rt;
#define GEMMS_(MT_, RT_, EX_)                                                                                    \
  if (mt == MT_ && rt == RT_ && exact == EX_) {                                                                  \
    launch_gemm<MT_, RT_, 0, EX_>(x, h1, w13q, w13sf, w13gs, ids, ls, lc, rglob, M, h1, H, 2 * I / rows, S1,  \
                                  pss, pnw, (float)pre_eps);                                                    \
    launch_gemm<MT_, RT_, 1, EX_>(x, h1, w2q, w2sf, w2gs, ids, ls, lc, rglob, M, y, I, H / rows, S2);          \
  } else
#define GEMMS(MT_, RT_) GEMMS_(MT_, RT_, false) GEMMS_(MT_, RT_, true)
  GEMMS(1, 2) GEMMS(2, 2) GEMMS(4, 2)
  TORCH_CHECK(false, "no moe_gemm for this config");
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(M, 1 + NORM_ZC);
  cfg.blockDim = dim3(512);
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  cudaLaunchKernelEx(&cfg, norm_kernel, y.data_ptr<float>(), h1.data_ptr<float>(), (const int*)ids.data_ptr<int>(),
                     (const float*)tw.data_ptr<float>(), (const float*)nw.data_ptr<float>(), (int)local_start,
                     (int)local_count, (float)eps, reinterpret_cast<bf16*>(out.data_ptr()), pss);
}
"""

_moe_CPP_SRC = ("void moe_forward(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor w13q, "
           "torch::Tensor w13sf, torch::Tensor w13gs, torch::Tensor w2q, torch::Tensor w2sf, torch::Tensor w2gs, "
           "torch::Tensor nw, torch::Tensor rglob, torch::Tensor h1, torch::Tensor y, torch::Tensor out, int64_t mt, int64_t rt, int64_t S1, int64_t S2, "
           "int64_t local_start, int64_t local_count, double eps, bool exact, torch::Tensor pre_ss, "
           "torch::Tensor pre_w, double pre_eps);")

_moe_mod = None


def _moe_module():
    global _moe_mod
    if _moe_mod is None:
        _moe_mod = load_inline("muse_layer_moe_nvfp4", _moe_CPP_SRC, cuda_sources=_moe_CUDA_SRC, functions=["moe_forward"],
                           extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
    return _moe_mod


def _moe_unpack_codes(packed: torch.Tensor) -> torch.Tensor:
    """uint8 [..., K/2] -> uint8 E2M1 codes [..., K], element 2i from the low nibble."""
    return torch.stack((packed & 0x0F, packed >> 4), dim=-1).flatten(-2)


def _moe_bit_map():
    """(value, code bit, stored bit) for a packed word; see e2m1x8 in the kernel. Value v = 4b + r is block
    2w + b at position 4q + r; it is output k = r, half b. Output k's bits land (after the kernel's shift) at
    m: 6, e: 7-8, sign: 15, plus 16 for the high half."""
    out = []
    o3 = {6: 10, 7: 11, 8: 13, 15: 14}
    for b in range(2):
        for r in range(4):
            k, half = r, b
            for cb, tgt in enumerate((6, 7, 8, 15)):
                stored = tgt - (0, 3, 6)[k] if k < 3 else o3[tgt]
                out.append((4 * b + r, cb, stored + 16 * half))
    assert sorted(p for _, _, p in out) == list(range(32))
    return out


_moe_BIT_MAP = _moe_bit_map()


def _moe_scales_simple(sf: torch.Tensor) -> bool:
    """Whether every E4M3 scale is positive and normal (the kernel's fast conversion), else the exact one."""
    b = sf.view(torch.uint8)
    return bool(((b & 0x80) == 0).all() and ((b & 0x78) != 0).all() and ((b & 0x7f) != 0x7f).all())


def _moe_pack_expert(q: torch.Tensor, sf: torch.Tensor, rt: int, s: int):
    """One expert's weights [N, K] into the kernel's layout.

    Rows: row = ((tile * _moe_NW + warp) * _moe_RT + rt) * 16 + h * 8 + g. K: k = (rank * NST + step) * 128 + (2w + b) * 16
    + 4q + 2 r_hi + r_lo. Packed order: tile, rank, warp, step, rt, h, g, q, then per lane 32 nibbles
    (w, r_lo, b, r_hi). Scales: tile, rank, warp, step, rt, h, g, then the step's 8 block scales."""
    n, k = q.shape[0], q.shape[1] * 2
    tiles, nst = n // (_moe_NW * 16 * rt), k // (s * 128)
    codes = _moe_unpack_codes(q).view(tiles, _moe_NW, rt, 2, 8, s, nst, 4, 2, 4, 4)
    # Per lane word w: the 8 values (b, r) = (block 2w + b, position 4q + r), each code bit to its place.
    codes = codes.permute(0, 5, 1, 6, 2, 3, 4, 9, 7, 8, 10).contiguous().view(-1, 8).to(torch.int32)
    word = torch.zeros(codes.shape[0], dtype=torch.int32, device=q.device)
    for v, cb, pos in _moe_BIT_MAP:
        word |= ((codes[:, v] >> cb) & 1) << pos
    packed = word.view(torch.uint8)
    sfv = sf.view(torch.uint8).view(tiles, _moe_NW, rt, 2, 8, s, nst, 8)
    return packed, sfv.permute(0, 5, 1, 6, 2, 3, 4, 7).contiguous().view(-1)


def _moe_config(m: int):
    """(MT, _moe_RT, S1, S2): token tile of 8 * MT, 16 * _moe_RT rows per warp, split-K ways of the two GEMMs."""
    mt = next(t for top, t in _moe_MT_BY_M if m <= top)
    s1, s2 = next(sp for top, sp in _moe_SPLITS if m <= top)
    return mt, _moe_RT, s1, s2


def _moe_prepare(weights: Weights):
    dev = weights["w13"].device
    inner = weights["w13"].shape[1] // 2
    hidden = weights["w2"].shape[1]
    lo, hi = LOCAL_START, LOCAL_START + LOCAL_COUNT
    # One packing of each projection per split it is used with (the split changes the k order).
    packs = {}
    for name, idx in (("w13", 0), ("w2", 1)):
        for split in sorted({sp[idx] for _, sp in _moe_SPLITS}):
            qs, sfs = zip(*(_moe_pack_expert(weights[name][e], weights[name + "_sf"][e], _moe_RT, split) for e in range(lo, hi)))
            packs[(name, split)] = (torch.cat(qs), torch.cat(sfs))
    state = {
        "packs": packs,
        "w13gs": weights["w13_gs"][lo:hi].float().contiguous(),
        "w2gs": weights["w2_gs"][lo:hi].float().contiguous(),
        "nw": weights["post_norm_w"].float() + 1.0,
        "inner": inner,
        "exact": not all(_moe_scales_simple(weights[k][lo:hi]) for k in ("w13_sf", "w2_sf")),
        # FP32 split-K accumulators, one row per (token, slot). They start at zero, and the last kernel of each
        # call zeroes every row it consumed, so they are zero again at the start of the next call.
        "h1": torch.zeros((_moe_ACC_MAX_M * _moe_TOP_K, 2 * inner), dtype=torch.float32, device=dev),
        "y": torch.zeros((_moe_ACC_MAX_M * _moe_TOP_K, hidden), dtype=torch.float32, device=dev),
        "none": torch.empty(0, dtype=torch.float32, device=dev),
    }
    _moe_module()
    torch.cuda.synchronize()
    return state


def _moe_kernel_fn(data: Inputs, state) -> torch.Tensor:
    x = data["hidden_states"]
    m, h = x.shape
    dev = x.device
    mt, rt, s1, s2 = _moe_config(m)
    w13q, w13sf = state["packs"][("w13", s1)]
    w2q, w2sf = state["packs"][("w2", s2)]
    if m <= _moe_ACC_MAX_M:
        h1, y = state["h1"][: m * _moe_TOP_K], state["y"][: m * _moe_TOP_K]
    else:
        h1 = torch.zeros((m * _moe_TOP_K, 2 * state["inner"]), dtype=torch.float32, device=dev)
        y = torch.zeros((m * _moe_TOP_K, h), dtype=torch.float32, device=dev)
    out = torch.empty((m, h), dtype=torch.bfloat16, device=dev)
    rglob = torch.empty(_moe_ROUTE_INTS + _moe_TOP_K * m, dtype=torch.int32, device=dev)  # gemm1 -> gemm2 routing
    # The decode layer may pass ("pre_norm") the pre-expert RMSNorm to fold into gemm1: (per-token sum of squares
    # of x, a buffer the norm kernel zeroes after use; FP32 norm weight; eps).
    pre_ss, pre_w, pre_eps = data.get("pre_norm") or (state["none"], state["none"], 0.0)
    _moe_module().moe_forward(x, data["topk_ids"], data["topk_weights"], w13q, w13sf, state["w13gs"], w2q, w2sf,
                          state["w2gs"], state["nw"], rglob, h1, y, out, mt, rt, s1, s2, LOCAL_START,
                          LOCAL_COUNT, _moe_POST_NORM_EPS, state["exact"], pre_ss, pre_w, pre_eps)
    return out

# ---- the layer (SGLang's, with the above swapped in) ----
class Weights(TypedDict):
    qkv_gate: torch.Tensor            # bfloat16 [2304, 8192]: q 16x64, k 2x64, v 2x64, output gate 16x64
    o: torch.Tensor                   # bfloat16 [8192, 1024]
    attn_residual_gate: torch.Tensor  # bfloat16 [8192], the post-attention residual gate's weight
    pre_ffn_norm: torch.Tensor        # bfloat16 [8192] (applied as 1 + w)
    router: torch.Tensor              # float32 [256, 8192]
    router_bias: torch.Tensor         # float32 [256], added to the scores only to choose experts
    pooler: torch.Tensor              # bfloat16 [4096, 8192]
    pre_expert_norm: torch.Tensor     # bfloat16 [4096] (applied as 1 + w)
    w13: torch.Tensor                 # uint8 [32, 8192, 2048]: this GPU's experts, gate then up rows, packed E2M1
    w13_sf: torch.Tensor              # float8_e4m3fn [32, 8192, 256]
    w13_gs: torch.Tensor              # float32 [32]
    w2: torch.Tensor                  # uint8 [32, 4096, 2048]
    w2_sf: torch.Tensor               # float8_e4m3fn [32, 4096, 256]
    w2_gs: torch.Tensor               # float32 [32]
    post_expert_norm: torch.Tensor    # bfloat16 [4096] (applied as 1 + w, before normalizing)
    unpooler: torch.Tensor            # bfloat16 [8192, 4096]; up to 8 tokens only rows 0-1023 (this GPU's slice)
    post_ffn_norm: torch.Tensor       # bfloat16 [8192] (applied as 1 + w, before normalizing)
    ffn_residual_gate: torch.Tensor   # bfloat16 [8192]
    next_pre_attn_norm: torch.Tensor  # bfloat16 [8192], the next layer's input norm (applied as 1 + w)


class Config(TypedDict):
    batch: int            # B sequences
    q_len: int            # L: 1 for decode, 8 for a verify block (causal inside it); 200-16384 on muse_prefill_layer
    tokens: int           # T = B * L, request-major
    hidden: int           # 8192
    num_q_heads: int      # 16 per GPU under TP8
    num_kv_heads: int     # 2
    head_dim: int         # 64
    page_size: int        # 1 in the scored cases (as SGLang serves Muse on H200), 64 in some tests
    window: int           # 2048 on sliding-window layers (a token and the 2,047 before it); 0 = all
    rope_theta: float     # 500000 on sliding-window layers (interleaved RoPE); 0 = no RoPE
    qk_scale_factor: float
    sm_scale: float       # 1/8
    kv_scale: float       # the cache scales' value: 1 in the scored cases (SGLang leaves them unset)
    n_experts: int        # 256
    top_k: int            # 8
    local_experts: int    # 32: this GPU computes experts local_start .. local_start + 31


class Inputs(TypedDict):
    stream: torch.Tensor      # float32 [T, 8192], the residual stream
    attn_input: torch.Tensor  # bfloat16 [T, 8192], the normalized stream (the previous layer's boundary wrote it)
    k_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, 2, 64], written in place
    v_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, 2, 64], written in place
    positions: torch.Tensor   # int64 [T], each new token's position
    cache_loc: torch.Tensor   # int64 [T], each new token's flat cache row
    cache_seqlens: torch.Tensor  # int32 [B], seq_lens + L
    cu_seqlens_q: torch.Tensor   # int32 [B+1], 0, L, 2L, ...
    scheduler_metadata: torch.Tensor | None  # int32, FA3's tile schedule on full-attention decode steps
    k_scale: torch.Tensor     # float32 [], key = e4m3 * k_scale
    v_scale: torch.Tensor     # float32 []
    page_table: torch.Tensor  # int32 [B, max_pages], physical page of each logical page (-1 unused)
    seq_lens: torch.Tensor    # int32 [B], cached tokens per request; the block starts at this position
    local_start: torch.Tensor # int32 [1], this GPU's first expert (a multiple of 32; its weights are w13[0], ...)
    config: Config


HIDDEN = 8192
RMS_EPS = 1e-5
POST_NORM_EPS = 1e-8
FP8 = torch.float8_e4m3fn
N_EXPERTS = 256
TOP_K = 8
LOCAL_EXPERTS = 32
D_EXPERT = 4096
# MuseSpark1xRouter switches to the split BF16 GEMMs above this many tokens.
ROUTER_FP32_MAX_TOKENS = 8
# Up to this many tokens the unpooler runs on this GPU's slice and is all-gathered.
SLICED_UNPOOL_MAX_TOKENS = 8
UNPOOL_SLICE = HIDDEN // 8
# MuseSpark1xFp4Experts' candidates, smallest first.
BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64)
ROPE_POSITIONS = 1 << 21
ROPE_THETA = 500_000.0


def _gemm(x, w):
    return _bf16_gemm_dispatch_impl(x, w, None)


def _param(w):
    return nn.Parameter(w.contiguous(), requires_grad=False)


def _norm(cls, w, size, eps, **kw):
    """A Muse norm module holding w (applied as 1 + w), as the checkpoint loads it."""
    norm = cls(size, eps=eps, has_weight=w is not None, zero_centered_gamma=True, **kw)
    if w is not None:
        norm.weight = _param(w)
        norm.precompute_effective_weight()
    return norm


def _residual(w):
    gate = MuseSpark1xNormPreservingResidualConnection(HIDDEN, temperature=0.3)
    gate.gate = _param(w)
    gate.precompute_coefficients()
    return gate


def _marlin_bank(weights):
    """The local experts in Marlin's layout, via SGLang's own loader step."""
    device = weights["w13"].device
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"].contiguous(),
        w2_weight=weights["w2"].contiguous(),
        w13_weight_scale=weights["w13_sf"].contiguous(),
        w2_weight_scale=weights["w2_sf"].contiguous(),
        w13_weight_scale_2=weights["w13_gs"].contiguous(),
        w2_weight_scale_2=weights["w2_gs"].contiguous(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    expert_map = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
    expert_map[:LOCAL_EXPERTS] = torch.arange(LOCAL_EXPERTS, dtype=torch.int32, device=device)
    return {
        "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
        "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
        "expert_map": expert_map, "size_n": layer.intermediate_size_per_partition,
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
    }


def prepare(weights):
    device = weights["qkv_gate"].device
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    # [2^21, 64] float32 cos | sin, as SGLang's RotaryEmbedding builds it.
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, 64, 2, dtype=torch.float, device=device) / 64))
    freqs = torch.einsum("i,j -> ij", torch.arange(ROPE_POSITIONS, dtype=torch.float, device=device), inv_freq)
    pre_ffn = _norm(MuseSpark1xRMSNorm, weights["pre_ffn_norm"], HIDDEN, RMS_EPS)
    next_pre_attn = _norm(MuseSpark1xRMSNorm, weights["next_pre_attn_norm"], HIDDEN, RMS_EPS)
    post_attn = _norm(MuseSpark1xRMSNorm, None, HIDDEN, POST_NORM_EPS, is_post_norm=True,
                      post_norm_gain_center_type="one", residual_id=1)
    post_ffn = _norm(MuseSpark1xRMSNormWithInputScale, weights["post_ffn_norm"], HIDDEN, POST_NORM_EPS,
                     is_post_norm=True, post_norm_gain_center_type="one", residual_id=2)
    post_expert = _norm(MuseSpark1xRMSNormWithInputScale, weights["post_expert_norm"], D_EXPERT, POST_NORM_EPS,
                        is_post_norm=True, post_norm_gain_center_type="one", residual_id=2)
    state = {
        "dense": {name: weights[name].contiguous() for name in ("qkv_gate", "o", "pooler", "unpooler")},
        "router": router.contiguous(),
        "router_hi_t": hi.t().contiguous(),
        "router_lo_t": lo.t().contiguous(),
        "routing": FusedMuseSparkRouting(expert_bias=weights["router_bias"].float().contiguous(), route_scale=1.0,
                                         router_output_multiplier=0.1767766952966369,
                                         router_score_division_eps=1e-15),
        "cos_sin": torch.cat((freqs.cos(), freqs.sin()), dim=-1),
        "qk_norm": FusedQKNorm(head_dim=64, num_q_heads=16, num_kv_heads=2, eps=RMS_EPS),
        "o_norm_gate": FusedRMSNormGate(MuseSpark1xRMSNorm(64, eps=RMS_EPS, has_weight=False), True, 64),
        "attn_boundary": FusedPostNormResidualPreNorm(post_attn, _residual(weights["attn_residual_gate"]), pre_ffn),
        "ffn_boundary": FusedPostNormResidualPreNorm(post_ffn, _residual(weights["ffn_residual_gate"]), next_pre_attn),
        "pre_expert_norm": _norm(MuseSpark1xRMSNorm, weights["pre_expert_norm"], D_EXPERT, RMS_EPS),
        # MuseSpark1xFp4Experts binds the post-expert norm's effective (1 + w) weight.
        "post_expert_w": post_expert._get_effective_weight().detach().to(torch.bfloat16),
        "side_stream": torch.cuda.Stream(device),
        "_modules": (pre_ffn, next_pre_attn, post_attn, post_ffn, post_expert),
    }
    del freqs
    state["proj"] = _proj_prepare({name: weights[name] for name in ("qkv_gate", "o", "pooler", "unpooler", "router")})
    state["moe"] = _moe_prepare({"w13": weights["w13"], "w13_sf": weights["w13_sf"], "w13_gs": weights["w13_gs"],
                                 "w2": weights["w2"], "w2_sf": weights["w2_sf"], "w2_gs": weights["w2_gs"],
                                 "post_norm_w": weights["post_expert_norm"]})
    # The pre-expert RMSNorm folded into the pooler and gemm1 (up to SKINNY_MAX_M tokens): per-token sums of squares
    # (zero between calls: the MoE's norm kernel re-zeroes them) and the FP32 (1 + w).
    state["pre_ss"] = torch.zeros(256, dtype=torch.float32, device=device)
    state["pre_w"] = (1.0 + weights["pre_expert_norm"].float()).contiguous()
    torch.cuda.synchronize()
    return state


def _experts(x, topk_weights, topk_ids, state):
    """MuseSpark1xFp4Experts._marlin_expert_outputs and its post-expert-norm finalize."""
    bank = state["bank"]
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    local_ids = bank["expert_map"][topk_ids].contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()
    for block_size_m in BLOCK_SIZE_M_CANDIDATES:
        if num_tokens * TOP_K / LOCAL_EXPERTS / block_size_m < 0.9:
            break
    sorted_ids, block_experts, n_padded = moe_align_block_size(local_ids, block_size_m, N_EXPERTS)
    cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_tokens * TOP_K * size_k].view(-1, size_k)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=True,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False,
                  use_fp32_reduce=True, is_zp_float=False)
    ws = bank["workspace"]
    cache1 = moe_wna16_marlin_gemm(
        x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=TOP_K,
        size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    y = moe_wna16_marlin_gemm(
        cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=1,
        size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n, **common).view(num_tokens, TOP_K, size_k)
    return fused_post_expert_norm_weighted_sum(
        y, topk_weights, norm_weight=state["post_expert_w"], eps=POST_NORM_EPS, weight_first=True,
        slot_expert_ids=local_ids)


def kernel_fn(data, state):
    cfg = data["config"]
    batch, length, tokens = cfg["batch"], cfg["q_len"], cfg["tokens"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    dense = state["dense"]
    # Attention: MuseSpark1xAttention.forward.
    qkv = _proj_kernel_fn({"x": data["attn_input"], "config": {"m": data["attn_input"].shape[0], "n": dense["qkv_gate"].shape[0], "k": data["attn_input"].shape[1], "layer": "qkv_gate"}}, state["proj"])
    # QK-norm, RoPE, the cache store, attention and the gated output norm, as submitted to muse_attention_decode.
    o = _attn_kernel_fn({"qkv": qkv.view(batch, length, -1), "positions": data["positions"],
                         "cache_loc": data["cache_loc"], "cache_seqlens": data["cache_seqlens"],
                         "cu_seqlens_q": data["cu_seqlens_q"], "scheduler_metadata": data["scheduler_metadata"],
                         "k_cache": data["k_cache"], "v_cache": data["v_cache"], "k_scale": data["k_scale"],
                         "v_scale": data["v_scale"], "page_table": data["page_table"], "seq_lens": data["seq_lens"],
                         "config": cfg}).view(tokens, hq * d)
    attn_output = _proj_kernel_fn({"x": o, "config": {"m": o.shape[0], "n": dense["o"].shape[0], "k": o.shape[1], "layer": "o"}}, state["proj"])  # all-reduce: identity

    # Post-attention boundary, router and MoE block: MuseSpark1xMoE._forward_experts.
    split_router = tokens > ROUTER_FP32_MAX_TOKENS
    ffn_input, stream, router_input = state["attn_boundary"](
        data["stream"], attn_output, branch_dtype=torch.bfloat16, router_dtype=None if split_router else torch.float32)
    # The router and routing run on a side stream, overlapping the pooler on the main one.
    main_stream, side_stream = torch.cuda.current_stream(), state["side_stream"]
    side_stream.wait_stream(main_stream)
    # Where the pooler is the skinny kernel, it also sums each token's squares and gemm1 applies the pre-expert
    # RMSNorm as it stages x (one kernel fewer); elsewhere the norm runs on its own, as SGLang does.
    fold_norm = _proj_config("pooler", tokens) is not None
    expert_input = _proj_kernel_fn({"x": ffn_input, "config": {"m": ffn_input.shape[0], "n": dense["pooler"].shape[0],
                                    "k": ffn_input.shape[1], "layer": "pooler"},
                                    **({"ssq": state["pre_ss"]} if fold_norm else {})}, state["proj"])
    with torch.cuda.stream(side_stream):
        if split_router:
            logits = _proj_kernel_fn({"x": ffn_input, "config": {"m": ffn_input.shape[0], "n": 256, "k": ffn_input.shape[1], "layer": "router"}}, state["proj"])
        else:
            logits = F.linear(router_input, state["router"])
        topk_weights, topk_ids = state["routing"](None, logits, TOP_K, True)
    main_stream.wait_stream(side_stream)
    if not torch.cuda.is_current_stream_capturing():
        for tensor in (topk_weights, topk_ids):
            tensor.record_stream(main_stream)
    # This GPU's experts start at local_start: as SGLang's expert_map on that EP rank.
    local_ids = (topk_ids - data["local_start"]) % N_EXPERTS
    # all-reduce: identity
    expert_output = _moe_kernel_fn({
        "hidden_states": expert_input if fold_norm else state["pre_expert_norm"](expert_input).contiguous(),
        "pre_norm": (state["pre_ss"], state["pre_w"], RMS_EPS) if fold_norm else None,
        "topk_ids": local_ids.to(torch.int32).contiguous(),
        "topk_weights": topk_weights.to(torch.float32).contiguous(),
        "config": {"bs": tokens, "d_hidden": D_EXPERT, "d_expert": D_EXPERT, "n_experts": N_EXPERTS,
                   "local_start": 0, "local_count": LOCAL_EXPERTS, "top_k": TOP_K, "sharding": "ep8"}}, state["moe"])
    if tokens <= SLICED_UNPOOL_MAX_TOKENS:
        unpooled = _proj_kernel_fn({"x": expert_output, "config": {"m": expert_output.shape[0], "n": UNPOOL_SLICE, "k": expert_output.shape[1], "layer": "unpooler"}}, state["proj"])
        ffn_output = F.pad(unpooled, (0, HIDDEN - UNPOOL_SLICE))  # all-gather: identity
    else:
        ffn_output = _proj_kernel_fn({"x": expert_output, "config": {"m": expert_output.shape[0], "n": dense["unpooler"].shape[0], "k": expert_output.shape[1], "layer": "unpooler"}}, state["proj"])

    # Post-FFN boundary, with the next layer's pre-attention norm.
    next_attn_input, stream, _ = state["ffn_boundary"](stream, ffn_output, branch_dtype=torch.bfloat16)
    return next_attn_input, stream, topk_ids
