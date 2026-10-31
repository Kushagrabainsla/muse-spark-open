# okc: problem=muse_attention_decode
"""Gated attention decode: a fused Triton prep, a CUDA split-KV attention over the paged FP8 cache, and a Triton
combine with the gated output norm.

  prep     one program per token: per-head RMSNorm of q and k, interleaved RoPE (pairs (2i, 2i + 1)) on sliding
           layers, k and v divided by the cache scales, rounded to E4M3 and stored at the token's cache row, and
           the query, times the softmax scale, the query factor and log2(e), kept in FP16 (finer than the E4M3 the
           rules allow).
  attn     CUDA, one CTA per (request, pair of KV heads, key chunk). Keys come in stages of 32 through a cp.async
           ring: per key one page-table entry and a 128-byte read of both heads' keys, and of their values (keys
           past a request's length are zero-filled). S = Q K^T and P V are m16n8k16 FP16 MMAs with FP32
           accumulation; E4M3 keys and values become FP16 fragments with cvt.f16x2.e4m3x2 (exact). Head
           dimensions and output columns are permuted per lane so every shared-memory read is contiguous; the
           tiles are swizzled against bank conflicts. Online softmax in base 2 with the causal and window masks:
           every visible key is attended. Partial (max, sum, output) per query row go to a workspace.
  combine  one program per query row: merges the chunks, then RMSNorm over the head times sigmoid(gate).
"""

import math
from typing import TypedDict

import torch
import triton
import triton.language as tl
from sgl_kernel.flash_attn import flash_attn_with_kvcache
from torch.utils.cpp_extension import load_inline

from sglang.kernels.ops.muse_spark_v12.norm_gate import rms_norm_gate


class Config(TypedDict):
    batch: int            # B
    q_len: int            # L: 1 for decode, 8 for a speculative verify block (causal inside it)
    num_q_heads: int      # Hq: 16 per GPU under TP8
    num_kv_heads: int     # Hkv: 2; each serves Hq / Hkv query heads
    head_dim: int         # 64
    page_size: int        # 1 in the scored cases (as SGLang serves Muse on H200), 64 in some tests
    window: int           # 2048 on sliding-window layers (a token and the 2,047 before it); 0 = all
    rope_theta: float     # 500000 on sliding-window layers; 0 = no RoPE
    qk_scale_factor: float
    sm_scale: float       # 1/8
    kv_scale: float       # the cache scales' value: 1 in the scored cases (SGLang leaves them unset)


class Inputs(TypedDict):
    qkv: torch.Tensor          # bfloat16 [B, L, (2*Hq + 2*Hkv)*64], 2304 under tp8: q (Hq*64), keys (Hkv*64),
                               # values (Hkv*64), the output gate's logits (Hq*64)
    positions: torch.Tensor    # int64 [B*L], each new token's position
    cache_loc: torch.Tensor    # int64 [B*L], each new token's flat cache row
    cache_seqlens: torch.Tensor  # int32 [B], seq_lens + L
    cu_seqlens_q: torch.Tensor   # int32 [B+1], 0, L, 2L, ...
    scheduler_metadata: torch.Tensor | None  # int32, FA3's tile schedule on full-attention decode steps
    k_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, Hkv, 64], written in place
    v_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, Hkv, 64], written in place
    k_scale: torch.Tensor     # float32 [], key = e4m3 * k_scale
    v_scale: torch.Tensor     # float32 [], value = e4m3 * v_scale
    page_table: torch.Tensor  # int32 [B, max_pages], physical page of each logical page (-1 unused)
    seq_lens: torch.Tensor    # int32 [B], cached tokens per request; the block starts at this position
    config: Config


RMS_EPS = 1e-5
FP8 = torch.float8_e4m3fn
LOG2E = 1.4426950408889634
# Where the CUDA kernel beats FA3 (sweeps on an H200), and its keys per CTA there: decode steps (q_len 1) of
# full layers with many requests, and of sliding layers with few. Everything else runs SGLang's FA3 path with the
# fused prep.
CUDA_FULL_MIN_BATCH = 24
CUDA_SLIDING_MAX_BATCH = 4
CHUNK = {"sliding": 32, "full": 2048}
STAGES = 4


@triton.jit
def _rms_rope(x, pos, theta_log2, HAS_ROPE: tl.constexpr, R: tl.constexpr, D: tl.constexpr):
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
def _prep(qkv_ptr, pos_ptr, loc_ptr, ks_ptr, vs_ptr, q16_ptr, kc_ptr, vc_ptr, theta_log2, q_mult,
          HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, QKV_W: tl.constexpr, HAS_ROPE: tl.constexpr):
    t = tl.program_id(0)
    pos = tl.load(pos_ptr + t)
    dn = tl.arange(0, D)
    hq = tl.arange(0, HQ)
    q = tl.load(qkv_ptr + t * QKV_W + hq[:, None] * D + dn[None, :]).to(tl.float32)
    q = _rms_rope(q, pos, theta_log2, HAS_ROPE, HQ, D) * q_mult
    tl.store(q16_ptr + t * HQ * D + hq[:, None] * D + dn[None, :], q.to(tl.float16))
    hk = tl.arange(0, HKV)
    k = tl.load(qkv_ptr + t * QKV_W + (HQ + hk)[:, None] * D + dn[None, :]).to(tl.float32)
    k = _rms_rope(k, pos, theta_log2, HAS_ROPE, HKV, D)
    v = tl.load(qkv_ptr + t * QKV_W + (HQ + HKV + hk)[:, None] * D + dn[None, :]).to(tl.float32)
    row = tl.load(loc_ptr + t) * HKV
    tl.store(kc_ptr + (row + hk)[:, None] * D + dn[None, :], (k / tl.load(ks_ptr)).to(tl.float8e4nv))
    tl.store(vc_ptr + (row + hk)[:, None] * D + dn[None, :], (v / tl.load(vs_ptr)).to(tl.float8e4nv))


@triton.jit
def _attn_combine(qkv_ptr, out_ptr, ws_o, ws_m, ws_l, NCH,
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
def _qkv_prep(qkv_ptr, pos_ptr, loc_ptr, ks_ptr, vs_ptr, q8_ptr, kc_ptr, vc_ptr, theta_log2,
              HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, QKV_W: tl.constexpr, HAS_ROPE: tl.constexpr):
    t = tl.program_id(0)
    pos = tl.load(pos_ptr + t)
    dn = tl.arange(0, D)
    hq = tl.arange(0, HQ)
    q = tl.load(qkv_ptr + t * QKV_W + hq[:, None] * D + dn[None, :]).to(tl.float32)
    q = _rms_rope(q, pos, theta_log2, HAS_ROPE, HQ, D)
    tl.store(q8_ptr + t * HQ * D + hq[:, None] * D + dn[None, :], q.to(tl.float8e4nv))
    hk = tl.arange(0, HKV)
    k = tl.load(qkv_ptr + t * QKV_W + (HQ + hk)[:, None] * D + dn[None, :]).to(tl.float32)
    k = _rms_rope(k, pos, theta_log2, HAS_ROPE, HKV, D)
    v = tl.load(qkv_ptr + t * QKV_W + (HQ + HKV + hk)[:, None] * D + dn[None, :]).to(tl.float32)
    row = tl.load(loc_ptr + t) * HKV
    tl.store(kc_ptr + (row + hk)[:, None] * D + dn[None, :], (k / tl.load(ks_ptr)).to(tl.float8e4nv))
    tl.store(vc_ptr + (row + hk)[:, None] * D + dn[None, :], (v / tl.load(vs_ptr)).to(tl.float8e4nv))


def _kernel_fa3(data: Inputs) -> torch.Tensor:
    """SGLang's FA3 path with the fused prep (the A3 submission)."""
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    tokens = batch * length
    qkv = data["qkv"].view(tokens, -1)
    theta = cfg["rope_theta"]
    q8 = torch.empty((tokens, hq, d), dtype=FP8, device=qkv.device)
    k_cache, v_cache = data["k_cache"], data["v_cache"]
    _qkv_prep[(tokens,)](qkv, data["positions"], data["cache_loc"], data["k_scale"], data["v_scale"], q8, k_cache,
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
    o = rms_norm_gate(out.reshape(tokens, hq * d).to(torch.bfloat16), gate, None, head_dim=d, eps=RMS_EPS,
                      has_norm=True)
    return o.view(batch, length, hq * d)


CUDA_SRC = r"""
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

// Stage tile: 32 keys x 128 B of keys (both heads), then 32 x 128 B of values. Keys: 16-byte chunk c of key row k
// at chunk c ^ ((k & 1) << 2). Values: chunk c at c ^ (((k >> 1) & 3) << 1).
template <int L, int STG>
__global__ void __launch_bounds__(L == 1 ? 128 : 256) attn_kernel(
    const __half* __restrict__ q16, const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,
    const int* __restrict__ pt, const int* __restrict__ seqlens, float* __restrict__ ws_o, float* __restrict__ ws_m,
    float* __restrict__ ws_l, int HKV, int max_pages, int page, int window, int CH, int NCH) {
  constexpr int G = 8, D = 64, RH = G * L < 16 ? 16 : G * L, MT = RH / 16;
  constexpr int NT = L == 1 ? 128 : 256, NWARP = NT / 32;
  constexpr int KS = NWARP / (2 * MT);  // warps splitting each stage's keys: 2 (L = 1) or 1 (L = 8)
  constexpr int KPW = 32 / KS;          // keys per warp per stage
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

  const int nsteps = (ce - cs + 31) / 32;
  // Each thread copies 512 / NT 16-byte pieces per stage; their cache rows (from the page table) are loaded one
  // stage ahead of the copies, so the copies never wait on the page table.
  constexpr int PER = 512 / NT;
  int prow[PER];
  auto rows_of = [&](int kb) {
#pragma unroll
    for (int i = 0; i < PER; ++i) {
      const int pos = kb + ((threadIdx.x + i * NT) >> 4);
      prow[i] = pos < ce ? pt[(size_t)b * max_pages + pos / page] * page + pos % page : -1;
    }
  };
  auto issue = [&](int stage) {
    uint8_t* tile = smem + stage * 8192;
#pragma unroll
    for (int i = 0; i < PER; ++i) {
      const int idx = threadIdx.x + i * NT;
      const int key = idx >> 4, part = (idx >> 3) & 1, ch = idx & 7;
      const bool ok = prow[i] >= 0;
      const uint8_t* src = (part ? vc : kc) + ((long long)(ok ? prow[i] : 0) * HKV + h0) * D + ch * 16;
      const int sw = part ? (ch ^ (((key >> 1) & 3) << 1)) : (ch ^ ((key & 1) << 2));
      cp16(tile + part * 4096 + key * 128 + sw * 16, src, ok);
    }
  };
  rows_of(cs);
#pragma unroll
  for (int s = 0; s < STG - 1; ++s) {
    if (s < nsteps) {
      issue(s);
      rows_of(cs + 32 * (s + 1));
    }
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
      if (nx < nsteps) {
        issue(nx % STG);
        rows_of(cs + 32 * (nx + 1));
      }
      cp_commit();
    }
    const uint8_t* Kt = smem + (it % STG) * 8192;
    const uint8_t* Vt = Kt + 4096;
    const int kb = cs + 32 * it + ks * KPW;  // this warp's first key

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
    // Masks: element (row, key kb + 8t + 2q + e).
    float mxa = -INFINITY, mxb = -INFINITY;
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
        mxa = fmaxf(mxa, sc[t][e]);
        mxb = fmaxf(mxb, sc[t][2 + e]);
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
          int64_t window, int64_t CH, int64_t NCH, int64_t batch) {
  constexpr int STG = 4;
  const size_t smem = STG * 8192;
  dim3 grid(batch * (HKV / 2), NCH);
  auto stream = at::cuda::getCurrentCUDAStream();
#define LAUNCH(LL)                                                                                             \
  {                                                                                                           \
    auto k = attn_kernel<LL, STG>;                                                                             \
    static bool set = false;                                                                                   \
    if (!set) {                                                                                                \
      cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);                        \
      set = true;                                                                                              \
    }                                                                                                          \
    k<<<grid, LL == 1 ? 128 : 256, smem, stream>>>(                                                            \
        reinterpret_cast<const __half*>(q16.data_ptr()), reinterpret_cast<const uint8_t*>(kc.data_ptr()),      \
        reinterpret_cast<const uint8_t*>(vc.data_ptr()), pt.data_ptr<int>(), seqlens.data_ptr<int>(),           \
        ws_o.data_ptr<float>(), ws_m.data_ptr<float>(), ws_l.data_ptr<float>(), (int)HKV, (int)pt.size(1),     \
        (int)page, (int)window, (int)CH, (int)NCH);                                                            \
  }
  if (L == 1) LAUNCH(1) else if (L == 8) LAUNCH(8) else TORCH_CHECK(false, "attn: q_len must be 1 or 8");
}
"""

CPP_SRC = ("void attn(torch::Tensor q16, torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor seqlens, "
           "torch::Tensor ws_o, torch::Tensor ws_m, torch::Tensor ws_l, int64_t L, int64_t HKV, int64_t page, "
           "int64_t window, int64_t CH, int64_t NCH, int64_t batch);")
_mod = None


def _module():
    global _mod
    if _mod is None:
        _mod = load_inline("muse_attn_decode", CPP_SRC, cuda_sources=CUDA_SRC, functions=["attn"],
                           extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
    return _mod


def kernel_fn(data: Inputs) -> torch.Tensor:
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    group = hq // hkv
    window = cfg["window"]
    use_cuda = length == 1 and group == 8 and hkv % 2 == 0 and d == 64 and (
        batch <= CUDA_SLIDING_MAX_BATCH if window else batch >= CUDA_FULL_MIN_BATCH)
    if not use_cuda:
        return _kernel_fa3(data)
    return kernel_fn_cuda(data)


def kernel_fn_cuda(data: Inputs) -> torch.Tensor:
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
    q16 = torch.empty((tokens, hq, d), dtype=torch.float16, device=dev)
    _prep[(tokens,)](qkv, data["positions"], data["cache_loc"], data["k_scale"], data["v_scale"], q16, k_cache, v_cache,
                     math.log2(theta) if theta else 0.0, cfg["qk_scale_factor"] * cfg["sm_scale"] * LOG2E,
                     HQ=hq, HKV=hkv, D=d, QKV_W=qkv.shape[1], HAS_ROPE=bool(theta), num_warps=4)
    page = cfg["page_size"]
    pt = data["page_table"]
    ch = CHUNK["sliding" if window else "full"]
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
    _module().attn(q16, k_cache.view(torch.uint8), v_cache.view(torch.uint8), pt, data["seq_lens"], ws_o, ws_m, ws_l,
                   length, hkv, page, window, ch, nch, batch)
    if cfg["kv_scale"] != 1.0:
        ws_o.mul_(data["v_scale"])
    out = torch.empty((batch, length, hq * d), dtype=torch.bfloat16, device=dev)
    _attn_combine[(groups * rows,)](qkv, out, ws_o, ws_m, ws_l, nch, HKV=hkv, KVPC=2, G=group, L=length, D=d,
                                    RH=rh, QKV_W=qkv.shape[1], HQ=hq, CT=64, num_warps=2)
    return out
