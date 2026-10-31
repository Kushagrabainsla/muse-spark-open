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
# The CUDA kernels run decode steps (q_len 1), where they beat FA3 at every batch, and sliding verify blocks of at
# most CUDA_VERIFY_MAX_BATCH requests (multi-draw sweeps on an H200); other verify blocks run SGLang's FA3 path with
# the fused prep (the wgmma kernel is behind FA3 on full layers: its serial per-stage chain and the 64 B-row gather
# rate, see EXPERIMENTS.md A9). CHUNK overrides the keys per CTA (None: by shape).
CUDA_VERIFY_MAX_BATCH = 16
CHUNK = {"sliding": None, "full": None}
STAGE_KEYS = {"sliding": 64, "full": 64}
MMA8 = {1: False, 8: True}  # per q_len: E4M3 MMAs (attn8_kernel), else FP16 MMAs on converted fragments
WGMMA = True  # verify blocks (q_len 8, 8 query heads per KV head): the wgmma kernel (attnwg_kernel) instead of attn8
VERIFY_HEADS = {"sliding": 1, "full": 1}  # KV heads per CTA of the wgmma kernel


def _chunk(window: int, batch: int, length: int) -> int:
    """Keys per CTA: sliding layers aim at ~128 CTAs (128 keys at least), full layers grow chunks with the batch;
    verify blocks (one CTA per KV head) aim at ~300 CTAs on sliding layers and keep 4 CTAs per SM on full ones."""
    if CHUNK["sliding" if window else "full"]:
        return CHUNK["sliding" if window else "full"]
    if length > 1:
        if window:
            return min(512, max(64, triton.next_power_of_2(triton.cdiv(window * batch * 2, 300))))
        return 512 if batch <= 2 else 1024
    if window:
        return min(2048, max(128, triton.next_power_of_2(triton.cdiv(window * batch, 128))))
    return 512 if batch <= 2 else 1024 if batch <= 8 else 2048
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


def _wgmma_fn(n: int) -> str:
    """CUDA source of wgmma_<n>: one wgmma.mma_async m64n<n>k32 f32 += e4m3 (A in registers) x e4m3 (B by descriptor).
    The A registers are in-out operands, and the caller pins them again after wgmma.wait_group: the instruction reads
    them asynchronously, so the compiler must not reuse them in between."""
    nreg = n // 2
    d = ",".join(f"%{i}" for i in range(nreg))
    a = ",".join(f"%{nreg + i}" for i in range(4))
    outs = ", ".join(f'"+f"(d[{i}])' for i in range(nreg)) + ', "+r"(a[0]), "+r"(a[1]), "+r"(a[2]), "+r"(a[3])'
    return f"""
__device__ __forceinline__ void wgmma_{n}(float* d, uint32_t* a, uint64_t bdesc, int scale_d) {{
  asm volatile("{{ .reg .pred p; setp.ne.b32 p, %{nreg + 4}, 0; "
               "wgmma.mma_async.sync.aligned.m64n{n}k32.f32.e4m3.e4m3 {{{d}}}, {{{a}}}, %{nreg + 5}, p, 1, 1; }}"
               : {outs}
               : "r"(scale_d), "l"(bdesc));
}}"""


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
// in as they are, and the probabilities are rounded to E4M3 for P V (as FA3's FP8 kernel does). Lane q, k-step s
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
CUDA_SRC += _wgmma_fn(64) + _wgmma_fn(128) + r"""
// Verify blocks: one CTA per (request, HEADS KV heads, key chunk), one warpgroup per KV head, whose 64 query rows
// (8 tokens x 8 heads; row r = 8 j + g) are one m64 tile. Per stage of SK keys: S = Q K^T as m64nSKk32 E4M3 wgmmas
// (A = Q in registers; B = the stage's key tile: rows of 64 HEADS bytes, swizzled by their width, head hh at
// + 64 hh), softmax in registers, the values transposed once by all threads into VT (per head: 64 dim rows of SK
// bytes, swizzled by 64 B (128 B at SK = 128), keys in the P-fragment slot order: slot 32u + 16h + 4q + e <-> key
// 32u + 16h + 2q + (e & 1) + 8 (e >> 1)), then O += P V as m64n64k32 wgmmas with P (E4M3) from registers.
// HEADS = 2 reads whole 128 B cache rows (the faster gather) with two CTAs per SM; HEADS = 1 reads 64 B rows (the
// other head's CTA reads the same lines from L2) with four CTAs per SM, which suits the short sliding windows.
template <int STG, int SK, int HEADS>
__global__ void __launch_bounds__(128 * HEADS, HEADS == 1 ? 4 : 2) attnwg_kernel(
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
    for (int blk = threadIdx.x; blk < SK * 4 * HEADS; blk += NT) {
      const int dg = blk & 15, qq = (blk >> 4) & 3, hf = (blk >> 6) & 1, u = (blk >> 7) % KSP, hd = (blk >> 7) / KSP;
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
  float sc[NS];

  for (int it = 0; it < nsteps; ++it) {
    // One barrier per stage: after it, stage it + 1 has landed (for its transpose), VT of stage it is visible, and
    // the ring slot of stage it + STG - 1 (last read by stage it - 1) is free.
    cp_wait<STG - 3>();
    fence_async();
    __syncthreads();
    {
      const int nx = it + STG - 1;
      if (nx < nsteps) issue(nx % STG, cs + SK * nx);
      cp_commit();
    }
    if (it + 1 < nsteps) transpose(it + 1);
    const uint8_t* Kt = smem + (it % STG) * TILE + 64 * hl;
    const uint8_t* vt = VT + ((it & 1) * HEADS + hl) * (64 * SK);
    const int kbs = cs + SK * it;

    // The Q fragments are reloaded every stage (L1 hits): kept in registers across the loop, the compiler reuses
    // their registers while the asynchronous wgmma still reads them, whatever the operand constraints.
    uint32_t qa[2][4];
#pragma unroll
    for (int s = 0; s < 2; ++s) {
      qa[s][0] = __ldg(reinterpret_cast<const uint32_t*>(qa_p + 32 * s));
      qa[s][1] = __ldg(reinterpret_cast<const uint32_t*>(qb_p + 32 * s));
      qa[s][2] = __ldg(reinterpret_cast<const uint32_t*>(qa_p + 32 * s + 16));
      qa[s][3] = __ldg(reinterpret_cast<const uint32_t*>(qb_p + 32 * s + 16));
    }
    reg_fence<NS>(sc);
    wg_fence();
#pragma unroll
    for (int s = 0; s < 2; ++s) {
      if constexpr (SK == 64) wgmma_64(sc, qa[s], make_desc(Kt + 32 * s, KLAYOUT, KSBO), s);
      else wgmma_128(sc, qa[s], make_desc(Kt + 32 * s, KLAYOUT, KSBO), s);
    }
    wg_commit();
    wg_wait<0>();
    reg_fence<NS>(sc);
    reg_fence<4>(qa[0]);
    reg_fence<4>(qa[1]);
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
    uint32_t pa[KSP][4];
#pragma unroll
    for (int u = 0; u < KSP; ++u) {
      const float* p4 = sc + 16 * u;
      pa[u][0] = pack_e4m3x4(p4[0], p4[1], p4[4], p4[5]);
      pa[u][1] = pack_e4m3x4(p4[2], p4[3], p4[6], p4[7]);
      pa[u][2] = pack_e4m3x4(p4[8], p4[9], p4[12], p4[13]);
      pa[u][3] = pack_e4m3x4(p4[10], p4[11], p4[14], p4[15]);
    }
    reg_fence<32>(acc);
    wg_fence();
#pragma unroll
    for (int u = 0; u < KSP; ++u) wgmma_64(acc, pa[u], make_desc(vt + 32 * u, VLAYOUT, VSBO), 1);
    wg_commit();
    wg_wait<0>();
    reg_fence<32>(acc);
#pragma unroll
    for (int u = 0; u < KSP; ++u) reg_fence<4>(pa[u]);
  }
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

"""
CPP_SRC = ("void attn(torch::Tensor q16, torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor seqlens, "
           "torch::Tensor ws_o, torch::Tensor ws_m, torch::Tensor ws_l, int64_t L, int64_t HKV, int64_t page, "
           "int64_t window, int64_t CH, int64_t NCH, int64_t batch, int64_t SK);\n"
           "void attn8(torch::Tensor q8, torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor seqlens, "
           "torch::Tensor kscale, torch::Tensor vscale, torch::Tensor ws_o, torch::Tensor ws_m, torch::Tensor ws_l, "
           "double qscale, int64_t L, int64_t HKV, int64_t page, int64_t window, int64_t CH, int64_t NCH, "
           "int64_t batch);\n"
           "void attnwg(torch::Tensor q8, torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor seqlens, "
           "torch::Tensor kscale, torch::Tensor vscale, torch::Tensor ws_o, torch::Tensor ws_m, torch::Tensor ws_l, "
           "double qscale, int64_t HKV, int64_t page, int64_t window, int64_t CH, int64_t NCH, int64_t batch, "
           "int64_t SK, int64_t HEADS);")
_mod = None


def _module():
    global _mod
    if _mod is None:
        _mod = load_inline("muse_attn_decode", CPP_SRC, cuda_sources=CUDA_SRC, functions=["attn", "attn8", "attnwg"],
                           extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
    return _mod


def kernel_fn(data: Inputs) -> torch.Tensor:
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    group = hq // hkv
    use_cuda = group == 8 and hkv % 2 == 0 and d == 64 and (
        length == 1 or (length == 8 and cfg["window"] > 0 and batch <= CUDA_VERIFY_MAX_BATCH))
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
    if MMA8[length]:
        return _kernel_cuda8(data)
    q16 = torch.empty((tokens, hq, d), dtype=torch.float16, device=dev)
    _prep[(tokens,)](qkv, data["positions"], data["cache_loc"], data["k_scale"], data["v_scale"], q16, k_cache, v_cache,
                     math.log2(theta) if theta else 0.0, cfg["qk_scale_factor"] * cfg["sm_scale"] * LOG2E,
                     HQ=hq, HKV=hkv, D=d, QKV_W=qkv.shape[1], HAS_ROPE=bool(theta), num_warps=4)
    page = cfg["page_size"]
    pt = data["page_table"]
    ch = _chunk(window, batch, length)
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
                   length, hkv, page, window, ch, nch, batch, STAGE_KEYS["sliding" if window else "full"])
    if cfg["kv_scale"] != 1.0:
        ws_o.mul_(data["v_scale"])
    out = torch.empty((batch, length, hq * d), dtype=torch.bfloat16, device=dev)
    _attn_combine[(groups * rows,)](qkv, out, ws_o, ws_m, ws_l, nch, HKV=hkv, KVPC=2, G=group, L=length, D=d,
                                    RH=rh, QKV_W=qkv.shape[1], HQ=hq, CT=64, num_warps=2)
    return out


def _kernel_cuda8(data: Inputs) -> torch.Tensor:
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
    q8 = torch.empty((tokens, hq, d), dtype=FP8, device=dev)
    _qkv_prep[(tokens,)](qkv, data["positions"], data["cache_loc"], data["k_scale"], data["v_scale"], q8, k_cache,
                         v_cache, math.log2(theta) if theta else 0.0, HQ=hq, HKV=hkv, D=d, QKV_W=qkv.shape[1],
                         HAS_ROPE=bool(theta), num_warps=4)
    pt = data["page_table"]
    page = cfg["page_size"]
    ch = _chunk(window, batch, length)
    span = (window - 1 + length) if window else pt.shape[1] * page
    nch = triton.cdiv(span, ch)
    rh = max(16, group * length)
    rows = 2 * rh
    groups = batch * (hkv // 2)
    ws_o = torch.empty((groups * nch * rows, d), dtype=torch.float32, device=dev)
    ws_m = torch.empty((groups * nch * rows,), dtype=torch.float32, device=dev)
    ws_l = torch.empty_like(ws_m)
    if WGMMA and length == 8:
        _module().attnwg(q8.view(torch.uint8), k_cache.view(torch.uint8), v_cache.view(torch.uint8), pt, data["seq_lens"],
                         data["k_scale"], data["v_scale"], ws_o, ws_m, ws_l, cfg["qk_scale_factor"] * cfg["sm_scale"] * LOG2E,
                         hkv, page, window, ch, nch, batch, STAGE_KEYS["sliding" if window else "full"],
                         VERIFY_HEADS["sliding" if window else "full"])
    else:
        _module().attn8(q8.view(torch.uint8), k_cache.view(torch.uint8), v_cache.view(torch.uint8), pt, data["seq_lens"],
                        data["k_scale"], data["v_scale"], ws_o, ws_m, ws_l, cfg["qk_scale_factor"] * cfg["sm_scale"] * LOG2E,
                        length, hkv, page, window, ch, nch, batch)
    out = torch.empty((batch, length, hq * d), dtype=torch.bfloat16, device=dev)
    _attn_combine[(groups * rows,)](qkv, out, ws_o, ws_m, ws_l, nch, HKV=hkv, KVPC=2, G=group, L=length, D=d,
                                    RH=rh, QKV_W=qkv.shape[1], HQ=hq, CT=64, num_warps=2)
    return out
