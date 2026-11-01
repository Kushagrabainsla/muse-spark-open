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

Sliding decode steps run all three in one launch (attnc): each request's chunks are one thread-block cluster, every
CTA preps its own queries, the chunk holding the new token stores it, and the partials merge over DSMEM. Sliding
verify blocks of up to 16 requests do the same around the warp-specialized wgmma kernel (attnpcc), the partials
pushed into the merging CTAs' shared memory, whose merge code the idle producer runs once early (cold-L2 replays
otherwise fetch it from DRAM).
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
VERIFY_PC = {"sliding": True, "full": True}  # the warp-specialized wgmma kernel (attnpc_kernel) instead
FUSED = {"sliding": True, "full": False}  # decode steps in one launch (attnc), when the chunks divide 16
FUSED_VERIFY = True  # sliding verify blocks in one launch (attnpcc)


def _chunk(window: int, batch: int, length: int) -> int:
    """Keys per CTA: sliding layers aim at ~128 CTAs (128 keys at least), full layers grow chunks with the batch;
    verify blocks (one CTA per KV head) aim at ~300 CTAs on sliding layers and keep 4 CTAs per SM on full ones."""
    if CHUNK["sliding" if window else "full"]:
        return CHUNK["sliding" if window else "full"]
    if length > 1:
        if window:
            return 64 if batch <= 4 else 512  # bs1 1.45x at 64, bs16 1.08x at 512 (5 draws, the attnpc kernel)
        return 512 if batch <= 2 else 2048
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


CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>

// Programmatic dependent launch: with the launch attribute, griddep_wait() waits for the previous kernel and its
// memory (the decode layer's qkv projection); without it, it returns at once.
__device__ __forceinline__ void griddep_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
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
CUDA_SRC += r"""
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
CUDA_SRC += r"""
#include <cooperative_groups.h>
// Sliding decode steps in one launch (attn + _prep + _attn_combine), the chunks of each (request, KV-head pair) one
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
  // First what the previous kernel (the decode layer's qkv projection, when launched with programmatic
  // serialization) does not write: the scalars (pinned in registers: left to the compiler, the scale loads sink to
  // their uses, a cold DRAM read each at the end of the kernel) and the chunk's page-table rows.
  float pos = (float)positions[b], ksc = kscale[0], vsc = vscale[0];
  long long loc = cache_loc[b];
  int seq = seqlens[b];
  asm volatile("" : "+f"(pos), "+f"(ksc), "+f"(vsc), "+l"(loc), "+r"(seq));
  const int hi = seq + 1;
  const int lo = window > 0 ? max(seq - (window - 1), 0) : 0;
  const int cs = lo + c * CH, ce = min(cs + CH, hi);
  const bool empty = cs >= hi, owner = !empty && seq < ce;
  const int hh = warp & 1, ks = warp >> 1;
  if (!empty)
    for (int i = tid; i < ce - cs; i += NT) {
      const int p = cs + i;
      rows[i] = pt[(size_t)b * max_pages + p / page] * page + p % page;
    }
  griddep_wait();
  // Then the qkv rows: the gates, the queries, the new keys and values (warp 0; only the owner uses them).
  if (tid < RPC * 8) cp16(gs + tid * 8, tok + (HQ + 2 * HKV + h0 * G + c * RPC + jh) * D + d0, true);
  cp_commit();
  // (L2 reads: qkv comes from the kernel this one overlaps, and this SM's L1 may hold the last call's.)
  const uint4 qraw = __ldcg(reinterpret_cast<const uint4*>(tok + (h0 * G + jh) * D + d0));
  uint4 kvraw = make_uint4(0, 0, 0, 0);
  if (warp == 0)  // lanes 0-15 the new keys, 16-31 the new values (head h0 + bit 3 of the lane)
    kvraw = __ldcg(reinterpret_cast<const uint4*>(tok + (HQ + (lane >> 4) * HKV + h0 + ((lane >> 3) & 1)) * D + d0));

  if (!empty) {
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
  cudaLaunchAttribute attr[2];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = 1;
  attr[0].val.clusterDim.y = NCH;
  attr[0].val.clusterDim.z = 1;
  attr[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;  // starts while the qkv projection ends
  attr[1].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 2;
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
CUDA_SRC += r"""
// Sliding verify blocks in one launch (_qkv_prep + attnpc + _attn_combine): attnpc's CTA (one KV head; a producer
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
  // First what the previous kernel (the decode layer's qkv projection, when launched with programmatic
  // serialization) does not write: the scalars (pinned in registers), the first stages' page-table rows (one per
  // thread, stored below) and the RoPE table. Then the qkv rows: the queries (two (row, 8-dim) slices per thread),
  // the new keys and values (threads 0-127: k for 0-63, v for 64-127; token (t >> 3) & 7) and the gates.
  const int s8 = (threadIdx.x & 7) * 8;
  const int kj = (threadIdx.x >> 3) & 7, kpart = (threadIdx.x >> 6) & 1;
  float ksc = kscale[0], vsc = vscale[0], tpos = (float)positions[b * L + (threadIdx.x >> 5)];
  long long kloc = cache_loc[b * L + kj];
  int seq = seqlens[b];
  asm volatile("" : "+f"(ksc), "+f"(vsc), "+f"(tpos), "+l"(kloc), "+r"(seq));
  const int hi = seq + L;
  const int lo = window > 0 ? max(seq - (window - 1), 0) : 0;
  const int cs = lo + c * CH, ce = min(cs + CH, hi);
  const bool empty = cs >= hi, owner = !empty && ce > seq;
  const int first = empty ? 0 : min(ce - cs, SK * (STG - 1));  // <= NT: one page-table read per thread
  const int prow = threadIdx.x < first ? pt[(size_t)b * max_pages + (cs + threadIdx.x) / page] * page +
                                             (cs + threadIdx.x) % page : 0;
  if (!empty && rope) {  // thread: token threadIdx.x / 32, pair threadIdx.x % 32
    float sn, cn;
    sincosf(tpos * exp2f(-(2.f * (float)(threadIdx.x & 31) / 64.f) * theta_log2), &sn, &cn);
    rt[threadIdx.x] = make_float2(sn, cn);
  }
  griddep_wait();
  uint4 qraw[2];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int r = (threadIdx.x >> 3) + 32 * i, j = r >> 3;
    qraw[i] = __ldcg(reinterpret_cast<const uint4*>(tok + (size_t)j * QKV_W + (h * G + (r & 7)) * D + s8));
  }
  uint4 kvraw = make_uint4(0, 0, 0, 0);
  if (threadIdx.x < 128)  // (L2 reads, as attnc_kernel's)
    kvraw = __ldcg(reinterpret_cast<const uint4*>(tok + (size_t)kj * QKV_W + (HQ + kpart * HKV + h) * D + s8));
  if (threadIdx.x < RPC * 8 && c * RPC + (threadIdx.x >> 3) < 64) {
    const int j = c * RPC + (threadIdx.x >> 3);
    cp16(gs + threadIdx.x * 8, tok + (size_t)(j >> 3) * QKV_W + (HQ + 2 * HKV + h * G + (j & 7)) * D + s8, true);
  }
  cp_commit();

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
    __syncthreads();  // the RoPE table
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
  cudaLaunchAttribute attr[2];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = 1;
  attr[0].val.clusterDim.y = NCH;
  attr[0].val.clusterDim.z = 1;
  attr[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;  // starts while the qkv projection ends
  attr[1].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 2;
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
_mod = None


def _module():
    global _mod
    if _mod is None:
        _mod = load_inline("muse_attn_decode_f", CPP_SRC, cuda_sources=CUDA_SRC, functions=["attn", "attn8", "attnwg", "attnpc", "attnc", "attnpcc"],
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
    if length == 8 and window and FUSED_VERIFY and triton.cdiv(window - 1 + length, _vchunk(batch)) <= 16:
        return _kernel_verify_cluster(data)
    fused = FUSED["sliding" if window else "full"] if length == 1 else None
    if fused and 16 % triton.cdiv(window or pt_span(data), _chunk(window, batch, 1)) == 0:
        return _kernel_cluster(data)
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


def pt_span(data: Inputs) -> int:
    return data["page_table"].shape[1] * data["config"]["page_size"]


def _kernel_cluster(data: Inputs) -> torch.Tensor:
    """Decode steps in one launch, each request's chunks one cluster (attnc): the prep, the attention, the merge."""
    cfg = data["config"]
    batch, hq, hkv, d = cfg["batch"], cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    window, theta, page = cfg["window"], cfg["rope_theta"], cfg["page_size"]
    qkv = data["qkv"].view(batch, -1)
    ch = _chunk(window, batch, 1)
    nch = triton.cdiv(window or pt_span(data), ch)
    out = torch.empty((batch, 1, hq * d), dtype=torch.bfloat16, device=qkv.device)
    _module().attnc(qkv, data["positions"], data["cache_loc"], data["k_scale"], data["v_scale"],
                    data["k_cache"].view(torch.uint8), data["v_cache"].view(torch.uint8), data["page_table"],
                    data["seq_lens"], out, math.log2(theta) if theta else 0.0, int(bool(theta)),
                    cfg["qk_scale_factor"] * cfg["sm_scale"] * LOG2E, hkv, page, window, ch, nch, batch,
                    STAGE_KEYS["sliding" if window else "full"] if CHUNK["sliding" if window else "full"] else
                    32 if ch <= 128 else 64)
    return out


def _vchunk(batch: int) -> int:
    """Keys per CTA of attnpcc: at most 16 chunks (one cluster) of the 2,055 keys a sliding verify block sees."""
    return CHUNK["sliding"] or (136 if batch <= 4 else 512)


def _kernel_verify_cluster(data: Inputs) -> torch.Tensor:
    """Sliding verify blocks in one launch, each (request, KV head)'s chunks one cluster (attnpcc)."""
    cfg = data["config"]
    batch, length, hq, hkv, d = cfg["batch"], cfg["q_len"], cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    window, theta, page = cfg["window"], cfg["rope_theta"], cfg["page_size"]
    qkv = data["qkv"].view(batch * length, -1)
    ch = _vchunk(batch)
    out = torch.empty((batch, length, hq * d), dtype=torch.bfloat16, device=qkv.device)
    _module().attnpcc(qkv, data["positions"], data["cache_loc"], data["k_scale"], data["v_scale"],
                      data["k_cache"].view(torch.uint8), data["v_cache"].view(torch.uint8), data["page_table"],
                      data["seq_lens"], out, math.log2(theta) if theta else 0.0, int(bool(theta)),
                      cfg["qk_scale_factor"] * cfg["sm_scale"] * LOG2E, hkv, page, window, ch,
                      triton.cdiv(window - 1 + length, ch), batch)
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
    if WGMMA and length == 8 and VERIFY_PC["sliding" if window else "full"]:
        _module().attnpc(q8.view(torch.uint8), k_cache.view(torch.uint8), v_cache.view(torch.uint8), pt, data["seq_lens"],
                         data["k_scale"], data["v_scale"], ws_o, ws_m, ws_l, cfg["qk_scale_factor"] * cfg["sm_scale"] * LOG2E,
                         hkv, page, window, ch, nch, batch, 64)
    elif WGMMA and length == 8:
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
