# okc: problem=muse_bf16_proj
"""Dense projections as a cluster split-K skinny GEMM in CUDA.

Up to 32 tokens the projections are bound by streaming the weights. Each warp owns a [16 * RT, 32 *
NSTEPS] slab of W and issues every load of it at once (16-byte loads, no shared memory); up to 8 tokens the
weights are stored losslessly in 12 bits (low byte + a 4-bit code of sign and exponent, rare out-of-window weights
corrected from a per-CTA exception list), which is 25 % less to stream, then runs
m16n8k16 BF16 MMAs with W as the left operand, so the tokens only fill the MMA's 8-wide side. A
16-byte load holds 8 consecutive k of one row; A and B use the same permutation of k inside each
32-wide block, so it cancels in the dot product. The S CTAs splitting K form one thread-block
cluster and sum their partials through distributed shared memory: no workspace, no atomics.
The router runs as a BF16 hi + lo pair of its FP32 weights with FP32 accumulation; above 32 tokens
it is one split-K Triton GEMM over both halves (SGLang runs two GEMMs and an add). Otherwise, above
32 tokens and for the shapes CONFIGS leaves out, it uses cuBLAS, as SGLang does.
"""

from typing import TypedDict

import torch
import triton
import triton.language as tl
from torch.utils.cpp_extension import load_inline


class Weights(TypedDict):
    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
    o: torch.Tensor         # bfloat16 [8192, 1024]
    pooler: torch.Tensor    # bfloat16 [4096, 8192]
    unpooler: torch.Tensor  # bfloat16 [8192, 4096]
    router: torch.Tensor    # float32 [256, 8192]


class Config(TypedDict):
    m: int
    n: int      # output width; the unpooler's is 1024 up to 8 tokens (this GPU's slice), else 8192
    k: int
    layer: str  # which weight: "qkv_gate", "o", "pooler", "unpooler" or "router"


class Inputs(TypedDict):
    x: torch.Tensor  # bfloat16 [m, k]
    config: Config


SKINNY_MAX_M = 32
# The router's split-K GEMM above SKINNY_MAX_M: (BM, BN, BK, S, warps, stages), from a sweep at m = 128, 256.
ROUTER_GEMM = (64, 64, 128, 8, 4, 4)
ROUTER_GEMM_MAX_M = 4096


@triton.jit
def _router_gemm(x_ptr, hi_ptr, lo_ptr, out_ptr, ws_ptr, cnt_ptr, M, N, K,
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


def _router_gemm_launch(x, state):
    bm, bn, bk, s, warps, stages = ROUTER_GEMM
    m, k = x.shape
    n = state["router_hi"].shape[0]
    out = torch.empty((m, n), dtype=torch.float32, device=x.device)
    _router_gemm[(triton.cdiv(m, bm), n // bn, s)](x, state["router_hi"], state["router_lo"], out, state["ws"],
                                                   state["cnt"], m, n, k, BM=bm, BN=bn, BK=bk, S=s,
                                                   num_warps=warps, num_stages=stages)
    return out

CUDA_SRC = r"""
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

// 12-bit weights (C12): per lane and (step, rt), 16 low bytes (the BF16 low byte: exponent LSB + mantissa) and 8 code
// bytes (low nibble: value k of row g, high nibble: value k of row g + 8; nibble = sign << 3 | (exponent >> 1) - e0h).
// Returns the BF16 pair of values 2 pair, 2 pair + 1 of the 4 in lbw (the codes in cdw).
__device__ __forceinline__ uint32_t dec12(uint32_t lbw, uint32_t cdw, int pair, bool hi_half, uint32_t base) {
  const uint32_t t = __byte_perm(lbw, 0, pair ? 0x4342 : 0x4140);
  uint32_t t2 = __byte_perm(cdw, 0, pair ? 0x3424 : 0x1404);
  if (hi_half) t2 >>= 4;
  return t | ((t2 & 0x08000800u) << 4) | ((t2 & 0x07000700u) + base);
}

// Weights are read once: don't keep them in L1.
__device__ __forceinline__ uint4 ld_stream(const void* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}

// out[M, N] = x[M, K] @ W[N, K]^T (+ x @ Wlo^T). Grid: (N / BN) * S CTAs of 128 threads, clusters of S
// along x; CTA rank r, warp w covers k in [(4r + w) * 32 * NSTEPS * CH, +32 * NSTEPS * CH), as CH chunks of
// NSTEPS 32-wide steps. Each chunk's loads are all issued before its MMAs; other resident warps hide the wait.
// C12: the weights are 12-bit (see dec12); weights whose exponent is below the 16-binade window decode to the
// window's bottom, and the CTA's exception list (row in the tile, k, true and decoded BF16 values) corrects them
// exactly in FP32 before the cluster reduction.
template <int MT, int RT, int NSTEPS, int CH, bool LO, bool OUT_F32, bool C12>
__global__ void __launch_bounds__(128) skinny_kernel(const __nv_bfloat16* __restrict__ x,
                                                     const __nv_bfloat16* __restrict__ w,
                                                     const __nv_bfloat16* __restrict__ wlo,
                                                     const uint2* __restrict__ wc, const uint2* __restrict__ wloc,
                                                     const uint2* __restrict__ exc, const int* __restrict__ exoff,
                                                     int e0h, int e0h_lo,
                                                     void* __restrict__ out, int M, int N, int K, int S) {
  constexpr int BN = 16 * RT, TK = 8 * MT, ELEMS = BN * TK;
  __shared__ float red[4][ELEMS];
  __shared__ float part[ELEMS];

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
  // C12 exceptions (at most 128 per CTA, prepare() checks): thread t takes entry t, loaded here so it arrives while
  // the weights stream; its x values are read at the end (L2 hits: this CTA reads those columns).
  int ecnt = 0;
  uint2 en = make_uint2(0, 0);
  if constexpr (C12) {
    const int ebeg = exoff[blockIdx.x];
    ecnt = exoff[blockIdx.x + 1] - ebeg;
    if ((int)threadIdx.x < ecnt) en = exc[ebeg + threadIdx.x];
  }

#pragma unroll 1
  for (int ch = 0; ch < CH; ++ch) {
    const size_t wbase = wwarp * CH + ch;
    const int kw = (int)wbase % (S * 4 * CH) * (32 * NSTEPS) + 8 * q;

    uint4 wv[C12 ? 1 : NSTEPS][RT][2];
    uint4 lv[LO && !C12 ? NSTEPS : 1][RT][2];
    uint4 wb[C12 ? NSTEPS : 1][RT], lb[C12 && LO ? NSTEPS : 1][RT];
    uint2 wcd[C12 ? NSTEPS : 1][RT], lcd[C12 && LO ? NSTEPS : 1][RT];
#pragma unroll
    for (int s = 0; s < NSTEPS; ++s)
#pragma unroll
      for (int rt = 0; rt < RT; ++rt) {
        if constexpr (C12) {
          const size_t g = ((wbase * NSTEPS + s) * RT + rt) * 32 + lane;  // this lane's 16 values
          wb[s][rt] = ld_stream(reinterpret_cast<const uint8_t*>(w) + g * 16);
          wcd[s][rt] = __ldcs(wc + g);
          if constexpr (LO) {
            lb[s][rt] = ld_stream(reinterpret_cast<const uint8_t*>(wlo) + g * 16);
            lcd[s][rt] = __ldcs(wloc + g);
          }
        } else {
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const size_t off = (((wbase * NSTEPS + s) * RT + rt) * 2 + h) * 256 + lane * 8;
            wv[s][rt][h] = ld_stream(w + off);
            if constexpr (LO) lv[s][rt][h] = ld_stream(wlo + off);
          }
        }
      }
    const uint32_t base = ((uint32_t)e0h << 8) | ((uint32_t)e0h << 24);
    const uint32_t base_lo = ((uint32_t)e0h_lo << 8) | ((uint32_t)e0h_lo << 24);

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
      for (int rt = 0; rt < RT; ++rt) {
        uint4 a, b, c, d;
        if constexpr (C12) {
          // Word j of row g holds values 2j, 2j + 1: low bytes 2j, 2j + 1 of the 16, codes in bytes 2j, 2j + 1.
          const uint4 q4 = wb[s][rt];
          const uint2 c2 = wcd[s][rt];
          a = make_uint4(dec12(q4.x, c2.x, 0, false, base), dec12(q4.x, c2.x, 1, false, base),
                         dec12(q4.y, c2.y, 0, false, base), dec12(q4.y, c2.y, 1, false, base));
          b = make_uint4(dec12(q4.z, c2.x, 0, true, base), dec12(q4.z, c2.x, 1, true, base),
                         dec12(q4.w, c2.y, 0, true, base), dec12(q4.w, c2.y, 1, true, base));
          if constexpr (LO) {
            const uint4 r4 = lb[s][rt];
            const uint2 d2 = lcd[s][rt];
            c = make_uint4(dec12(r4.x, d2.x, 0, false, base_lo), dec12(r4.x, d2.x, 1, false, base_lo),
                           dec12(r4.y, d2.y, 0, false, base_lo), dec12(r4.y, d2.y, 1, false, base_lo));
            d = make_uint4(dec12(r4.z, d2.x, 0, true, base_lo), dec12(r4.z, d2.x, 1, true, base_lo),
                           dec12(r4.w, d2.y, 0, true, base_lo), dec12(r4.w, d2.y, 1, true, base_lo));
          }
        } else {
          a = wv[s][rt][0];
          b = wv[s][rt][1];
          if constexpr (LO) {
            c = lv[s][rt][0];
            d = lv[s][rt][1];
          }
        }
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          const uint4 v = xv[s][mt];
          mma_16816(acc[rt][mt], a.x, b.x, a.y, b.y, v.x, v.y);
          mma_16816(acc[rt][mt], a.z, b.z, a.w, b.w, v.z, v.w);
          if constexpr (LO) {
            mma_16816(acc[rt][mt], c.x, d.x, c.y, d.y, v.x, v.y);
            mma_16816(acc[rt][mt], c.z, d.z, c.w, d.w, v.z, v.w);
          }
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
  for (int i = threadIdx.x; i < ELEMS; i += 128) part[i] = red[0][i] + red[1][i] + red[2][i] + red[3][i];
  if constexpr (C12) {
    // Exceptions: entry .x = row in the tile << 16 | k, .y = true BF16 << 16 | decoded BF16.
    __syncthreads();
    if ((int)threadIdx.x < ecnt) {
      const int row = en.x >> 16, kk = en.x & 0xFFFF;
      const float wt = __uint_as_float(en.y & 0xFFFF0000u), wd = __uint_as_float(en.y << 16);
      float xe[TK];
#pragma unroll
      for (int tok = 0; tok < TK; ++tok) xe[tok] = tok < M ? __bfloat162float(x[(size_t)tok * K + kk]) : 0.f;
#pragma unroll
      for (int tok = 0; tok < TK; ++tok)
        if (tok < M) atomicAdd(&part[tok * BN + row], xe[tok] * wt - xe[tok] * wd);
    }
  }

  cluster.sync();
  // Each CTA of the cluster reduces every S-th element across the cluster and writes it.
  for (int i = rank * 128 + threadIdx.x; i < ELEMS; i += S * 128) {
    const int tok = i / BN, row = i % BN;
    if (tok >= M) continue;
    float sum = 0.f;
    for (int r = 0; r < S; ++r) sum += cluster.map_shared_rank(part, r)[i];
    const size_t o = (size_t)tok * N + n0 + row;
    if constexpr (OUT_F32) reinterpret_cast<float*>(out)[o] = sum;
    else reinterpret_cast<__nv_bfloat16*>(out)[o] = __float2bfloat16(sum);
  }
  cluster.sync();  // keep this CTA's shared memory alive until the others have read it
}

template <int MT, int RT, int NSTEPS, int CH, bool LO, bool C12>
void launch(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& wlo, const torch::Tensor& wc,
            const torch::Tensor& wloc, const torch::Tensor& exc, const torch::Tensor& exoff, int e0h, int e0h_lo,
            torch::Tensor& out, int S) {
  const int M = x.size(0), K = x.size(1), N = out.size(1);
  auto kernel = skinny_kernel<MT, RT, NSTEPS, CH, LO, LO, C12>;
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
                     reinterpret_cast<const __nv_bfloat16*>(wlo.data_ptr()),
                     reinterpret_cast<const uint2*>(wc.data_ptr()), reinterpret_cast<const uint2*>(wloc.data_ptr()),
                     reinterpret_cast<const uint2*>(exc.data_ptr()), exoff.data_ptr<int>(), e0h, e0h_lo,
                     out.data_ptr(), M, N, K, S);
}

void skinny(torch::Tensor x, torch::Tensor w, torch::Tensor wlo, torch::Tensor wc, torch::Tensor wloc,
            torch::Tensor exc, torch::Tensor exoff, int64_t e0h, int64_t e0h_lo, torch::Tensor out, int64_t mt,
            int64_t rt, int64_t ns, int64_t ch, bool lo, bool c12, int64_t s) {
#define CASE(MT, RT, NS, CH, LO, C12) \
  if (mt == MT && rt == RT && ns == NS && ch == CH && lo == LO && c12 == C12) \
    return launch<MT, RT, NS, CH, LO, C12>(x, w, wlo, wc, wloc, exc, exoff, (int)e0h, (int)e0h_lo, out, s);
  CASES
  TORCH_CHECK(false, "no skinny kernel for this config");
}
"""

CPP_SRC = ("void skinny(torch::Tensor x, torch::Tensor w, torch::Tensor wlo, torch::Tensor wc, torch::Tensor wloc, "
           "torch::Tensor exc, torch::Tensor exoff, int64_t e0h, int64_t e0h_lo, torch::Tensor out, int64_t mt, "
           "int64_t rt, int64_t ns, int64_t ch, bool lo, bool c12, int64_t s);")

_mod = None


def _module():
    global _mod
    if _mod is None:
        cases = " ".join(f"CASE({mt}, {rt}, {ns}, {ch}, {'true' if layer == 'router' else 'false'}, {c12})"
                         for (layer, mt), (rt, ns, ch, _) in CONFIGS.items()
                         for c12 in (("true", "false") if mt <= C12_MAX_MT else ("false",)))
        _mod = load_inline("muse_bf16_skinny_c12ref", CPP_SRC, cuda_sources=CUDA_SRC.replace("CASES", cases),
                           functions=["skinny"],
                           extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
    return _mod


def _mt(m: int) -> int:
    return 1 if m <= 8 else 2 if m <= 16 else 4


# (layer, MT) -> (RT, NSTEPS, CH, S): row tiles of 16 per warp, 32-wide k steps per chunk, chunks per warp,
# CTAs per cluster (K = 128 * NSTEPS * CH * S). From sweeps on an H200 under CUDA graphs with a cold L2;
# shapes not listed (o, the whole unpooler) are faster in cuBLAS. MT = 1 (up to 8 tokens) runs the 12-bit weights.
CONFIGS = {
    ("qkv_gate", 1): (1, 16, 1, 4), ("qkv_gate", 2): (2, 4, 2, 8), ("qkv_gate", 4): (2, 4, 4, 4),
    ("pooler", 1): (2, 16, 1, 4), ("pooler", 2): (2, 4, 4, 4), ("pooler", 4): (2, 4, 8, 2),
    ("unpooler", 1): (1, 2, 2, 8),
    ("router", 1): (1, 2, 2, 16), ("router", 2): (1, 4, 1, 16), ("router", 4): (2, 4, 1, 16),
}
# Token tiles (MT) up to which the weights are read as 12 bits: fewer bytes win while streaming dominates; with more
# tokens per weight the decode costs more than it saves (sweeps: 10-12 % faster at 1-8 tokens, slower at 32).
C12_MAX_MT = 1


def _config(layer: str, m: int):
    """(MT, RT, NSTEPS, CH, S) for the skinny kernel, or None for cuBLAS."""
    if m > SKINNY_MAX_M or (layer, _mt(m)) not in CONFIGS:
        return None
    return (_mt(m),) + CONFIGS[(layer, _mt(m))]


def _pack(w: torch.Tensor, rt: int, ns: int, ch: int, s: int) -> torch.Tensor:
    """W [N, K] reordered (losslessly) so each warp's loads are contiguous:
    [tile][rank][warp][chunk][step][rt][h][lane][8]."""
    n, k = w.shape
    return (w.view(n // (16 * rt), rt, 2, 8, s, 4, ch, ns, 4, 8).permute(0, 4, 5, 6, 7, 1, 2, 3, 8, 9)
            .contiguous().view(n, k))


def _compress12(w: torch.Tensor, rt: int, ns: int, ch: int, s: int):
    """Lossless 12-bit form of W [N, K] for the C12 kernel: (low bytes, codes, exceptions, exception offsets, e0h).

    Packed like _pack, then per lane and (step, rt): the 16 values' low bytes (row g's 8, then row g + 8's) and 8 code
    bytes (value k of row g in the low nibble, of row g + 8 in the high one). Code = sign << 3 | (exponent >> 1) - e0h,
    e0h = max(exponent >> 1) - 7: 16 binades below the largest weight. A weight below them decodes to code 0 (the
    window's bottom with its own low byte); its CTA's exception list carries (row in the tile << 16 | k, true BF16
    << 16 | decoded BF16), so the kernel adds x * true - x * decoded."""
    n, k = w.shape
    u = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    hi = u >> 8
    eh = hi & 0x7F
    e0h = int(eh.max()) - 7
    code = ((hi >> 7) << 3) | (eh - e0h).clamp(0, 7)
    dec = ((((code >> 3) << 7) | ((code & 7) + e0h)) << 8) | (u & 0xFF)  # what the kernel decodes
    # Exceptions, ordered by CTA: tile = n // (16 rt), rank = k // (K / s); CTA = tile * s + rank.
    en, ek = torch.nonzero(dec != u, as_tuple=True)
    cta = (en // (16 * rt)) * s + ek // (k // s)
    order = torch.argsort(cta)
    en, ek, cta = en[order], ek[order], cta[order]
    ex = torch.stack(((en % (16 * rt)) << 16 | ek, (u[en, ek] << 16) | dec[en, ek]), dim=1).to(torch.int64)
    ex = ((ex + (1 << 31)) % (1 << 32) - (1 << 31)).to(torch.int32).contiguous()  # uint32 bits in int32
    ctas = (n // (16 * rt)) * s
    exoff = torch.zeros(ctas + 1, dtype=torch.int32, device=w.device)
    exoff[1:] = torch.cumsum(torch.bincount(cta, minlength=ctas), 0).to(torch.int32)
    # 12-bit planes in the kernel's order.
    p = _pack(w, rt, ns, ch, s).view(torch.int16).to(torch.int32) & 0xFFFF
    pc = _pack(code.to(torch.int16), rt, ns, ch, s).to(torch.int32)
    low = (p & 0xFF).view(-1, 2, 32, 8).permute(0, 2, 1, 3).contiguous().to(torch.uint8).view(-1)
    g = pc.view(-1, 2, 32, 8)
    nib = (g[:, 0] | (g[:, 1] << 4)).to(torch.uint8).contiguous().view(-1)
    return low, nib, ex, exoff, e0h


def _packed(state, name, rt, ns, ch, s, c12=False):
    key = (name, rt, ns, ch, s, c12)
    if key not in state["packed"]:
        state["packed"][key] = _compress12(state[name], rt, ns, ch, s) if c12 else _pack(state[name], rt, ns, ch, s)
    return state["packed"][key]


def _launch(x, name, lo_name, cfg, out, state):
    mt, rt, ns, ch, s = cfg
    if mt <= C12_MAX_MT and state["c12_ok"][(name, rt, ns, ch, s)]:
        w, wc, ex, exoff, e0h = _packed(state, name, rt, ns, ch, s, True)
        wlo, wloc, e0h_lo = w, wc, e0h
        if lo_name is not None:
            wlo, wloc, _, _, e0h_lo = _packed(state, lo_name, rt, ns, ch, s, True)
            ex, exoff = state["packed"][("router_ex", rt, ns, ch, s)]
    else:
        w = _packed(state, name, rt, ns, ch, s)
        wlo = w if lo_name is None else _packed(state, lo_name, rt, ns, ch, s)
        wc = wloc = ex = exoff = state["dummy"]
        e0h = e0h_lo = 0
    c12 = mt <= C12_MAX_MT and state["c12_ok"][(name, rt, ns, ch, s)]
    if mt <= C12_MAX_MT and not c12:
        w = _packed(state, name, rt, ns, ch, s)
        wlo = w if lo_name is None else _packed(state, lo_name, rt, ns, ch, s)
    _module().skinny(x, w, wlo, wc, wloc, ex, exoff, e0h, e0h_lo, out, mt, rt, ns, ch, lo_name is not None, c12, s)
    return out


def prepare(weights: Weights):
    state = {name: w.contiguous() for name, w in weights.items()}
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state["router_hi"] = hi.contiguous()
    state["router_lo"] = lo.contiguous()
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()
    state["packed"] = {}
    _module()
    state["dummy"] = torch.zeros(8, dtype=torch.int32, device=router.device)
    # The 12-bit form needs at most 128 out-of-window weights per CTA (one per thread); weights with more (not these
    # Gaussian ones: ~1e-4 of them, a few per CTA) keep the BF16 path.
    state["c12_ok"] = {}
    # Pack every configuration up front: packing is untimed and must not happen in a timed call.
    for (layer, mt), (rt, ns, ch, s) in CONFIGS.items():
        c12 = mt <= C12_MAX_MT
        for name in (("router_hi", "router_lo") if layer == "router" else (layer,)):
            _packed(state, name, rt, ns, ch, s, c12)
        if layer == "router" and c12:
            # The router's two planes share one exception list per CTA (the kernel adds both corrections).
            _, _, ex_h, off_h, _ = _packed(state, "router_hi", rt, ns, ch, s, True)
            _, _, ex_l, off_l, _ = _packed(state, "router_lo", rt, ns, ch, s, True)
            ctas = off_h.numel() - 1
            cnt = (off_h[1:] - off_h[:-1]) + (off_l[1:] - off_l[:-1])
            off = torch.zeros(ctas + 1, dtype=torch.int32, device=router.device)
            off[1:] = torch.cumsum(cnt, 0).to(torch.int32)
            cta_h = torch.repeat_interleave(torch.arange(ctas, device=router.device), (off_h[1:] - off_h[:-1]).long())
            cta_l = torch.repeat_interleave(torch.arange(ctas, device=router.device), (off_l[1:] - off_l[:-1]).long())
            allc = torch.cat((cta_h, cta_l))
            order = torch.argsort(allc, stable=True)
            state["packed"][("router_ex", rt, ns, ch, s)] = (torch.cat((ex_h, ex_l))[order].contiguous(), off)
        if c12:
            ex_key = ("router_ex", rt, ns, ch, s) if layer == "router" else None
            off = state["packed"][ex_key][1] if ex_key else _packed(state, layer, rt, ns, ch, s, True)[3]
            ok = int((off[1:] - off[:-1]).max()) <= 128
            state["c12_ok"][("router_hi" if layer == "router" else layer, rt, ns, ch, s)] = ok
            if not ok:  # BF16 fallback for this shape
                for name in (("router_hi", "router_lo") if layer == "router" else (layer,)):
                    _packed(state, name, rt, ns, ch, s)
    bm, bn, _, splits, _, _ = ROUTER_GEMM
    state["ws"] = torch.empty(splits * ROUTER_GEMM_MAX_M * router.shape[0], dtype=torch.float32, device=router.device)
    state["cnt"] = torch.zeros(triton.cdiv(ROUTER_GEMM_MAX_M, bm) * (router.shape[0] // bn), dtype=torch.int32,
                               device=router.device)
    # Compile the router GEMM outside any graph capture (one compile covers every m).
    _router_gemm_launch(torch.zeros((64, router.shape[1]), dtype=torch.bfloat16, device=router.device), state)
    torch.cuda.synchronize()
    return state


def kernel_fn(data: Inputs, state) -> torch.Tensor:
    x, cfg = data["x"], data["config"]
    m, n, layer = cfg["m"], cfg["n"], cfg["layer"]
    skinny = _config(layer, m)
    if layer == "router":
        if skinny is not None:
            out = torch.empty((m, n), dtype=torch.float32, device=x.device)
            return _launch(x, "router_hi", "router_lo", skinny, out, state)
        if m <= ROUTER_GEMM_MAX_M:
            return _router_gemm_launch(x, state)
        acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
        acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
        return acc
    if skinny is not None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        return _launch(x, layer, None, skinny, out, state)
    return x @ state[layer][:n].T
