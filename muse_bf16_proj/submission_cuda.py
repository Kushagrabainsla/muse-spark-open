# okc: problem=muse_bf16_proj
"""Dense projections as a cluster split-K skinny GEMM in CUDA.

Up to 32 tokens the projections are bound by streaming the weights. Each warp owns a [16 * RT, 32 *
NSTEPS] slab of W and issues every load of it at once (16-byte loads, no shared memory), then runs
m16n8k16 BF16 MMAs with W as the left operand, so the tokens only fill the MMA's 8-wide side. A
16-byte load holds 8 consecutive k of one row; A and B use the same permutation of k inside each
32-wide block, so it cancels in the dot product. The S CTAs splitting K form one thread-block
cluster and sum their partials through distributed shared memory: no workspace, no atomics.
The router runs as a BF16 hi + lo pair of its FP32 weights with FP32 accumulation. Above 32
tokens it uses cuBLAS, as SGLang does.
"""

from typing import TypedDict

import torch
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
template <int MT, int RT, int NSTEPS, int CH, bool LO, bool OUT_F32>
__global__ void __launch_bounds__(128) skinny_kernel(const __nv_bfloat16* __restrict__ x,
                                                     const __nv_bfloat16* __restrict__ w,
                                                     const __nv_bfloat16* __restrict__ wlo,
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
  for (int i = threadIdx.x; i < ELEMS; i += 128) part[i] = red[0][i] + red[1][i] + red[2][i] + red[3][i];

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

template <int MT, int RT, int NSTEPS, int CH, bool LO>
void launch(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& wlo, torch::Tensor& out, int S) {
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
                     reinterpret_cast<const __nv_bfloat16*>(wlo.data_ptr()), out.data_ptr(), M, N, K, S);
}

#define CASE(MT, RT, NS, CH, LO) \
  if (mt == MT && rt == RT && ns == NS && ch == CH && lo == LO) return launch<MT, RT, NS, CH, LO>(x, w, wlo, out, s);
#define CASES_CH(MT, RT, NS, LO) CASE(MT, RT, NS, 1, LO) CASE(MT, RT, NS, 2, LO) CASE(MT, RT, NS, 4, LO) CASE(MT, RT, NS, 8, LO)
#define CASES_BF16(MT, RT) CASES_CH(MT, RT, 2, false) CASES_CH(MT, RT, 4, false) CASES_CH(MT, RT, 8, false)
#define CASES_LO(MT, RT) CASES_CH(MT, RT, 2, true) CASES_CH(MT, RT, 4, true)

void skinny(torch::Tensor x, torch::Tensor w, torch::Tensor wlo, torch::Tensor out, int64_t mt, int64_t rt,
            int64_t ns, int64_t ch, bool lo, int64_t s) {
  CASES_BF16(1, 1) CASES_BF16(1, 2) CASES_BF16(1, 4)
  CASES_BF16(2, 1) CASES_BF16(2, 2) CASES_BF16(2, 4)
  CASES_BF16(4, 1) CASES_BF16(4, 2) CASES_BF16(4, 4)
  CASES_LO(1, 1) CASES_LO(1, 2) CASES_LO(2, 1) CASES_LO(2, 2) CASES_LO(4, 1) CASES_LO(4, 2)
  TORCH_CHECK(false, "no skinny kernel for this config");
}
"""

CPP_SRC = ("void skinny(torch::Tensor x, torch::Tensor w, torch::Tensor wlo, torch::Tensor out, int64_t mt, "
           "int64_t rt, int64_t ns, int64_t ch, bool lo, int64_t s);")

_mod = None


def _module():
    global _mod
    if _mod is None:
        _mod = load_inline("muse_bf16_skinny", CPP_SRC, cuda_sources=CUDA_SRC, functions=["skinny"],
                           extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
    return _mod


def _config(n: int, k: int, m: int, lo: bool):
    """(MT, RT, NSTEPS, CH, S): token tiles of 8, row tiles of 16 per warp, 32-wide k steps per chunk, chunks
    per warp, CTAs per cluster (S = K / (128 * NSTEPS * CH))."""
    mt = 1 if m <= 8 else 2 if m <= 16 else 4
    rt = 1 if m <= 16 else 2
    ns = 8 if k >= 8192 else 4 if k >= 4096 else 2
    ch = 1
    if lo:
        ns = 4
    s = k // (128 * ns * ch)
    return mt, rt, ns, ch, s


def _pack(w: torch.Tensor, rt: int, ns: int, ch: int, s: int) -> torch.Tensor:
    """W [N, K] reordered (losslessly) so each warp's loads are contiguous:
    [tile][rank][warp][chunk][step][rt][h][lane][8]."""
    n, k = w.shape
    return (w.view(n // (16 * rt), rt, 2, 8, s, 4, ch, ns, 4, 8).permute(0, 4, 5, 6, 7, 1, 2, 3, 8, 9)
            .contiguous().view(n, k))


def _packed(state, name, rt, ns, ch, s):
    key = (name, rt, ns, ch, s)
    if key not in state["packed"]:
        state["packed"][key] = _pack(state[name], rt, ns, ch, s)
    return state["packed"][key]


def _launch(x, name, lo_name, n, out, state):
    m, k = x.shape
    mt, rt, ns, ch, s = _config(n, k, m, lo_name is not None)
    w = _packed(state, name, rt, ns, ch, s)
    wlo = w if lo_name is None else _packed(state, lo_name, rt, ns, ch, s)
    _module().skinny(x, w, wlo, out, mt, rt, ns, ch, lo_name is not None, s)
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
    # Pack every configuration up front: packing is untimed and must not happen in a timed call.
    for name, (n, k) in {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192),
                         "unpooler": (8192, 4096), "router": (256, 8192)}.items():
        for m in (1, 8, 16, 32):
            lo = name == "router"
            _, rt, ns, ch, s = _config(n, k, m, lo)
            _packed(state, "router_hi" if lo else name, rt, ns, ch, s)
            if lo:
                _packed(state, "router_lo", rt, ns, ch, s)
    torch.cuda.synchronize()
    return state


def kernel_fn(data: Inputs, state) -> torch.Tensor:
    x, cfg = data["x"], data["config"]
    m, n, layer = cfg["m"], cfg["n"], cfg["layer"]
    if layer == "router":
        if m <= SKINNY_MAX_M:
            out = torch.empty((m, n), dtype=torch.float32, device=x.device)
            return _launch(x, "router_hi", "router_lo", n, out, state)
        acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
        acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
        return acc
    if m <= SKINNY_MAX_M:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        return _launch(x, layer, None, n, out, state)
    return x @ state[layer][:n].T
