# okc: problem=muse_bf16_proj
"""Dense projections as a cluster split-K skinny GEMM in CUDA.

Up to 32 tokens the projections are bound by streaming the weights. Each warp owns a [16 * RT, 32 *
NSTEPS] slab of W and issues every load of it at once (16-byte loads, no shared memory), then runs
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

void skinny(torch::Tensor x, torch::Tensor w, torch::Tensor wlo, torch::Tensor out, int64_t mt, int64_t rt,
            int64_t ns, int64_t ch, bool lo, int64_t s) {
#define CASE(MT, RT, NS, CH, LO) \
  if (mt == MT && rt == RT && ns == NS && ch == CH && lo == LO) return launch<MT, RT, NS, CH, LO>(x, w, wlo, out, s);
  CASES
  TORCH_CHECK(false, "no skinny kernel for this config");
}
"""

CPP_SRC = ("void skinny(torch::Tensor x, torch::Tensor w, torch::Tensor wlo, torch::Tensor out, int64_t mt, "
           "int64_t rt, int64_t ns, int64_t ch, bool lo, int64_t s);")

_mod = None


def _module():
    global _mod
    if _mod is None:
        cases = " ".join(f"CASE({mt}, {rt}, {ns}, {ch}, {'true' if layer == 'router' else 'false'})"
                         for (layer, mt), (rt, ns, ch, _) in CONFIGS.items())
        _mod = load_inline("muse_bf16_skinny", CPP_SRC, cuda_sources=CUDA_SRC.replace("CASES", cases),
                           functions=["skinny"],
                           extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
    return _mod


def _mt(m: int) -> int:
    return 1 if m <= 8 else 2 if m <= 16 else 4


# (layer, MT) -> (RT, NSTEPS, CH, S): row tiles of 16 per warp, 32-wide k steps per chunk, chunks per warp,
# CTAs per cluster (K = 128 * NSTEPS * CH * S). From sweeps on an H200 under CUDA graphs with a cold L2;
# shapes not listed (o, the whole unpooler) are faster in cuBLAS.
CONFIGS = {
    ("qkv_gate", 1): (2, 8, 1, 8), ("qkv_gate", 2): (2, 4, 2, 8), ("qkv_gate", 4): (2, 4, 4, 4),
    ("pooler", 1): (2, 8, 2, 4), ("pooler", 2): (2, 4, 4, 4), ("pooler", 4): (2, 4, 8, 2),
    ("unpooler", 1): (1, 4, 1, 8),
    ("router", 1): (1, 2, 2, 16), ("router", 2): (1, 4, 1, 16), ("router", 4): (2, 4, 1, 16),
}


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


def _packed(state, name, rt, ns, ch, s):
    key = (name, rt, ns, ch, s)
    if key not in state["packed"]:
        state["packed"][key] = _pack(state[name], rt, ns, ch, s)
    return state["packed"][key]


def _launch(x, name, lo_name, cfg, out, state):
    mt, rt, ns, ch, s = cfg
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
    for (layer, _), (rt, ns, ch, s) in CONFIGS.items():
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
