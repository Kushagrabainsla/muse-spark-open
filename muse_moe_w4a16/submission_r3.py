# okc: problem=muse_moe_w4a16
"""Routed experts as NVFP4 grouped GEMMs in CUDA.

The step is bound by reading the touched experts' weights, so only (token, expert) pairs routed to local
experts are computed, and each touched expert's weights are streamed once per token group.

  route    in every gemm CTA (shared memory): per local expert, its pairs (token * 8 + slot) in pair order;
           the active experts; and per active expert the token groups (of 8 * MT) before it.
  gemm1    gate/up, h1[pair] = x[token] @ W13[e]^T; gemm2: down, y[pair] = act(h1[pair]) @ W2[e]^T, where act
           is silu(gate) * up rounded to BF16 (as SGLang's intermediate is) when gemm2 stages its input.
           Both are persistent: the work is units of (k split, expert, token group, 256-row tile), and each
           CTA takes an equal contiguous run of them. A CTA stages its run's x[tokens, K / S] in shared memory
           once; each of its 8 warps streams its own 16 * RT rows through a 3-stage cp.async ring and adds
           its split-K partial into FP32 accumulators with red.global.add.
  norm     per token: each local slot's y * (1 + w), RMS-normalized, weighted and summed in slot order; then
           zeroes the accumulator rows it consumed, so they are zero again for the next call.

The weights are repacked once (a lossless permutation of the E2M1 bits and scale bytes) so that each warp
load is a contiguous 512 B and each lane's 16 bytes feed eight m16n8k16 BF16 MMAs, one per 16-wide scale
block. An E2M1 code becomes BF16 by moving its bits into place (sign to 15, e to 8-7, m to 6): exact, up to
a uniform 2^-126, with code 1 (0.5) a BF16 subnormal. The block's E4M3 scale, times 2^100, multiplies those
values in BF16 (exact: E2M1 x E4M3 fits BF16's mantissa, and the products are normal); the 2^26 left over is
folded into the expert's FP32 global scale. x stays BF16. MMAs accumulate in FP32.
"""

from typing import TypedDict

import torch
from torch.utils.cpp_extension import load_inline


class Weights(TypedDict):
    w13: torch.Tensor          # uint8 [256, 2I, H/2], gate rows [0, I) then up rows [I, 2I), packed E2M1
    w13_sf: torch.Tensor       # float8_e4m3fn [256, 2I, H/16]
    w13_gs: torch.Tensor       # float32 [256]
    w2: torch.Tensor           # uint8 [256, H, I/2]
    w2_sf: torch.Tensor        # float8_e4m3fn [256, H, I/16]
    w2_gs: torch.Tensor        # float32 [256]
    post_norm_w: torch.Tensor  # bfloat16 [H], the post-expert RMSNorm's weight (applied as 1 + w)


class Config(TypedDict):
    bs: int           # tokens, M
    d_hidden: int     # H = 4096, the width the MoE block runs its experts at
    d_expert: int     # I = 4096
    n_experts: int    # 256 experts in the model
    local_start: int  # this GPU computes experts [local_start, local_start + local_count)
    local_count: int  # 32: one GPU under 8-way expert parallelism
    top_k: int        # 8
    sharding: str     # "ep8"


class Inputs(TypedDict):
    hidden_states: torch.Tensor  # bfloat16 [M, H]
    topk_ids: torch.Tensor       # int32 [M, 8], global expert ids (0-255)
    topk_weights: torch.Tensor   # float32 [M, 8]
    config: Config


LOCAL_START, LOCAL_COUNT = 0, 32
TOP_K = 8
POST_NORM_EPS = 1e-8
# Split-K ways of the gate/up and down GEMMs, and rows per warp (16 * RT).
# (largest m, (split-K ways of gate/up, of down)). From sweeps on an H200: 16 up to 16 tokens (finer units balance
# the few active experts), 8 above.
SPLITS = ((16, (16, 16)), (1 << 30, (8, 8)))
# (largest m, MT): the token tile is 8 * MT; an expert with more tokens takes several groups.
MT_BY_M = ((16, 1), (1 << 30, 2))
RT = 2
# Tokens up to which the FP32 accumulators are kept in state; above, they are allocated per call.
ACC_MAX_M = 512
NW = 8  # warps per CTA, matching the kernel's NW
ROUTE_INTS = 34 + 32 * 4 + 3 * NW * 32  # the kernel's R_PAIRS

CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;
typedef __nv_bfloat16 bf16;
constexpr int NW = 8;

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

// A packed word holds four BF16 pairs of E2M1 values, bit-permuted at prepare time so each comes out with
// shifts and a mask: each value's sign at bit 15 (31), e at bits 8-7 (24-23), m at bit 6 (22) of its half.
constexpr uint32_t E2M1_MASK = 0x81C081C0u;
__device__ __forceinline__ void e2m1x8(uint32_t q, uint32_t& o0, uint32_t& o1, uint32_t& o2, uint32_t& o3) {
  o0 = q & E2M1_MASK;
  o1 = (q << 3) & E2M1_MASK;
  o2 = (q << 6) & E2M1_MASK;
  o3 = ((q >> 4) & 0x00C000C0u) | ((q >> 5) & 0x01000100u) | ((q << 1) & 0x80008000u);
}

// Bytes j, j + 1 of the 4 E4M3 scales in word, times 2^100, as a BF16 pair. Exact either way.
// EXACT: the hardware conversion. Otherwise, valid only for positive normal scales (prepare() checks): the byte's
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
constexpr int R_BALL = R_WC + NW * 32, R_PRE = R_BALL + NW * 32, R_PAIRS = R_PRE + NW * 32;

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
  const bool one_pass = M <= NW * 32;
  if (tid < 32) {
    R[R_CNT + tid] = 0;
    R[R_RUN + tid] = 0;
  }
  R[R_WC + tid] = 0;  // NW * 32 threads: one entry each
  R[R_BALL + tid] = 0;
  __syncthreads();
#pragma unroll 1
  for (int t0 = 0; t0 < M; t0 += NW * 32) {  // counts (and, for one pass, the ballots)
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
  for (int t0 = 0; t0 < M; t0 += NW * 32) {  // positions
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
    if (tid < 32) R[R_RUN + tid] = R[R_PRE + (NW - 1) * 32 + tid] + R[R_WC + (NW - 1) * 32 + tid];
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
// A CTA tile is NW * 16 * RT rows (each warp 16 * RT of them); K is split S ways (KC = K / S, NST steps of
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

__device__ __forceinline__ void red_add(float* p, float v) {
  asm volatile("red.global.add.f32 [%0], %1;" ::"l"(p), "f"(v) : "memory");
}

template <int MT, int RT, int MODE, bool EXACT, int KC>
__global__ void __launch_bounds__(NW * 32) moe_gemm(const bf16* __restrict__ x, const float* __restrict__ h1,
                                                     const uint4* __restrict__ wq, const uint2* __restrict__ wsf,
                                                     const float* __restrict__ gscale, const int* __restrict__ ids,
                                                     int local_start, int local_count, int* __restrict__ rglob, int M,
                                                     float* __restrict__ acc_out,
                                                     int out_stride, int K, int tiles, int S) {
  constexpr int TK = 8 * MT, ROWS = NW * 16 * RT;
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
  int* R = reinterpret_cast<int*>(smem + ring_offset(TK, XS) + (size_t)NW * STG * SUB * RT * 2 * (32 * 16 + 8 * 8));
  if constexpr (MODE == 0) {
    route_cta<TK>(ids, M, local_start, local_count, R);
    // CTA 0 publishes the routing (meta, act, cnt, off, then the pairs) before it lets gemm2 launch.
    if (blockIdx.x == 0) {
      const int np = R[R_OFF + 31] + R[R_CNT + 31];
      for (int i = threadIdx.x; i < R_OFF + 32; i += NW * 32) rglob[i] = R[i];
      for (int i = threadIdx.x; i < np; i += NW * 32) rglob[R_PAIRS + i] = R[R_PAIRS + i];
      __threadfence();
      __syncthreads();
      griddep_launch();
    }
  } else {
    // gemm2 launches only after gemm1's CTA 0 published (L2 reads: this SM's L1 may hold the last call's).
    for (int i = threadIdx.x; i < R_OFF + 32; i += NW * 32) R[i] = __ldcg(rglob + i);
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
  const size_t tile_step = ((size_t)S * NW * nst - (nst - SUB)) * RT * 2;  // slot (tile + 1, 0) - (tile, nst - SUB)
  auto set_run = [&](It& it) {
    it.le = act[it.a];
    it.ngrp = (cnt[it.le] + TK - 1) / TK;
    it.te = min(tiles, it.tb + it.left);
    it.tile = it.tb;
    it.st = 0;
    it.slot = (((((size_t)it.le * tiles + it.tile) * S + it.rank) * NW + warp) * nst) * RT * 2;
  };
  auto next = [&](It& it) {
    if (++it.st < nsg) {
      it.slot += SUB * RT * 2;
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
  uint4* wring = reinterpret_cast<uint4*>(smem + ring_offset(TK, XS)) + warp * STG * SUB * RT * 2 * 32;
  uint2* sring = reinterpret_cast<uint2*>(reinterpret_cast<uint4*>(smem + ring_offset(TK, XS)) +
                                          NW * STG * SUB * RT * 2 * 32) +
                 warp * STG * SUB * RT * 2 * 8;
  auto issue = [&](int stage, size_t slot0) {  // SUB consecutive steps: their slots are contiguous
#pragma unroll
    for (int j = 0; j < SUB * RT * 2; ++j) {
      const int i = stage * SUB * RT * 2 + j;
      cp_async16(wring + i * 32 + lane, wq + (slot0 + j) * 32 + lane);
    }
    // The steps' scale blocks (64 B each) are contiguous too: SUB * RT * 2 * 64 B as 16 B per lane.
    static_assert(SUB * RT * 2 * 64 == 32 * 16, "one 16-byte scale copy per lane per stage");
    cp_async16(reinterpret_cast<uint4*>(sring + stage * SUB * RT * 2 * 8) + lane,
               reinterpret_cast<const uint4*>(wsf + slot0 * 8) + lane);
  };

  float acc[RT][MT][4];
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
        for (int i = threadIdx.x; i < TK * chunks; i += NW * 32) {
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
      for (int rt = 0; rt < RT; ++rt)
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
    uint4 wv[RT][2];
    uint2 sv[RT][2];
#pragma unroll
    for (int rt = 0; rt < RT; ++rt)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int i = (stage * SUB + sub) * RT * 2 + rt * 2 + h;
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
      for (int rt = 0; rt < RT; ++rt) {
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
      const int row0 = tile * ROWS + warp * 16 * RT + g;
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int p = stok[mt * 8 + 2 * q + (i & 1)];
          if (p < 0) continue;
#pragma unroll
          for (int rt = 0; rt < RT; ++rt)
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
                                                   float eps, bf16* __restrict__ out) {
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
template <int MT, int RT, int MODE, bool EXACT, int KC>
void launch_gemm_kc(const torch::Tensor& x, const torch::Tensor& h1, const torch::Tensor& wq, const torch::Tensor& wsf,
                 const torch::Tensor& gs, const torch::Tensor& ids, int ls, int lc, const torch::Tensor& rglob, int M,
                 torch::Tensor& out, int K,
                 int tiles, int S) {
  constexpr int TK = 8 * MT;
  auto kernel = moe_gemm<MT, RT, MODE, EXACT, KC>;
  const size_t smem = ring_offset(TK, KC + 32) + (size_t)NW * STG * SUB * RT * 2 * (32 * 16 + 8 * 8) +
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
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, NW * 32, smem);
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
  cfg.blockDim = dim3(NW * 32);
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
                     tiles, S);
}

// K per split (KC) is a compile-time constant of the kernel: 512 (split 8) or 256 (split 16) of K = 4096.
template <int MT, int RT, int MODE, bool EXACT>
void launch_gemm(const torch::Tensor& x, const torch::Tensor& h1, const torch::Tensor& wq, const torch::Tensor& wsf,
                 const torch::Tensor& gs, const torch::Tensor& ids, int ls, int lc, const torch::Tensor& rglob, int M,
                 torch::Tensor& out, int K,
                 int tiles, int S) {
  if (K / S == 512)
    return launch_gemm_kc<MT, RT, MODE, EXACT, 512>(x, h1, wq, wsf, gs, ids, ls, lc, rglob, M, out, K, tiles, S);
  if (K / S == 256)
    return launch_gemm_kc<MT, RT, MODE, EXACT, 256>(x, h1, wq, wsf, gs, ids, ls, lc, rglob, M, out, K, tiles, S);
  TORCH_CHECK(false, "moe_gemm is built for K / S = 512 or 256");
}

void moe_forward(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor w13q, torch::Tensor w13sf,
                 torch::Tensor w13gs, torch::Tensor w2q, torch::Tensor w2sf, torch::Tensor w2gs, torch::Tensor nw,
                 torch::Tensor rglob, torch::Tensor h1, torch::Tensor y, torch::Tensor out, int64_t mt, int64_t rt, int64_t S1, int64_t S2,
                 int64_t local_start, int64_t local_count, double eps, bool exact) {
  const int M = x.size(0), H = x.size(1), I = h1.size(1) / 2;
  auto stream = at::cuda::getCurrentCUDAStream();
  const int ls = (int)local_start, lc = (int)local_count;
  const int rows = NW * 16 * (int)rt;
#define GEMMS_(MT_, RT_, EX_)                                                                                    \
  if (mt == MT_ && rt == RT_ && exact == EX_) {                                                                  \
    launch_gemm<MT_, RT_, 0, EX_>(x, h1, w13q, w13sf, w13gs, ids, ls, lc, rglob, M, h1, H, 2 * I / rows, S1);  \
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
                     (int)local_count, (float)eps, reinterpret_cast<bf16*>(out.data_ptr()));
}
"""

CPP_SRC = ("void moe_forward(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor w13q, "
           "torch::Tensor w13sf, torch::Tensor w13gs, torch::Tensor w2q, torch::Tensor w2sf, torch::Tensor w2gs, "
           "torch::Tensor nw, torch::Tensor rglob, torch::Tensor h1, torch::Tensor y, torch::Tensor out, int64_t mt, int64_t rt, int64_t S1, int64_t S2, "
           "int64_t local_start, int64_t local_count, double eps, bool exact);")

_mod = None


def _module():
    global _mod
    if _mod is None:
        _mod = load_inline("muse_moe_nvfp4_r3", CPP_SRC, cuda_sources=CUDA_SRC, functions=["moe_forward"],
                           extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
    return _mod


def _unpack_codes(packed: torch.Tensor) -> torch.Tensor:
    """uint8 [..., K/2] -> uint8 E2M1 codes [..., K], element 2i from the low nibble."""
    return torch.stack((packed & 0x0F, packed >> 4), dim=-1).flatten(-2)


def _bit_map():
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


_BIT_MAP = _bit_map()


def _scales_simple(sf: torch.Tensor) -> bool:
    """Whether every E4M3 scale is positive and normal (the kernel's fast conversion), else the exact one."""
    b = sf.view(torch.uint8)
    return bool(((b & 0x80) == 0).all() and ((b & 0x78) != 0).all() and ((b & 0x7f) != 0x7f).all())


def _pack_expert(q: torch.Tensor, sf: torch.Tensor, rt: int, s: int):
    """One expert's weights [N, K] into the kernel's layout.

    Rows: row = ((tile * NW + warp) * RT + rt) * 16 + h * 8 + g. K: k = (rank * NST + step) * 128 + (2w + b) * 16
    + 4q + 2 r_hi + r_lo. Packed order: tile, rank, warp, step, rt, h, g, q, then per lane 32 nibbles
    (w, r_lo, b, r_hi). Scales: tile, rank, warp, step, rt, h, g, then the step's 8 block scales."""
    n, k = q.shape[0], q.shape[1] * 2
    tiles, nst = n // (NW * 16 * rt), k // (s * 128)
    codes = _unpack_codes(q).view(tiles, NW, rt, 2, 8, s, nst, 4, 2, 4, 4)
    # Per lane word w: the 8 values (b, r) = (block 2w + b, position 4q + r), each code bit to its place.
    codes = codes.permute(0, 5, 1, 6, 2, 3, 4, 9, 7, 8, 10).contiguous().view(-1, 8).to(torch.int32)
    word = torch.zeros(codes.shape[0], dtype=torch.int32, device=q.device)
    for v, cb, pos in _BIT_MAP:
        word |= ((codes[:, v] >> cb) & 1) << pos
    packed = word.view(torch.uint8)
    sfv = sf.view(torch.uint8).view(tiles, NW, rt, 2, 8, s, nst, 8)
    return packed, sfv.permute(0, 5, 1, 6, 2, 3, 4, 7).contiguous().view(-1)


def _config(m: int):
    """(MT, RT, S1, S2): token tile of 8 * MT, 16 * RT rows per warp, split-K ways of the two GEMMs."""
    mt = next(t for top, t in MT_BY_M if m <= top)
    s1, s2 = next(sp for top, sp in SPLITS if m <= top)
    return mt, RT, s1, s2


def prepare(weights: Weights):
    dev = weights["w13"].device
    inner = weights["w13"].shape[1] // 2
    hidden = weights["w2"].shape[1]
    lo, hi = LOCAL_START, LOCAL_START + LOCAL_COUNT
    # One packing of each projection per split it is used with (the split changes the k order).
    packs = {}
    for name, idx in (("w13", 0), ("w2", 1)):
        for split in sorted({sp[idx] for _, sp in SPLITS}):
            qs, sfs = zip(*(_pack_expert(weights[name][e], weights[name + "_sf"][e], RT, split) for e in range(lo, hi)))
            packs[(name, split)] = (torch.cat(qs), torch.cat(sfs))
    state = {
        "packs": packs,
        "w13gs": weights["w13_gs"][lo:hi].float().contiguous(),
        "w2gs": weights["w2_gs"][lo:hi].float().contiguous(),
        "nw": weights["post_norm_w"].float() + 1.0,
        "inner": inner,
        "exact": not all(_scales_simple(weights[k][lo:hi]) for k in ("w13_sf", "w2_sf")),
        # FP32 split-K accumulators, one row per (token, slot). They start at zero, and the last kernel of each
        # call zeroes every row it consumed, so they are zero again at the start of the next call.
        "h1": torch.zeros((ACC_MAX_M * TOP_K, 2 * inner), dtype=torch.float32, device=dev),
        "y": torch.zeros((ACC_MAX_M * TOP_K, hidden), dtype=torch.float32, device=dev),
    }
    _module()
    torch.cuda.synchronize()
    return state


def kernel_fn(data: Inputs, state) -> torch.Tensor:
    x = data["hidden_states"]
    m, h = x.shape
    dev = x.device
    mt, rt, s1, s2 = _config(m)
    w13q, w13sf = state["packs"][("w13", s1)]
    w2q, w2sf = state["packs"][("w2", s2)]
    if m <= ACC_MAX_M:
        h1, y = state["h1"][: m * TOP_K], state["y"][: m * TOP_K]
    else:
        h1 = torch.zeros((m * TOP_K, 2 * state["inner"]), dtype=torch.float32, device=dev)
        y = torch.zeros((m * TOP_K, h), dtype=torch.float32, device=dev)
    out = torch.empty((m, h), dtype=torch.bfloat16, device=dev)
    rglob = torch.empty(ROUTE_INTS + TOP_K * m, dtype=torch.int32, device=dev)  # gemm1 -> gemm2 routing
    _module().moe_forward(x, data["topk_ids"], data["topk_weights"], w13q, w13sf, state["w13gs"], w2q, w2sf,
                          state["w2gs"], state["nw"], rglob, h1, y, out, mt, rt, s1, s2, LOCAL_START,
                          LOCAL_COUNT, POST_NORM_EPS, state["exact"])
    return out
