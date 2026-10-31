"""Split-K Hopper GEMM for out[M, N] = x[M, K] @ W[N, K]^T (BF16, FP32 accumulation), 32 <= M <= 256.

One thread-block cluster of S CTAs per 128-row tile of W; CTA rank r takes K blocks [r KB / S, (r + 1) KB / S) of 64.
Per CTA: warpgroup 0 (one thread) streams 128 x 64 tiles of W and TOK x 64 tiles of x into a ring of STG stages by
TMA (128 B swizzle, out-of-range token rows zero-filled), warpgroups 1 and 2 each run wgmma m64nTOKk16 on 64 rows of W
(A) against the tokens (B), both from shared memory. The S partial tiles are summed through distributed shared
memory, in rank order, each CTA writing 1/S of the tile.
python wgemm.py  (correctness + timing against SGLang's cuBLAS call)"""
import sys
import torch
from torch.utils.cpp_extension import load_inline


def _wgmma_fn(n: int) -> str:
    nreg = n // 2
    d = ",".join(f"%{i}" for i in range(nreg))
    outs = ", ".join(f'"+f"(d[{i}])' for i in range(nreg))
    return f"""
__device__ __forceinline__ void wgmma_{n}(float* d, uint64_t da, uint64_t db) {{
  asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{d}}}, %{nreg}, %{nreg + 1}, 1, 1, 1, 0, 0;"
               : {outs}
               : "l"(da), "l"(db));
}}"""


SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(uint64_t* bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, int parity) {
  asm volatile(
      "{\n .reg .pred p;\n LAB_WAIT:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @p bra DONE;\n"
      " bra LAB_WAIT;\n DONE:\n }\n" ::"r"(smem_u32(bar)), "r"(parity) : "memory");
}
__device__ __forceinline__ void tma_load(void* dst, const CUtensorMap* map, uint64_t* bar, int c0, int c1) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
      ::"r"(smem_u32(dst)), "l"(map), "r"(smem_u32(bar)), "r"(c0), "r"(c1) : "memory");
}
// K-major tile of 128-byte rows, 128 B swizzle (as TMA writes it), 8-row groups 1024 B apart.
__device__ __forceinline__ uint64_t desc128(const void* p) {
  return (uint64_t)((smem_u32(p) & 0x3FFFFu) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) |
         ((uint64_t)1 << 62);
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void wg_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(N) : "memory");
}
template <int N>
__device__ __forceinline__ void pinf(float* r) {
#pragma unroll
  for (int i = 0; i < N; ++i) asm volatile("" : "+f"(r[i])::"memory");
}
WGMMA_FNS

template <int TOK>
__device__ __forceinline__ void wgmma_tok(float* d, uint64_t da, uint64_t db) {
  if constexpr (TOK == 32) wgmma_32(d, da, db);
  else if constexpr (TOK == 64) wgmma_64(d, da, db);
  else if constexpr (TOK == 128) wgmma_128(d, da, db);
  else wgmma_256(d, da, db);
}

constexpr int PSTRIDE = 132;  // floats per token row of the partial tile (128 + 4: conflict-free stores)

template <int TOK, int STG>
__global__ void __launch_bounds__(384, 1) wgemm_kernel(const __grid_constant__ CUtensorMap wmap,
                                                       const __grid_constant__ CUtensorMap xmap,
                                                       __nv_bfloat16* __restrict__ out, float* __restrict__ ws,
                                                       int* __restrict__ cnt, int M, int N, int KB, int S) {
  constexpr int WB = 128 * 128, XB = TOK * 128, SB = WB + XB;
  extern __shared__ uint8_t smem_raw[];
  uint8_t* smem = smem_raw + ((1024 - (smem_u32(smem_raw) & 1023)) & 1023);
  uint64_t* full = reinterpret_cast<uint64_t*>(smem + STG * SB);
  uint64_t* empty = full + STG;
  float* part = reinterpret_cast<float*>(smem);  // after the main loop: [TOK][PSTRIDE]

  const int tile = blockIdx.x / S, rank = blockIdx.x % S;
  const int wg = threadIdx.x >> 7, tid = threadIdx.x & 127;
  const int n0 = tile * 128;
  if (threadIdx.x == 0) {
    for (int s = 0; s < STG; ++s) {
      mbar_init(&full[s], 1);
      mbar_init(&empty[s], 2);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  const int kb0 = rank * KB / S, nk = (rank + 1) * KB / S - kb0;
  __syncthreads();

  float acc[TOK / 2];
#pragma unroll
  for (int i = 0; i < TOK / 2; ++i) acc[i] = 0.f;
  if (wg == 0) {
    if (tid == 0) {
      for (int it = 0; it < nk; ++it) {
        const int s = it % STG, r = it / STG;
        if (r > 0) mbar_wait(&empty[s], (r - 1) & 1);
        uint8_t* st = smem + s * SB;
        mbar_expect_tx(&full[s], SB);
        const int kc = (kb0 + it) * 64;
        tma_load(st, &wmap, &full[s], 0, (tile * KB + kb0 + it) * 128);  // packed W: one 16 KB run
        tma_load(st + WB, &xmap, &full[s], kc, 0);
      }
    }
  } else {
    const int c = wg - 1;
    for (int it = 0; it < nk; ++it) {
      const int s = it % STG;
      mbar_wait(&full[s], (it / STG) & 1);
      const uint8_t* wt = smem + s * SB + c * 64 * 128;
      const uint8_t* xt = smem + s * SB + WB;
      uint64_t da[4], db[4];
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        da[k] = desc128(wt + 32 * k);
        db[k] = desc128(xt + 32 * k);
      }
      wg_fence();
#pragma unroll
      for (int k = 0; k < 4; ++k) wgmma_tok<TOK>(acc, da[k], db[k]);
      wg_commit();
      wg_wait<1>();  // the previous stage's wgmmas are done: release its slot
      if (it > 0 && tid == 0) mbar_arrive(&empty[(it - 1) % STG]);
    }
    wg_wait<0>();
    pinf<TOK / 2>(acc);
  }
  __syncthreads();  // every wgmma has read its stage: the ring becomes the partial tile
  if (wg > 0) {
    const int c = wg - 1, w4 = tid >> 5, g = (tid & 31) >> 2, q = tid & 3;
    const int ra = 64 * c + 16 * w4 + g;
#pragma unroll
    for (int j = 0; j < TOK / 8; ++j)
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int tok = 8 * j + 2 * q + e;
        part[tok * PSTRIDE + ra] = acc[4 * j + e];
        part[tok * PSTRIDE + ra + 8] = acc[4 * j + 2 + e];
      }
  }
  __syncthreads();
  // Split-K: every split's partial tile goes to ws[split][token][row]; the S CTAs of the tile meet at a counter
  // (the grid is one wave, so they are all resident), then each sums its 1/S of the tile in split order and writes
  // it as BF16. The last to leave resets the counters for the next call.
  const int E4 = M * 32, e0 = rank * E4 / S, e1 = (rank + 1) * E4 / S;
  if (S > 1) {
    for (int i = threadIdx.x; i < E4; i += 384) {
      if (i >= e0 && i < e1) continue;  // its own slice stays in shared memory
      const int tok = i >> 5, r4 = (i & 31) * 4;
      __stcg(reinterpret_cast<float4*>(ws + ((size_t)rank * M + tok) * N + n0 + r4),
             *reinterpret_cast<const float4*>(part + tok * PSTRIDE + r4));
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
      atomicAdd(&cnt[2 * tile], 1);
      int seen;
      do {
        asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(seen) : "l"(&cnt[2 * tile]) : "memory");
      } while (seen < S);
    }
    __syncthreads();
  }
  for (int i = e0 + threadIdx.x; i < e1; i += 384) {
    const int tok = i >> 5, r4 = (i & 31) * 4;
    float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int src = 0; src < S; ++src) {
      const float4 v = src == rank ? *reinterpret_cast<const float4*>(part + tok * PSTRIDE + r4)
                                   : __ldcg(reinterpret_cast<const float4*>(ws + ((size_t)src * M + tok) * N + n0 + r4));
      sum.x += v.x;
      sum.y += v.y;
      sum.z += v.z;
      sum.w += v.w;
    }
    __nv_bfloat162 lo = __floats2bfloat162_rn(sum.x, sum.y), hi = __floats2bfloat162_rn(sum.z, sum.w);
    uint2 pk = make_uint2(*reinterpret_cast<uint32_t*>(&lo), *reinterpret_cast<uint32_t*>(&hi));
    *reinterpret_cast<uint2*>(out + (size_t)tok * N + n0 + r4) = pk;
  }
  if (S > 1 && threadIdx.x == 0 && atomicAdd(&cnt[2 * tile + 1], 1) == S - 1) {
    cnt[2 * tile] = 0;
    cnt[2 * tile + 1] = 0;
  }
}

static PFN_cuTensorMapEncodeTiled_v12000 encode_fn = nullptr;

static CUtensorMap make_map(const void* ptr, uint64_t rows, uint64_t cols, uint32_t box_rows) {
  if (!encode_fn) {
    cudaDriverEntryPointQueryResult q;
    TORCH_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", (void**)&encode_fn, cudaEnableDefault, &q) ==
                    cudaSuccess && q == cudaDriverEntryPointSuccess, "cuTensorMapEncodeTiled unavailable");
  }
  CUtensorMap m;
  cuuint64_t dims[2] = {cols, rows};
  cuuint64_t strides[1] = {cols * 2};
  cuuint32_t box[2] = {64, box_rows};
  cuuint32_t estr[2] = {1, 1};
  CUresult r = encode_fn(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr), dims, strides, box, estr,
                         CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                         CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed");
  return m;
}

template <int TOK, int STG>
static int occupancy(int S) {  // clusters of S CTAs resident at once
  auto kernel = wgemm_kernel<TOK, STG>;
  const size_t smem = (size_t)STG * (128 * 128 + TOK * 128) + 2 * STG * 8 + 1024;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  int per_sm = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, 384, smem);
  return per_sm * 132 / S;
}

template <int TOK, int STG>
static void launch(const torch::Tensor& x, const torch::Tensor& w, torch::Tensor& out, torch::Tensor& ws,
                   torch::Tensor& cnt, int S) {
  const int M = x.size(0), K = x.size(1), N = out.size(1);
  TORCH_CHECK(M <= TOK && N % 128 == 0 && K % 64 == 0, "wgemm: shape");
  // w is packed [N / 128][K / 64][128][64]: a 2D map of 64-wide rows, boxes of 128 rows.
  const CUtensorMap wmap = make_map(w.data_ptr(), (uint64_t)N * K / 64, 64, 128), xmap = make_map(x.data_ptr(), M, K, TOK);
  auto kernel = wgemm_kernel<TOK, STG>;
  const size_t smem = (size_t)STG * (128 * 128 + TOK * 128) + 2 * STG * 8 + 1024;
  static bool set = false;
  if (!set) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    set = true;
  }
  TORCH_CHECK(S == 1 || ws.numel() >= (int64_t)S * M * N, "wgemm: workspace");
  TORCH_CHECK((N / 128) * S <= 132, "wgemm: the split-K barrier needs a single wave");
  kernel<<<(N / 128) * S, 384, smem, at::cuda::getCurrentCUDAStream()>>>(
      wmap, xmap, reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), ws.data_ptr<float>(), cnt.data_ptr<int>(), M, N,
      K / 64, S);
}

#define CONFIGS(X) X(32, 4) X(32, 8) X(64, 3) X(64, 6) X(128, 3) X(128, 6) X(256, 3) X(256, 4)
void wgemm(torch::Tensor x, torch::Tensor w, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, int64_t tok,
           int64_t stg, int64_t S) {  // w packed
#define CASE(T, G) if (tok == T && stg == G) return launch<T, G>(x, w, out, ws, cnt, (int)S);
  CONFIGS(CASE)
  TORCH_CHECK(false, "wgemm: no (tok, stg) instance");
}
int64_t wgemm_occ(int64_t tok, int64_t stg, int64_t S) {
#define OCC(T, G) if (tok == T && stg == G) return occupancy<T, G>((int)S);
  CONFIGS(OCC)
  return -1;
}
"""

_mod = None


def module():
    global _mod
    if _mod is None:
        src = SRC.replace("WGMMA_FNS", "".join(_wgmma_fn(n) for n in (32, 64, 128, 256)))
        _mod = load_inline("wgemm_t", "void wgemm(torch::Tensor x, torch::Tensor w, torch::Tensor out, torch::Tensor ws, "
                           "torch::Tensor cnt, int64_t tok, int64_t stg, int64_t S); int64_t wgemm_occ(int64_t tok, int64_t stg, int64_t S);",
                           cuda_sources=src, functions=["wgemm", "wgemm_occ"],
                           extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
    return _mod


def pack(w: torch.Tensor) -> torch.Tensor:
    """W [N, K] -> [N / 128][K / 64][128][64] (lossless), each tile of a CTA's K loop one contiguous 16 KB run."""
    n, k = w.shape
    return w.view(n // 128, 128, k // 64, 64).permute(0, 2, 1, 3).contiguous().view(n, k)


def tok_of(m):
    return 32 if m <= 32 else 64 if m <= 64 else 128 if m <= 128 else 256


if __name__ == "__main__":
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    import importlib.util
    import reference as ref
    from timing import kernel_us
    spec = importlib.util.spec_from_file_location("base", __file__.rsplit("/", 1)[0] + "/baseline.py")
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    weights = ref.generate_weights(ref.WEIGHTS_SEED)
    st_b = base.prepare({k: v.clone() for k, v in weights.items()})
    mod = module()
    WS = torch.zeros(16 * 256 * 8192, dtype=torch.float32, device="cuda")
    CNT = torch.zeros(1024, dtype=torch.int32, device="cuda")
    layers = sys.argv[1].split(",") if len(sys.argv) > 1 else ["qkv_gate", "pooler", "o", "unpooler"]
    ms = [int(v) for v in sys.argv[2].split(",")] if len(sys.argv) > 2 else [32, 128, 256]
    splits = [int(v) for v in sys.argv[3].split(",")] if len(sys.argv) > 3 else [1, 2, 4, 7, 8]
    stgs = [int(v) for v in sys.argv[4].split(",")] if len(sys.argv) > 4 else [0]
    for layer in layers:
        for m in ms:
            data = ref.generate_input(m, layer, seed=99)
            x, n = data["x"], data["config"]["n"]
            w = weights[layer][:n].contiguous()
            wp = pack(w)
            refo = x.float() @ w.float().T
            tb = kernel_us(base.kernel_fn, data, st_b)
            line = f"{layer:9s} m={m:4d}: sglang {tb:6.1f}"
            tk = tok_of(m)
            for g in stgs:
                g = g or {32: 8, 64: 6, 128: 6, 256: 4}[tk]
                for s in splits:
                    if (n // 128) * s > 300 or s > 16:
                        continue
                    out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
                    try:
                        mod.wgemm(x, wp, out, WS, CNT, tk, g, s)
                        torch.cuda.synchronize()
                    except Exception as e:  # noqa: BLE001
                        line += f"  G{g}S{s} ERR {str(e)[:40]}"
                        continue
                    err = float((out.float() - refo).norm() / refo.norm())
                    t = kernel_us(lambda d, st, s=s, g=g, out=out: mod.wgemm(x, wp, out, WS, CNT, tk, g, s), data, None)
                    occ = mod.wgemm_occ(tk, g, s)
                    line += f"  G{g}S{s} {t:5.1f}[{(n // 128)}x{s}/{occ * s}]{'' if err < 1e-2 else ' BAD'}"
            print(line, flush=True)
