"""cuBLASLt algorithm search for out[m, n] = x[m, k] @ W[n, k]^T (BF16): every heuristic candidate timed like the
judge (CUDA graphs, cold L2), against SGLang's baseline call. python ltprobe.py [layers] [ms] [max_algos]"""
import sys
import torch
from torch.utils.cpp_extension import load_inline
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import reference as ref
from timing import kernel_us

SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublasLt.h>
#include <map>
#include <tuple>
#include <vector>

struct Entry {
  cublasLtMatmulDesc_t op;
  cublasLtMatrixLayout_t a, b, c;
  std::vector<cublasLtMatmulHeuristicResult_t> res;
};
static cublasLtHandle_t handle = nullptr;
static std::map<std::tuple<int64_t, int64_t, int64_t>, Entry> cache;
static void* ws = nullptr;
static const size_t WS = 64ull << 20;
#define CK(x) TORCH_CHECK((x) == CUBLAS_STATUS_SUCCESS, #x)

// Column-major view: C (n x m, ld n) = A^T (A: k x n, ld k = W row-major) * B (k x m, ld k = x row-major).
int64_t lt_search(int64_t m, int64_t n, int64_t k, int64_t max_algos) {
  if (!handle) {
    CK(cublasLtCreate(&handle));
    TORCH_CHECK(cudaMalloc(&ws, WS) == cudaSuccess);
  }
  auto key = std::make_tuple(m, n, k);
  Entry e;
  CK(cublasLtMatmulDescCreate(&e.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  CK(cublasLtMatmulDescSetAttribute(e.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
  CK(cublasLtMatmulDescSetAttribute(e.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
  CK(cublasLtMatrixLayoutCreate(&e.a, CUDA_R_16BF, k, n, k));
  CK(cublasLtMatrixLayoutCreate(&e.b, CUDA_R_16BF, k, m, k));
  CK(cublasLtMatrixLayoutCreate(&e.c, CUDA_R_16BF, n, m, n));
  cublasLtMatmulPreference_t pref;
  CK(cublasLtMatmulPreferenceCreate(&pref));
  size_t wsz = WS;
  CK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsz, sizeof(wsz)));
  e.res.resize(max_algos);
  int found = 0;
  CK(cublasLtMatmulAlgoGetHeuristic(handle, e.op, e.a, e.b, e.c, e.c, pref, (int)max_algos, e.res.data(), &found));
  e.res.resize(found);
  cublasLtMatmulPreferenceDestroy(pref);
  cache[key] = e;
  return found;
}

void lt_run(torch::Tensor x, torch::Tensor w, torch::Tensor out, int64_t idx) {
  const int64_t m = x.size(0), k = x.size(1), n = out.size(1);
  auto& e = cache.at(std::make_tuple(m, n, k));
  const float alpha = 1.f, beta = 0.f;
  CK(cublasLtMatmul(handle, e.op, &alpha, w.data_ptr(), e.a, x.data_ptr(), e.b, &beta, out.data_ptr(), e.c,
                    out.data_ptr(), e.c, &e.res[idx].algo, ws, WS, at::cuda::getCurrentCUDAStream()));
}

std::vector<int64_t> lt_info(int64_t m, int64_t n, int64_t k, int64_t idx) {
  auto& e = cache.at(std::make_tuple(m, n, k));
  const cublasLtMatmulAlgo_t* a = &e.res[idx].algo;
  std::vector<int64_t> v;
  auto get = [&](cublasLtMatmulAlgoConfigAttributes_t attr) {
    uint64_t val = 0;  // every attribute used here fits
    size_t written = 0;
    cublasLtMatmulAlgoConfigGetAttribute(a, attr, &val, sizeof(val), &written);
    v.push_back((int64_t)val);
  };
  get(CUBLASLT_ALGO_CONFIG_ID);
  get(CUBLASLT_ALGO_CONFIG_TILE_ID);
  get(CUBLASLT_ALGO_CONFIG_STAGES_ID);
  get(CUBLASLT_ALGO_CONFIG_SPLITK_NUM);
  get(CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME);
  get(CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING);
  get(CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID);
  v.push_back((int64_t)e.res[idx].workspaceSize);
  v.push_back((int64_t)(e.res[idx].wavesCount * 1000));
  return v;
}
"""
CPP = ("int64_t lt_search(int64_t m, int64_t n, int64_t k, int64_t max_algos);"
       "void lt_run(torch::Tensor x, torch::Tensor w, torch::Tensor out, int64_t idx);"
       "std::vector<int64_t> lt_info(int64_t m, int64_t n, int64_t k, int64_t idx);")
mod = load_inline("ltprobe", SRC, functions=["lt_search", "lt_run", "lt_info"], with_cuda=True,
                  extra_ldflags=["-lcublasLt"], extra_cflags=["-O2"], verbose=False)  # host code only

import importlib.util
spec = importlib.util.spec_from_file_location("base", __file__.rsplit("/", 1)[0] + "/baseline.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
weights = ref.generate_weights(ref.WEIGHTS_SEED)
st_b = base.prepare({k: v.clone() for k, v in weights.items()})
layers = sys.argv[1].split(",") if len(sys.argv) > 1 else ["qkv_gate", "o", "pooler", "unpooler"]
ms = [int(v) for v in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1, 8, 16, 32, 128, 256]
max_algos = int(sys.argv[3]) if len(sys.argv) > 3 else 64
for layer in layers:
    for m in ms:
        data = ref.generate_input(m, layer, seed=99)
        x, n = data["x"], data["config"]["n"]
        w = weights[layer][:n].contiguous()
        k = x.shape[1]
        out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
        tb = kernel_us(base.kernel_fn, data, st_b)
        cnt = mod.lt_search(m, n, k, max_algos)
        res = []
        for i in range(cnt):
            try:
                t = kernel_us(lambda d, s, i=i: mod.lt_run(x, w, out, i), data, None)
            except Exception as e:  # noqa: BLE001
                continue
            res.append((t, i))
        res.sort()
        ref_out = (x.float() @ w.float().T)
        mod.lt_run(x, w, out, res[0][1])
        err = float((out.float() - ref_out).norm() / ref_out.norm())
        best = ", ".join(f"{t:.1f}us#{i}{mod.lt_info(m, n, k, i)[:4]}" for t, i in res[:3])
        print(f"{layer:9s} m={m:4d} n={n:5d}: sglang {tb:6.1f}  lt-default {res and [t for t, i in res if i == 0][0]:6.1f}  "
              f"best {res[0][0]:6.1f} x{tb / res[0][0]:.3f}  ({cnt} algos; err {err:.1e})  top: {best}", flush=True)
