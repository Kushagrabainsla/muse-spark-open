"""Random-row gather bandwidth: rows of RB bytes at random indices into a big array (no writes but a checksum)."""
import torch
from torch.utils.cpp_extension import load_inline
src = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
template <int RB>
__global__ void g(const uint4* __restrict__ a, const int* __restrict__ idx, int n, uint32_t* out) {
  constexpr int TPR = RB / 16;
  uint32_t acc = 0;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < (long long)n * TPR; i += (long long)gridDim.x * blockDim.x) {
    const int r = idx[i / TPR];
    uint4 v = __ldcs(a + (long long)r * TPR + i % TPR);
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x12345678) out[0] = acc;
}
void run(torch::Tensor a, torch::Tensor idx, torch::Tensor out, int64_t rb, int64_t blocks) {
  auto s = at::cuda::getCurrentCUDAStream();
  const uint4* p = reinterpret_cast<const uint4*>(a.data_ptr());
  int n = idx.numel();
  if (rb == 64) g<64><<<blocks, 256, 0, s>>>(p, idx.data_ptr<int>(), n, (uint32_t*)out.data_ptr());
  if (rb == 128) g<128><<<blocks, 256, 0, s>>>(p, idx.data_ptr<int>(), n, (uint32_t*)out.data_ptr());
  if (rb == 256) g<256><<<blocks, 256, 0, s>>>(p, idx.data_ptr<int>(), n, (uint32_t*)out.data_ptr());
  if (rb == 512) g<512><<<blocks, 256, 0, s>>>(p, idx.data_ptr<int>(), n, (uint32_t*)out.data_ptr());
}
"""
m = load_inline("gath", "void run(torch::Tensor a, torch::Tensor idx, torch::Tensor out, int64_t rb, int64_t blocks);",
                cuda_sources=src, functions=["run"], extra_cuda_cflags=["-O3", "-arch=sm_90a"])
a = torch.empty(8 << 30, dtype=torch.uint8, device="cuda")  # 8 GB
out = torch.zeros(1, dtype=torch.int32, device="cuda")
for rb in (64, 128, 256, 512):
    rows = a.numel() // rb
    n = (300 << 20) // rb  # 300 MB gathered
    for kind in ("random", "seq"):
        idx = (torch.randint(0, rows, (n,), device="cuda") if kind == "random" else torch.arange(n, device="cuda")).int()
        best = 1e9
        for blocks in (132 * 8, 132 * 16, 132 * 32):
            for _ in range(2): m.run(a, idx, out, rb, blocks)
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            torch.cuda.synchronize(); e0.record()
            for _ in range(5): m.run(a, idx, out, rb, blocks)
            e1.record(); torch.cuda.synchronize()
            best = min(best, e0.elapsed_time(e1) / 5)
        print(f"row {rb:4d} B {kind:6s}: {n * rb / best / 1e9:6.2f} TB/s", flush=True)
