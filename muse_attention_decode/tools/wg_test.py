"""wgmma m64n64k32 e4m3 layout test: S = Q K^T with K in a 128B-swizzled smem tile (128 B rows, head hh at +64 B),
and O = P V with P in registers (keys in the P-fragment slot order) and V transposed into a 64B-swizzled tile."""
import torch
from torch.utils.cpp_extension import load_inline

def wgmma_fn(n):
    nreg = n // 2
    d = ",".join(f"%{i}" for i in range(nreg))
    a = ",".join(f"%{nreg + i}" for i in range(4))
    outs = ", ".join(f'"+f"(d[{i}])' for i in range(nreg))
    return f'''
__device__ __forceinline__ void wgmma_{n}(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t bdesc,
                                          int scale_d) {{
  asm volatile("{{ .reg .pred p; setp.ne.b32 p, %{nreg + 4}, 0; "
               "wgmma.mma_async.sync.aligned.m64n{n}k32.f32.e4m3.e4m3 {{{d}}}, {{{a}}}, %{nreg + 5}, p, 1, 1; }}"
               : {outs}
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(scale_d), "l"(bdesc));
}}'''

SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
__device__ __forceinline__ uint64_t make_desc(const void* smem, int layout, int sbo) {
  uint64_t d = 0;
  d |= (uint64_t)(((uint32_t)__cvta_generic_to_shared(smem) & 0x3FFFFu) >> 4);
  d |= (uint64_t)(sbo >> 4) << 32;
  d |= (uint64_t)layout << 62;
  return d;
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(N) : "memory"); }
__device__ __forceinline__ void fence_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
''' + wgmma_fn(64) + r'''

// q: [64][64] e4m3 rows = 8 tokens x 8 heads for one head group; k: [64 keys][2 heads][64]; v: same; p: [64][64 keys].
// out_s: [64][64] f32 = q k_hh^T; out_o: [64][64] f32 = p v_hh.
__global__ void __launch_bounds__(128) wg_test_kernel(const uint8_t* q, const uint8_t* k, const uint8_t* v,
                                                      const uint8_t* p, float* out_s, float* out_o, int hh) {
  __shared__ __align__(1024) uint8_t Kt[64 * 128];
  __shared__ __align__(1024) uint8_t VT[64 * 64];
  const int t = threadIdx.x, w = t >> 5, l = t & 31, g = l >> 2, qq = l & 3;
  // K tile: key row r (128 B), 16 B chunk c at c ^ (r & 7).
  for (int i = t; i < 64 * 8; i += 128) {
    const int r = i >> 3, c = i & 7;
    *reinterpret_cast<uint4*>(Kt + r * 128 + ((c ^ (r & 7)) << 4)) = *reinterpret_cast<const uint4*>(k + r * 128 + c * 16);
  }
  // VT: dim row d (64 B), slot s holds key key(s): s = 32u + 16hf + 4qq + e -> key 32u + 16hf + 2qq + (e & 1) + 8 (e >> 1).
  // 16 B chunk c of row d at c ^ ((d >> 1) & 3).
  for (int i = t; i < 64 * 64; i += 128) {
    const int d = i >> 6, s = i & 63;
    const int key = (s & 48) + 2 * ((s >> 2) & 3) + (s & 1) + 8 * ((s >> 1) & 1);
    const int c = s >> 4;
    VT[d * 64 + ((c ^ ((d >> 1) & 3)) << 4) + (s & 15)] = v[key * 128 + hh * 64 + d];
  }
  fence_async();
  __syncthreads();
  const int ra = 16 * w + g, rb = ra + 8;
  float sacc[32];
  for (int i = 0; i < 32; ++i) sacc[i] = 0.f;
  wg_fence();
  for (int s = 0; s < 2; ++s) {
    const uint32_t a0 = *reinterpret_cast<const uint32_t*>(q + ra * 64 + 32 * s + 4 * qq);
    const uint32_t a1 = *reinterpret_cast<const uint32_t*>(q + rb * 64 + 32 * s + 4 * qq);
    const uint32_t a2 = *reinterpret_cast<const uint32_t*>(q + ra * 64 + 32 * s + 16 + 4 * qq);
    const uint32_t a3 = *reinterpret_cast<const uint32_t*>(q + rb * 64 + 32 * s + 16 + 4 * qq);
    wgmma_64(sacc, a0, a1, a2, a3, make_desc(Kt + 64 * hh + 32 * s, 1, 1024), s > 0);
  }
  wg_commit();
  wg_wait<0>();
  for (int i = 0; i < 8; ++i) {
    out_s[ra * 64 + 8 * i + 2 * qq] = sacc[4 * i];
    out_s[ra * 64 + 8 * i + 2 * qq + 1] = sacc[4 * i + 1];
    out_s[rb * 64 + 8 * i + 2 * qq] = sacc[4 * i + 2];
    out_s[rb * 64 + 8 * i + 2 * qq + 1] = sacc[4 * i + 3];
  }
  float oacc[32];
  for (int i = 0; i < 32; ++i) oacc[i] = 0.f;
  wg_fence();
  for (int u = 0; u < 2; ++u) {
    // a0: row ra, slots 4qq..+3 of step u = keys 32u + 2qq + {0, 1, 8, 9}; a2: slots 16 + 4qq.. = keys + 16.
    uint32_t a[4];
    for (int e = 0; e < 4; ++e) {
      const int half = e >> 1, row = (e & 1) ? rb : ra;
      uint32_t x = 0;
      for (int j = 0; j < 4; ++j) {
        const int key = 32 * u + 16 * half + 2 * qq + (j & 1) + 8 * (j >> 1);
        x |= (uint32_t)p[row * 64 + key] << (8 * j);
      }
      a[e] = x;
    }
    wgmma_64(oacc, a[0], a[1], a[2], a[3], make_desc(VT + 32 * u, 2, 512), u > 0);
  }
  wg_commit();
  wg_wait<0>();
  for (int i = 0; i < 8; ++i) {
    out_o[ra * 64 + 8 * i + 2 * qq] = oacc[4 * i];
    out_o[ra * 64 + 8 * i + 2 * qq + 1] = oacc[4 * i + 1];
    out_o[rb * 64 + 8 * i + 2 * qq] = oacc[4 * i + 2];
    out_o[rb * 64 + 8 * i + 2 * qq + 1] = oacc[4 * i + 3];
  }
}
void wg_test(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor p, torch::Tensor out_s, torch::Tensor out_o,
             int64_t hh) {
  wg_test_kernel<<<1, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      (const uint8_t*)q.data_ptr(), (const uint8_t*)k.data_ptr(), (const uint8_t*)v.data_ptr(), (const uint8_t*)p.data_ptr(),
      out_s.data_ptr<float>(), out_o.data_ptr<float>(), (int)hh);
}
'''
m = load_inline("wg_test", "void wg_test(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor p, torch::Tensor out_s, torch::Tensor out_o, int64_t hh);",
                cuda_sources=SRC, functions=["wg_test"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"])
torch.manual_seed(0)
F8 = torch.float8_e4m3fn
q = (torch.randn(64, 64, device="cuda") * 0.5).to(F8)
k = (torch.randn(64, 2, 64, device="cuda") * 0.5).to(F8)
v = (torch.randn(64, 2, 64, device="cuda") * 0.5).to(F8)
p = torch.rand(64, 64, device="cuda").to(F8)
for hh in (0, 1):
    out_s = torch.zeros(64, 64, device="cuda"); out_o = torch.zeros(64, 64, device="cuda")
    m.wg_test(q.view(torch.uint8), k.view(torch.uint8), v.view(torch.uint8), p.view(torch.uint8), out_s, out_o, hh)
    torch.cuda.synchronize()
    ref_s = q.float() @ k[:, hh].float().T
    ref_o = p.float() @ v[:, hh].float()
    print(f"hh={hh} S err {(out_s - ref_s).abs().max().item():.3e} (ref max {ref_s.abs().max().item():.2f})  "
          f"O err {(out_o - ref_o).abs().max().item():.3e} (ref max {ref_o.abs().max().item():.2f})")
