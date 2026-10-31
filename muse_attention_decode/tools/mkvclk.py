"""python mkvclk.py in.py out.py: attnpcc with per-CTA phase clocks (thread 0 = consumer warp 0; 16 slots)."""
import sys
t = open(sys.argv[1]).read()
k0 = t.index("__global__ void __launch_bounds__(256, 2) attnpcc_kernel(")
k1 = t.index("void attnpcc(torch::Tensor qkv")
head, body, tail = t[:k0], t[k0:k1], t[k1:]
def rep(a, b):
    global body
    assert body.count(a) == 1, a
    body = body.replace(a, b)
rep('  cooperative_groups::cluster_group cl = cooperative_groups::this_cluster();\n',
    '  cooperative_groups::cluster_group cl = cooperative_groups::this_cluster();\n  long long* CK = g_vclk + (blockIdx.y * gridDim.x + blockIdx.x) * 16;\n  if (threadIdx.x == 0) { CK[0] = vgtime(); CK[1] = clock64(); CK[9] = 0; }\n')
rep('''    __syncthreads();  // P1
''', '''    __syncthreads();  // P1
    if (threadIdx.x == 0) CK[2] = clock64();
''')
rep('''      load_rows(first, ce - cs);
      __syncthreads();  // P2: stages 0 and 1 landed
''', '''      load_rows(first, ce - cs);
      __syncthreads();  // P2: stages 0 and 1 landed
      if (threadIdx.x == 0) { CK[3] = clock64(); CK[9] = nsteps; }
''')
rep('''      // Rows ra and rb go to the CTAs merging them (DSMEM stores: no wait).''', '''      if (threadIdx.x == 0) { CK[4] = clock64(); CK[5] = CK[4]; }
      // Rows ra and rb go to the CTAs merging them (DSMEM stores: no wait).''')
rep('''  cp_wait<0>();  // the gate logits
  cl.sync();      // every chunk's partials have landed
''', '''  cp_wait<0>();  // the gate logits
  if (threadIdx.x == 0) CK[6] = clock64();
  cl.sync();      // every chunk's partials have landed
  if (threadIdx.x == 0) CK[7] = clock64();
''')

i = body.rindex('}')
body = body[:i] + '''  if (threadIdx.x == 0) { CK[8] = clock64(); CK[10] = vgtime() - CK[0]; }\n''' + body[i:]
i = head.rindex("template <int STG, int SK>\n")
head = head[:i] + ('__device__ long long g_vclk[8192 * 16];\n__device__ __forceinline__ long long vgtime() { long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }\n'
    'torch::Tensor vclk_read() { auto t = torch::empty({8192 * 16}, torch::dtype(torch::kInt64)); cudaMemcpyFromSymbol(t.data_ptr(), g_vclk, sizeof(g_vclk)); return t; }\n') + head[i:]
t = head + body + tail
t = t.replace('"int64_t window, int64_t CH, int64_t NCH, int64_t batch);")\n_mod', '"int64_t window, int64_t CH, int64_t NCH, int64_t batch);\\ntorch::Tensor vclk_read();")\n_mod')
t = t.replace('"attnc", "attnpcc"]', '"attnc", "attnpcc", "vclk_read"]')
t = t.replace('load_inline("muse_attn_decode_f"', 'load_inline("muse_attn_decode_vclk"')
assert '"vclk_read"]' in t and "vclk_read();\")" in t
open(sys.argv[2], "w").write(t)
