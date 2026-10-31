"""python mkcclk2.py in.py out.py: attnc with per-CTA phase clocks (16 slots per CTA in g_clk, read by clk_read())."""
import sys
src, dst = sys.argv[1], sys.argv[2]
t = open(src).read()
k0 = t.index("__global__ void __launch_bounds__(128) attnc_kernel(")
head, body = t[:k0], t[k0:]
def rep(a, b):
    global body
    assert body.count(a) >= 1, a
    body = body.replace(a, b, 1)
rep('  cooperative_groups::cluster_group cl = cooperative_groups::this_cluster();\n',
    '  cooperative_groups::cluster_group cl = cooperative_groups::this_cluster();\n  long long* CK = g_clk + (blockIdx.y * gridDim.x + blockIdx.x) * 16;\n  if (threadIdx.x == 0) { CK[0] = gtime(); CK[1] = clock64(); }\n')
rep('''  __syncthreads();

  if (!empty) {
    uint32_t qa''', '''  __syncthreads();
  if (threadIdx.x == 0) { CK[2] = clock64(); CK[7] = owner + 2 * empty; }

  if (!empty) {
    uint32_t qa''')
rep('''    cp_wait<0>();
    la += __shfl_xor_sync''', '''    cp_wait<0>();
    if (threadIdx.x == 0) CK[3] = clock64();
    la += __shfl_xor_sync''')
rep('''  cl.sync();  // every chunk's partials have landed
''', '''  if (threadIdx.x == 0) CK[4] = clock64();
  cl.sync();  // every chunk's partials have landed
  if (threadIdx.x == 0) { CK[5] = clock64(); CK[8] = CK[9] = CK[10] = CK[11] = CK[5]; }
''')
i = body.index("void attnc(torch::Tensor qkv")
j = body.rindex("}", 0, i)
body = body[:j] + '''  if (threadIdx.x == 0) { CK[6] = clock64(); CK[7] += 8 * (gtime() - CK[0]); }
''' + body[j:]
i = head.rindex("template <int STG, int SK>\n")
head = head[:i] + ('__device__ long long g_clk[4096 * 16];\n__device__ __forceinline__ long long gtime() { long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }\n'
    'torch::Tensor clk_read() { auto t = torch::empty({4096 * 16}, torch::dtype(torch::kInt64)); cudaMemcpyFromSymbol(t.data_ptr(), g_clk, sizeof(g_clk)); return t; }\n') + head[i:]
t = head + body
t = t.replace('"int64_t window, int64_t CH, int64_t NCH, int64_t batch);")\n_mod', '"int64_t window, int64_t CH, int64_t NCH, int64_t batch);\\ntorch::Tensor clk_read();")\n_mod')
t = t.replace('"attnc", "attnpcc"]', '"attnc", "attnpcc", "clk_read"]')
t = t.replace('load_inline("muse_attn_decode_f"', 'load_inline("muse_attn_decode_cclk"')
assert "clk_read();\")" in t and '"clk_read"]' in t
open(dst, "w").write(t)
