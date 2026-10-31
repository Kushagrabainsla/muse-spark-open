"""Debug variants of the MoE submission (M16): python mkdbg2.py <dbg> -> sub_dbg<dbg>.py beside it.
dbg & 1: skip the decode + scale + MMA math (keep the weight stream). dbg & 8: skip the weight / scale loads (math on
stale shared data). Timing only: the outputs are wrong."""
import sys
HERE = __file__.rsplit("/", 1)[0]
d = int(sys.argv[1])
s = open(HERE + "/submission.py").read()
s = s.replace('"muse_moe_nvfp4"', f'"muse_moe_nvfp4_dbg{d}"')
s = s.replace("constexpr int NW = 8;\n", f"constexpr int NW = 8;\nconstexpr int DBG = {d};\n", 1)
a = "#pragma unroll\n    for (int sub = 0; sub < SUB; ++sub) {\n    uint4 wv[RT][2];"
assert s.count(a) == 1
s = s.replace(a, "    if (DBG & 1) {\n      acc[0][0][0] += __uint_as_float(wring[stage * SUB * RT * 2 * 32 + lane].x ^ "
                 "sring[stage * SUB * RT * 2 * 8 + g].y);\n    } else\n" + a)
b = '''#pragma unroll
    for (int j = 0; j < SUB * RT * 2; ++j) {
      const int i = stage * SUB * RT * 2 + j;
      cp_async16(wring + i * 32 + lane, wq + (slot0 + j) * 32 + lane);
    }'''
assert s.count(b) == 1, "issue loop"
s = s.replace(b, "    if (!(DBG & 8)) {\n" + b + "\n    }")
c = '''    cp_async16(reinterpret_cast<uint4*>(sring + stage * SUB * RT * 2 * 8) + lane,
               reinterpret_cast<const uint4*>(wsf + slot0 * 8) + lane);'''
assert s.count(c) == 1, "scale copy"
s = s.replace(c, "    if (!(DBG & 8))\n" + c)
ra = "            red_add(acc_out + (size_t)p * out_stride + row0 + rt * 16 + (i >= 2 ? 8 : 0), acc[rt][mt][i] * gs);"
assert s.count(ra) == 1, "red_add"
s = s.replace(ra, "            if (!(DBG & 2)) " + ra.strip())
g1 = "    launch_gemm<MT_, RT_, 0, EX_>("
g2 = "    launch_gemm<MT_, RT_, 1, EX_>("
nm = "  cudaLaunchKernelEx(&cfg, norm_kernel,"
assert s.count(g1) == 1 and s.count(g2) == 1 and s.count(nm) == 1
s = s.replace(g1, "    if (!(DBG & 64)) launch_gemm<MT_, RT_, 0, EX_>(").replace(g2, "    if (!(DBG & 32)) launch_gemm<MT_, RT_, 1, EX_>(")
s = s.replace(nm, "  if (!(DBG & 16)) cudaLaunchKernelEx(&cfg, norm_kernel,")
open(f"{HERE}/sub_dbg{d}.py", "w").write(s)
print("wrote", f"{HERE}/sub_dbg{d}.py")
