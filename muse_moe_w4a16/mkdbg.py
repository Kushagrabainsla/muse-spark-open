"""Write debug variants of submission.py: python mkdbg.py <dbg> -> sub_dbg<dbg>.py
dbg & 1: skip the dequant + MMA math (keep loads). dbg & 2: skip the atomics. dbg & 4: skip x staging loads. dbg & 8: skip the weight loads (math on stale data)."""
import sys
HERE = __file__.rsplit("/", 1)[0]
d = int(sys.argv[1])
s = open(HERE + "/submission.py").read()
s = s.replace('_mod = load_inline("muse_moe_nvfp4"', f'_mod = load_inline("muse_moe_nvfp4_dbg{d}"')
s = s.replace("constexpr int NW = 8;", f"constexpr int NW = 8;\nconstexpr int DBG = {d};", 1)
a = "    const int kl = st * 128 + 4 * q;\n#pragma unroll\n    for (int rt = 0; rt < RT; ++rt) {"
assert a in s
s = s.replace(a, "    const int kl = st * 128 + 4 * q;\n    if (DBG & 1) {\n      acc[0][0][0] += __uint_as_float(wv[0][0].x ^ wv[RT - 1][1].w ^ sv[0][0].x ^ sv[RT - 1][1].y);\n    } else\n#pragma unroll\n    for (int rt = 0; rt < RT; ++rt) {")
a = "    if (st == nst - 1) {\n      const float gs"
assert a in s
s = s.replace(a, "    if (st == nst - 1 && !(DBG & 2)) {\n      const float gs")
a = "          if (p >= 0) {\n            __nv_bfloat162* hv"
assert a in s
s = s.replace(a, "          if (p >= 0 && !(DBG & 4)) {\n            __nv_bfloat162* hv")
a = "        cp_async16(wring + i, wq + (slot0 + rt * 2 + h) * 32 + lane);\n        cp_async8(sring + i, wsf + (slot0 + rt * 2 + h) * 8 + g);"
assert a in s
s = s.replace(a, "        if (!(DBG & 8)) {\n" + a + "\n        }")
open(HERE + f"/sub_dbg{d}.py", "w").write(s)
