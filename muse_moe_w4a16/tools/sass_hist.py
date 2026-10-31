"""Opcode histogram of one moe_gemm instantiation's SASS (the built extension), whole function and the hottest loop.
python sass_hist.py <so> <mangled-name-substring>"""
import collections
import re
import subprocess
import sys

so, want = sys.argv[1], sys.argv[2]
out = subprocess.run(["cuobjdump", "-sass", so], capture_output=True, text=True).stdout
funcs = re.split(r"\n\s+Function : ", out)
body = next(f for f in funcs if want in f.split("\n", 1)[0])
name = body.split("\n", 1)[0]
lines = [l for l in body.split("\n") if re.search(r"/\*[0-9a-f]{4,}\*/", l)]
ops = []
for l in lines:
    m = re.search(r"/\*([0-9a-f]+)\*/\s+(@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]+)", l)
    if m:
        ops.append((int(m.group(1), 16), m.group(3).split(".")[0], l))
print(name[:120])
print("instructions:", len(ops))
# Loops: backward branches (BRA to a lower address); report the largest-weight loop body.
loops = []
for addr, op, l in ops:
    if op == "BRA":
        t = re.search(r"BRA\s+(?:`\(\.L_x_\d+\)|0x([0-9a-f]+))", l)
        tgt = re.search(r"0x([0-9a-f]+)", l.split("BRA", 1)[1])
        if tgt and int(tgt.group(1), 16) < addr:
            loops.append((int(tgt.group(1), 16), addr))
for lo, hi in sorted(loops, key=lambda x: x[1] - x[0], reverse=True)[:3]:
    body_ops = [op for a, op, _ in ops if lo <= a <= hi]
    h = collections.Counter(body_ops)
    print(f"\nloop {lo:#x}-{hi:#x}: {len(body_ops)} instructions")
    print("  " + ", ".join(f"{k} {v}" for k, v in h.most_common(30)))
h = collections.Counter(op for _, op, _ in ops)
print("\nwhole function: " + ", ".join(f"{k} {v}" for k, v in h.most_common(30)))
