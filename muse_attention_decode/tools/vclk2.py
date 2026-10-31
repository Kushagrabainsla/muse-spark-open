import importlib.util, sys, torch, statistics
sys.path.insert(0, "/root/okc_attn")
import reference as ref
path, b = sys.argv[1], int(sys.argv[2])
spec = importlib.util.spec_from_file_location("mine", path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
if len(sys.argv) > 3: m.CHUNK["sliding"] = int(sys.argv[3])
d = ref.generate_input(batch=b, ctx=80000, heads="tp8", layer="sliding", page=1, q_len=8, seed=1000, spread="agentic" if b > 1 else "fixed", dist="soft")
flush = torch.ones(128 << 20, dtype=torch.float16, device="cuda")
for _ in range(3): m.kernel_fn_cuda(d)
torch.cuda.synchronize()
flush.sum(); m.kernel_fn_cuda(d); m.kernel_fn_cuda(d) if len(sys.argv) > 4 else None; torch.cuda.synchronize()
ch = m._vchunk(b); nch = (2055 + ch - 1) // ch; n = b * 2 * nch
c = m._module().vclk_read().view(-1, 16)[:n].tolist()
names = ["prep", "P1-P2", "loop", "-", "push", "sync1", "merge", "total ns", "steps"]
P = []
for r in c:
    if r[9] == 0: continue
    P.append([r[2] - r[1], r[3] - r[2], r[4] - r[3], r[5] - r[4], r[6] - r[5], r[7] - r[6], r[8] - r[7], r[10], r[9]])
print(" ".join(f"{x:>11s}" for x in names))
for k in ("min", "median", "max"):
    f = {"min": min, "median": statistics.median, "max": max}[k]
    print(" ".join(f"{f(p[i] for p in P):11.0f}" for i in range(len(names))), k)
starts = [r[0] for r in c]; g0 = min(starts)
print("end ns max:", max(r[0] - g0 + r[10] for r in c), " start spread ns:", max(starts) - g0)
