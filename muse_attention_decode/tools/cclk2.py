import importlib.util, sys, torch, statistics
sys.path.insert(0, "/root/okc_attn")
import reference as ref
path, b = sys.argv[1], int(sys.argv[2])
spec = importlib.util.spec_from_file_location("mine", path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
if len(sys.argv) > 3: m.CHUNK["sliding"] = int(sys.argv[3])
if len(sys.argv) > 4: m.STAGE_KEYS["sliding"] = int(sys.argv[4])
d = ref.generate_input(batch=b, ctx=80000, heads="tp8", layer="sliding", page=1, seed=1000, spread="agentic" if b > 1 else "fixed", dist="soft")
flush = torch.ones(128 << 20, dtype=torch.float16, device="cuda")
for _ in range(3): m.kernel_fn_cuda(d)
torch.cuda.synchronize()
flush.sum(); m.kernel_fn_cuda(d); torch.cuda.synchronize()
c = m._module().clk_read().view(-1, 16)
ch = m._chunk(2048, b, 1); n = b * (2048 // ch)
c = c[:n].tolist()
names = ["prep", "loop", "merge", "sync1", "dsmem", "arr+bar", "-", "sum", "out"]
def ph(r):
    fl = r[7] % 8
    pre = r[3] if not fl & 2 else r[2]
    return [r[2] - r[1], (r[3] - r[2]) if not fl & 2 else 0, r[4] - pre, r[5] - r[4], r[8] - r[5], r[9] - r[8], r[10] - r[9], r[11] - r[10], r[6] - r[11]]
P = [ph(r) for r in c]
print(" ".join(f"{x:>7s}" for x in names))
print(" ".join(f"{statistics.median(p[k] for p in P):7.0f}" for k in range(len(names))), " (median)")
print("owner:", [p for p, r in zip(P, c) if r[7] % 8 & 1])
g0 = min(r[0] for r in c)
ends = sorted(r[0] - g0 + r[7] // 8 for r in c)
starts = sorted(r[0] - g0 for r in c)
print("start ns: min/med/max", starts[0], starts[len(starts)//2], starts[-1], " end ns: min/med/max", ends[0], ends[len(ends)//2], ends[-1])
