"""python icache.py file.py: per scored bs, the MoE call's time with a cold L2 (code and data) vs warm code and cold
data (the same kernels just ran on other weights and inputs, then nothing flushed in between). Graph replays."""
import importlib.util, statistics, sys, torch
sys.path.insert(0, "/root/okc_moe")
import reference as ref
path = sys.argv[1]
spec = importlib.util.spec_from_file_location("mine", path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
w1, w2 = ref.generate_weights(ref.WEIGHTS_SEED), ref.generate_weights(ref.WEIGHTS_SEED + 1)
s1, s2 = m.prepare(w1), m.prepare(w2)
flush = torch.ones(128 << 20, dtype=torch.float16, device="cuda")
REPS = 20
def cap(body):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s): body()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): body()
    return g
def tm(g):
    ts = []
    for _ in range(7):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(); a.record(); g.replay(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1000)
    return statistics.median(ts)
for bs in [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "1,16,32,8,128,256").split(",")]:
    A, B = ref.generate_input(bs, "ep8", seed=400 + bs), ref.generate_input(bs, "ep8", seed=900 + bs)
    gf = cap(lambda: [flush.sum() for _ in range(REPS)])
    gc = cap(lambda: [(flush.sum(), m.kernel_fn(A, s1)) for _ in range(REPS)])
    gp = cap(lambda: [(flush.sum(), m.kernel_fn(B, s2)) for _ in range(REPS)])
    gb = cap(lambda: [(flush.sum(), m.kernel_fn(B, s2), m.kernel_fn(A, s1)) for _ in range(REPS)])
    r = {k: [] for k in "fcpb"}
    for _ in range(3):
        for k, g in zip("fcpb", (gf, gc, gp, gb)): r[k].append(tm(g))
    f, c, p, b2 = (statistics.median(r[k]) for k in "fcpb")
    cold, warm = (c - f) / REPS, (b2 - p) / REPS
    print(f"bs {bs:3d}: cold {cold:6.1f} us   warm code, cold data {warm:6.1f} us   code share {100 * (cold - warm) / cold:4.1f}%", flush=True)
