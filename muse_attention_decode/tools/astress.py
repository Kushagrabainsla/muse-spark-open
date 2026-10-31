"""python astress.py file.py iters: graph replays of kernel_fn vs its eager output (bitwise) on sliding/full decode
shapes and a few tests, a cold-L2 flush between replays."""
import importlib.util, sys, torch
sys.path.insert(0, "/root/okc_attn")
import reference as ref
path, iters = sys.argv[1], int(sys.argv[2])
spec = importlib.util.spec_from_file_location("mine", path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
flush = torch.ones(128 << 20, dtype=torch.float16, device="cuda")
cases = [dict(batch=b, ctx=80000, heads="tp8", layer="sliding", page=1, dist="soft", spread="agentic" if b > 1 else "fixed") for b in (1, 4, 16, 32)]
cases += [dict(batch=1, ctx=10, heads="tp8", layer="sliding"), dict(batch=3, ctx=2047, heads="tp8", layer="sliding"),
          dict(batch=4, ctx=6000, heads="tp8", layer="sliding", dist="edges", page=1), dict(batch=5, ctx=3000, heads="dp", layer="sliding", spread="mixed")]
cases += [dict(batch=b, ctx=80000, heads="tp8", layer="sliding", page=1, q_len=8, dist="soft", spread="agentic" if b > 1 else "fixed") for b in (1, 4, 16)]
cases += [dict(batch=5, ctx=5000, heads="tp8", layer="sliding", q_len=8, spread="mixed", page=1),
          dict(batch=3, ctx=4000, heads="tp8", layer="sliding", q_len=8, page=1, kv_scale=0.015625),
          dict(batch=4, ctx=6000, heads="tp8", layer="sliding", q_len=8, dist="soft", page=1),
          dict(batch=2, ctx=1500, heads="dp", layer="sliding", q_len=8, page=64)]
if len(sys.argv) > 3: cases = cases[8:]
bad = 0
for kw in cases:
    d = ref.generate_input(seed=7, **kw)
    ok0, msg = ref.check_implementation(d, ref.collect(d, m.kernel_fn(d)))
    want = m.kernel_fn(d).clone()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        m.kernel_fn(d)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = m.kernel_fn(d)
    nbad = 0
    for i in range(iters):
        flush.sum()
        g.replay()
        if not torch.equal(out, want):
            nbad += 1
    torch.cuda.synchronize()
    bad += nbad + (not ok0)
    print(f"{kw}: check {'ok' if ok0 else 'FAIL ' + msg}, replay mismatches {nbad}/{iters}", flush=True)
print("total bad:", bad)
