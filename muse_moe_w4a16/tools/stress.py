"""Graph-replay stress: per bs, capture flush + kernel_fn on static inputs, then for many seeds copy new
inputs in, replay, and check against the reference. python stress.py <file> <iters>"""
import importlib.util, sys, torch
sys.path.insert(0, "/root/okc_moe")
import reference as ref
path, iters = sys.argv[1], int(sys.argv[2])
spec = importlib.util.spec_from_file_location("mine", path)
mine = importlib.util.module_from_spec(spec); spec.loader.exec_module(mine)
w = ref.generate_weights(ref.WEIGHTS_SEED); st = mine.prepare(w)
flush = torch.ones(64 << 20, dtype=torch.float16, device="cuda")
bad = 0
for bs, dist in ((1, "normal"), (8, "normal"), (16, "skewed"), (32, "normal"), (128, "skewed"), (257, "normal")):
    d = ref.generate_input(bs, "ep8", seed=1, **({"dist": dist} if dist != "normal" else {}))
    static = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in d.items()}
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        mine.kernel_fn(static, st)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        flush.sum()
        out = mine.kernel_fn(static, st)
        flush.sum()
        out2 = mine.kernel_fn(static, st)
    for i in range(iters):
        di = ref.generate_input(bs, "ep8", seed=1000 + i, **({"dist": dist} if dist != "normal" else {}))
        for k, v in di.items():
            if torch.is_tensor(v):
                static[k].copy_(v)
        g.replay(); torch.cuda.synchronize()
        for o in (out, out2):
            ok, msg = ref.check_implementation(di, o, w)
            if not ok:
                bad += 1
                print("FAIL", bs, dist, i, msg, flush=True)
    print(f"bs={bs} {dist}: {iters} replays done", flush=True)
print("failures:", bad)
