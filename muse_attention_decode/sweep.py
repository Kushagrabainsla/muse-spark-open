"""Sweep TUNE per layer kind: python sweep.py sliding|full"""
import importlib.util, itertools, sys
HERE = __file__.rsplit("/", 1)[0]
sys.path.insert(0, HERE)
import reference as ref  # noqa: E402
from timing import kernel_us  # noqa: E402
spec = importlib.util.spec_from_file_location("mine", HERE + "/submission.py")
mine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mine)
kind = sys.argv[1]
cases = [dict(batch=b, ctx=80000, heads="tp8", layer=kind, page=1, q_len=L, dist="soft" if kind == "sliding" else "coherent",
              spread="agentic" if b > 1 else "fixed", seed=5) for b, L in ((1, 1), (16, 1), (32, 1), (1, 8), (16, 8), (32, 8))]
datas = [ref.generate_input(**c) for c in cases]
res = []
chs = (64, 128, 256, 512) if kind == "sliding" else (128, 256, 512, 1024)
for ch, bn, w, st in itertools.product(chs, (32, 64, 128), (4, 8), (1, 2, 3)):
    if bn > ch:
        continue
    mine.TUNE[kind] = (ch, bn, w, st)
    try:
        ts = [kernel_us(lambda d, s: mine.kernel_fn(d), d, None) for d in datas]
    except Exception as e:  # noqa: BLE001
        print("ERR", (ch, bn, w, st), str(e).splitlines()[0][:80], flush=True)
        continue
    res.append((sum(t * 1.0 for t in ts), (ch, bn, w, st), [round(t, 1) for t in ts]))
    print(res[-1], flush=True)
res.sort()
print("BEST", res[:5])
