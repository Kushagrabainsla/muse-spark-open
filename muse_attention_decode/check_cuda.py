"""Correctness of the forced CUDA path (kernel_fn_cuda) on every test and scored shape it can run, plus extras
(batch 1-33, page 1/64, scaled caches), at the given stage sizes: python check_cuda.py [file.py] [--sk=32,64]"""
import importlib.util, sys
HERE = __file__.rsplit("/", 1)[0]
sys.path.insert(0, HERE)
import reference as ref  # noqa: E402
from bench import SCORED, TESTS  # noqa: E402

args = sys.argv[1:]
path = next((a for a in args if a.endswith(".py")), HERE + "/submission.py")
spec = importlib.util.spec_from_file_location("mine", path)
mine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mine)
sks = [int(x) for x in next((a.split("=")[1] for a in args if a.startswith("--sk=")), "0").split(",")]
EXTRA = []
for b in (1, 2, 5, 17, 33):
    for layer in ("sliding", "full"):
        for L in (1, 8):
            EXTRA.append(dict(batch=b, ctx=3000 + 977 * b, heads="tp8", layer=layer, q_len=L, page=1 if b % 2 else 64,
                              spread="mixed" if b > 1 else "fixed", seed=500 + b))
EXTRA += [dict(batch=3, ctx=70, heads="tp8", layer="full", q_len=8, page=1, kv_scale=0.015625),
          dict(batch=4, ctx=2100, heads="tp8", layer="sliding", q_len=8, page=1, kv_scale=0.015625)]
bad = n = 0
for sk in sks:
    if sk:
        mine.STAGE_KEYS.update(sliding=sk, full=sk)
    for kw in TESTS + [k for _, _, k in SCORED] + EXTRA:
        kw = dict(kw)
        kw.setdefault("seed", 77)
        data = ref.generate_input(**kw)
        cfg = data["config"]
        if cfg["num_q_heads"] // cfg["num_kv_heads"] != 8 or cfg["num_kv_heads"] % 2:
            continue
        n += 1
        ok, msg = ref.check_implementation(data, ref.collect(data, mine.kernel_fn_cuda(data)))
        if not ok:
            bad += 1
            print(f"FAIL sk={sk} {kw}: {msg}", flush=True)
print(f"cuda checks done: {n} cases, {bad} failures", flush=True)
