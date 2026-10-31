"""Sweep submission_cuda.py's (RT, NSTEPS, S) per shape: python sweep_cuda.py [--m=1,16,32] [--layer=o]"""
import importlib.util
import itertools
import sys

HERE = __file__.rsplit("/", 1)[0]
sys.path.insert(0, HERE)
import reference as ref  # noqa: E402
from timing import kernel_us  # noqa: E402

spec = importlib.util.spec_from_file_location("mine", HERE + "/submission_cuda.py")
mine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mine)
args = sys.argv[1:]
ms = [int(v) for v in next((a.split("=", 1)[1] for a in args if a.startswith("--m=")), "1,16,32").split(",")]
layers = next((a.split("=", 1)[1] for a in args if a.startswith("--layer=")), "qkv_gate,o,pooler,unpooler,router").split(",")
weights = ref.generate_weights(ref.WEIGHTS_SEED)
orig = mine._config
state = mine.prepare(weights)
for m, layer in itertools.product(ms, layers):
    data = ref.generate_input(m, layer, seed=7)
    n, k = data["config"]["n"], data["config"]["k"]
    lo = layer == "router"
    mt = orig(n, k, m, lo)[0]
    res = []
    for rt, ns, ch in itertools.product((1, 2, 4), (2, 4, 8), (1, 2, 4, 8)):
        s = k // (128 * ns * ch)
        if s < 1 or s > 16 or k % (128 * ns * ch) or n % (16 * rt) or (lo and (rt > 2 or ns > 4)):
            continue
        cfg = (mt, rt, ns, ch, s)
        mine._config = lambda *_a, c=cfg: c
        try:
            ok, msg = ref.check_implementation(data, mine.kernel_fn(data, state), weights)
            if not ok:
                print("BAD", cfg, msg, flush=True)
                continue
            res.append((kernel_us(mine.kernel_fn, data, state), cfg, (n // (16 * rt)) * s))
        except Exception as e:  # noqa: BLE001
            print("ERR", cfg, str(e).splitlines()[0][:100], flush=True)
    res.sort()
    floor = ref.speed_of_light_ns(m, layer, {"hbm_bytes_per_s": 4.8e12, "bf16_dense_flops_per_s": 989e12}) / 1e3
    print(f"== m={m} {layer} floor {floor:.1f}: " + "  ".join(f"{t:.2f}us {c[1:]} ctas={nc}" for t, c, nc in res[:4]), flush=True)
