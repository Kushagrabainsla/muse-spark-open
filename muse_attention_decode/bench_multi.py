"""Multi-draw timing: CUDA kernel (forced, chunk per arg) vs the FA3 path, mean over seeds.
python bench_multi.py [file.py] [--seeds=5] [--shapes=full16,full32,...] [--chunks=1024,2048/64]
A chunk token is keys per CTA, optionally /stage keys/KV heads per CTA/warp-specialized (0 or 1); 0 keeps the file's settings."""
import importlib.util, statistics, sys
HERE = __file__.rsplit("/", 1)[0]
sys.path.insert(0, HERE)
import reference as ref  # noqa: E402
from timing import kernel_us  # noqa: E402

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

args = sys.argv[1:]
path = next((a for a in args if a.endswith(".py")), HERE + "/submission.py")
seeds = int(next((a.split("=")[1] for a in args if a.startswith("--seeds=")), 5))
mine = load(path, "mine")
fa3 = load(HERE + "/submission_a3.py", "fa3")
SH = {
    "slide1": dict(batch=1, layer="sliding", q_len=1), "slide16": dict(batch=16, layer="sliding", q_len=1),
    "slide32": dict(batch=32, layer="sliding", q_len=1), "full1": dict(batch=1, layer="full", q_len=1),
    "full16": dict(batch=16, layer="full", q_len=1), "full32": dict(batch=32, layer="full", q_len=1),
    "vslide1": dict(batch=1, layer="sliding", q_len=8), "vslide16": dict(batch=16, layer="sliding", q_len=8),
    "vslide32": dict(batch=32, layer="sliding", q_len=8), "vfull1": dict(batch=1, layer="full", q_len=8),
    "vfull16": dict(batch=16, layer="full", q_len=8), "vfull32": dict(batch=32, layer="full", q_len=8),
}
shapes = next((a.split("=")[1].split(",") for a in args if a.startswith("--shapes=")), list(SH))
chunks = next((a.split("=")[1] for a in args if a.startswith("--chunks=")), "0").split(",")
for name in shapes:
    kw = SH[name]
    tf, tc = [], {c: [] for c in chunks}
    for sd in range(seeds):
        d = ref.generate_input(ctx=80000, heads="tp8", page=1, seed=1000 + sd, spread="agentic" if kw["batch"] > 1 else "fixed",
                               dist="coherent" if kw["layer"] == "full" else "soft", **kw)
        tf.append(kernel_us(lambda x, s: fa3.kernel_fn(x), d, None))
        for c in chunks:
            parts = c.split("/")
            if int(parts[0]):
                mine.CHUNK[kw["layer"]] = int(parts[0])
            if len(parts) > 1:
                mine.STAGE_KEYS[kw["layer"]] = int(parts[1])
            if len(parts) > 2:
                mine.VERIFY_HEADS[kw["layer"]] = int(parts[2])
            if len(parts) > 3:
                mine.VERIFY_PC[kw["layer"]] = bool(int(parts[3]))
            tc[c].append(kernel_us(lambda x, s: mine.kernel_fn_cuda(x), d, None))
    line = f"{name:9s} fa3 {statistics.mean(tf):7.1f}"
    for c in chunks:
        line += f"  cuda[{c}] {statistics.mean(tc[c]):7.1f} x{statistics.mean(tf) / statistics.mean(tc[c]):.3f}"
    print(line, flush=True)
