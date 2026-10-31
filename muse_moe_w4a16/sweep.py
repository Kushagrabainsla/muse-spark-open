"""Sweep SPLITS at given bs: python sweep.py 128,256 's1,s2;...' (one config per process is safest)"""
import importlib.util
import sys
import torch
HERE = __file__.rsplit("/", 1)[0]
sys.path.insert(0, HERE)
import reference as ref  # noqa: E402
from timing import kernel_us  # noqa: E402

bss = [int(v) for v in sys.argv[1].split(",")]
w = ref.generate_weights(ref.WEIGHTS_SEED)
for c in sys.argv[2].split(";"):
    s1, s2, mt = (int(v) for v in c.split(","))
    spec = importlib.util.spec_from_file_location("mine", HERE + "/submission.py")
    mine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mine)
    mine.SPLITS = ((1 << 30, (s1, s2)),)
    mine.MT_BY_M = ((1 << 30, mt),)
    st = mine.prepare(w)
    line = f"s1={s1} s2={s2} mt={mt}:"
    for bs in bss:
        d = ref.generate_input(bs, "ep8", seed=400 + bs)
        ok, msg = ref.check_implementation(d, mine.kernel_fn(d, st), w)
        line += f"  bs{bs} {kernel_us(mine.kernel_fn, d, st):7.1f}{'' if ok else ' BAD'}"
    print(line, flush=True)
    del st
    torch.cuda.empty_cache()
