"""nsys helper: python prof.py <layer> <batch> <q_len> [file.py]"""
import importlib.util, sys
import torch
HERE = __file__.rsplit("/", 1)[0]
sys.path.insert(0, HERE)
import reference as ref  # noqa: E402
layer, b, L = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
path = sys.argv[4] if len(sys.argv) > 4 else HERE + "/submission.py"
spec = importlib.util.spec_from_file_location("mine", path)
mine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mine)
d = ref.generate_input(batch=b, ctx=80000, heads="tp8", layer=layer, page=1, q_len=L, seed=5,
                       spread="agentic" if b > 1 else "fixed", dist="soft")
for _ in range(3):
    mine.kernel_fn(d)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
for _ in range(10):
    mine.kernel_fn(d)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
