"""python prof2.py file.py layer batch L chunk sk [dist]: nsys target, forced CUDA path"""
import importlib.util, sys, torch
sys.path.insert(0, "/root/okc_attn")
import reference as ref
path, layer, b, L, ch, sk = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
dist = sys.argv[7] if len(sys.argv) > 7 else ("coherent" if layer == "full" else "soft")
spec = importlib.util.spec_from_file_location("mine", path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.CHUNK[layer] = ch
if hasattr(m, "STAGE_KEYS"): m.STAGE_KEYS[layer] = sk
if len(sys.argv) > 8 and hasattr(m, "VERIFY_HEADS"): m.VERIFY_HEADS[layer] = int(sys.argv[8])
if len(sys.argv) > 9 and hasattr(m, "VERIFY_PC"): m.VERIFY_PC[layer] = bool(int(sys.argv[9]))
d = ref.generate_input(batch=b, ctx=80000, heads="tp8", layer=layer, page=1, q_len=L, seed=1000,
                       spread="agentic" if b > 1 else "fixed", dist=dist)
for _ in range(3): m.kernel_fn_cuda(d)
torch.cuda.synchronize(); torch.cuda.cudart().cudaProfilerStart()
for _ in range(10): m.kernel_fn_cuda(d)
torch.cuda.synchronize(); torch.cuda.cudart().cudaProfilerStop()
