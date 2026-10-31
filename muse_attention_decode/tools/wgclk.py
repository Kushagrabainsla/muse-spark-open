import importlib.util, sys, torch
sys.path.insert(0, "/root/okc_attn")
import reference as ref
spec = importlib.util.spec_from_file_location("m", "/root/okc_attn/sub_wgclk.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
for layer, b, ch, heads in (("full", 16, 1024, 1), ("full", 16, 2048, 2), ("sliding", 32, 512, 1), ("sliding", 1, 64, 1)):
    m.CHUNK[layer] = ch; m.VERIFY_HEADS[layer] = heads
    d = ref.generate_input(batch=b, ctx=80000, heads="tp8", layer=layer, page=1, q_len=8, seed=1000, spread="agentic" if b > 1 else "fixed", dist="coherent" if layer == "full" else "soft")
    m.kernel_fn_cuda(d); torch.cuda.synchronize(); m.DBG.zero_()
    m.kernel_fn_cuda(d); torch.cuda.synchronize()
    t = m.DBG.tolist(); n = max(t[9], 1); ncta = max(t[10], 1)
    print(f"{layer} b{b} ch{ch} heads{heads}: CTAs {ncta} stages/CTA {n/ncta:.1f} | per stage: barrier+wait {t[0]/n:.0f} issue+S+transpose {t[1]/n:.0f} softmax {t[3]/n:.0f} PVissue {t[4]/n:.0f} | per CTA: prologue {t[6]/ncta:.0f} final wait {t[2]/ncta:.0f} epilogue {t[7]/ncta:.0f} lifetime {t[8]/ncta:.0f}", flush=True)
