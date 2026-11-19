#!/usr/bin/env python3
# Mřížka dávek pro interaktivní stránku: hlava 20·6 × neuron 20·3545, každé {0, 0.5, 1, 2, 4}; p po `return` (top 3) + kód. A přesné skóre pro „jak se z toho stane slovo“.
import json, re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
dev="cuda" if torch.cuda.is_available() else "cpu"; I="Qwen/Qwen2.5-Coder-1.5B-Instruct"; LZ=20; HD=6; NR=3545
Lk=json.load(open("orch_lock.json")); BSYS=Lk["builder_sys"]; USER=Lk["refiner_out"]; CODE0=Lk["builder_out"]
t=AutoTokenizer.from_pretrained(I); m=AutoModelForCausalLM.from_pretrained(I,dtype=(torch.bfloat16 if dev=="cuda" else torch.float32)).to(dev).eval()
H=m.config.num_attention_heads; D=m.config.hidden_size//H
tmpl=t.apply_chat_template([{"role":"system","content":BSYS},{"role":"user","content":USER}],tokenize=False,add_generation_prompt=True)
ids=t(tmpl,return_tensors="pt").input_ids.to(dev)
mm=re.search(r"```(?:python)?\s*(.*?)```",CODE0,re.S); code0=(mm.group(1) if mm else CODE0).strip()
pref=t(tmpl+"```python\n"+code0[:code0.index("return")+len("return")],return_tensors="pt").input_ids.to(dev)
def head_hook(f):
    def pre(mod,args):
        x=args[0].clone(); x[...,HD*D:(HD+1)*D]=x[...,HD*D:(HD+1)*D]*f; return (x,)
    return m.model.layers[LZ].self_attn.o_proj.register_forward_pre_hook(pre)
def neuron_hook(f):
    def pre(mod,args):
        x=args[0].clone(); x[...,NR]=x[...,NR]*f; return (x,)
    return m.model.layers[LZ].mlp.down_proj.register_forward_pre_hook(pre)
act={}
def cap(mod,args): act["n"]=float(args[0][0,-1,NR])
grid=[]
for hf in (0,0.5,1,2,4):
    for nf in (0,0.5,1,2,4):
        hs=([] if hf==1 else [head_hook(hf)])+([] if nf==1 else [neuron_hook(nf)])+[m.model.layers[LZ].mlp.down_proj.register_forward_pre_hook(cap)]
        with torch.no_grad():
            lg=m(pref).logits[0,-1].float(); p=torch.softmax(lg,-1); top=torch.topk(p,4)
            g=m.generate(ids,max_new_tokens=120,do_sample=False,pad_token_id=t.eos_token_id)
        for h in hs: h.remove()
        code=re.sub(r"```(?:python)?","",t.decode(g[0,ids.shape[1]:],skip_special_tokens=True)).strip()
        grid.append({"head":hf,"neuron":nf,"p":[(t.decode([int(i)]),round(float(v)*100,1)) for v,i in zip(top.values,top.indices)],"code":code,"neuron_act_at_return":round(act["n"],2)})
        print(f"h×{hf:<3g} n×{nf:<3g} → {grid[-1]['p'][0]}  {grid[-1]['p'][1]} | akt {act['n']:6.2f} | {code.replace(chr(10),' ⏎ ')[:90]}",flush=True)
with torch.no_grad(): out=m(pref,output_hidden_states=True)
lg=out.logits[0,-1].float(); p=torch.softmax(lg,-1); top=torch.topk(lg,6); hpost=out.hidden_states[-1][0,-1].float(); hpre=out.hidden_states[-2][0,-1].float()
nw={"h_after_layer28_8":[round(float(x),2) for x in hpre[:8]],"h_after_norm_8":[round(float(x),2) for x in hpost[:8]],"W_shape":list(m.lm_head.weight.shape),
    "top":[(t.decode([int(i)]),round(float(lg[i]),2),round(float(p[i])*100,1)) for i in top.indices],"logit_min":round(float(lg.min()),1),"logit_median":round(float(lg.median()),1)}
print("slovo:",nw["top"],"| min",nw["logit_min"],"medián",nw["logit_median"],flush=True)
json.dump({"grid":grid,"next_word":nw,"system_prompt":BSYS,"user":USER,"code":code0},open("play20_grid.json","w"),ensure_ascii=False); print("GRIDDONE",flush=True)
