#!/usr/bin/env python3
# Teplota na drobkovi: stejný prompt (orch_lock.json), rozdělení dalšího tokenu po `return` při T = 0.1 … 2.0 (top 5 + entropie v bitech),
# a celý kód vzorkovaný při T = 0.7 / 1.0 / 1.5 / 2.0, 5 vzorků na teplotu, seed 0..4. Výstup: temperature.json
import json, re, math, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
dev="cuda" if torch.cuda.is_available() else "cpu"; I="Qwen/Qwen2.5-Coder-1.5B-Instruct"
Lk=json.load(open("orch_lock.json")); BSYS=Lk["builder_sys"]; USER=Lk["refiner_out"]; CODE0=Lk["builder_out"]
t=AutoTokenizer.from_pretrained(I); m=AutoModelForCausalLM.from_pretrained(I,dtype=(torch.bfloat16 if dev=="cuda" else torch.float32)).to(dev).eval()
tmpl=t.apply_chat_template([{"role":"system","content":BSYS},{"role":"user","content":USER}],tokenize=False,add_generation_prompt=True)
ids=t(tmpl,return_tensors="pt").input_ids.to(dev)
mm=re.search(r"```(?:python)?\s*(.*?)```",CODE0,re.S); code0=(mm.group(1) if mm else CODE0).strip()
pref=t(tmpl+"```python\n"+code0[:code0.index("return")+len("return")],return_tensors="pt").input_ids.to(dev)
with torch.no_grad(): z=m(pref).logits[0,-1].float()
out={"next":{}, "samples":{}}
for T in (0.1,0.5,0.7,1.0,1.5,2.0):
    p=torch.softmax(z/T,-1); H=float(-(p[p>0]*torch.log2(p[p>0])).sum()); top=torch.topk(p,15)
    out["next"][str(T)]={"entropy_bits":round(H,2),"top":[(t.decode([int(i)]),round(float(v)*100,1)) for v,i in zip(top.values,top.indices)]}
    print(f"T={T}: entropie {H:.2f} bitů, top: {out['next'][str(T)]['top'][:3]}",flush=True)
for T in (0.7,1.0,1.5,2.0):
    out["samples"][str(T)]=[]
    for seed in range(5):
        torch.manual_seed(seed)
        with torch.no_grad(): g=m.generate(ids,max_new_tokens=120,do_sample=True,temperature=T,top_k=0,top_p=1.0,pad_token_id=t.eos_token_id)
        code=t.decode(g[0,ids.shape[1]:],skip_special_tokens=True).strip(); out["samples"][str(T)].append(code)
        body=re.sub(r"```(?:python)?","",code).strip().replace("\n"," ⏎ ")
        print(f"T={T} seed {seed}: {body[:150]}",flush=True)
json.dump(out,open("temperature.json","w"),ensure_ascii=False,indent=1); print("TEMPDONE",flush=True)
