#!/usr/bin/env python3
# Skóre místo pravděpodobností: na pozici `return` po každé vrstvě (norm + slovník = lens) top 3 slova s jejich skóre (logit). Výstup: lens_scores.json
import json, re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
dev="cuda" if torch.cuda.is_available() else "cpu"; I="Qwen/Qwen2.5-Coder-1.5B-Instruct"
Lk=json.load(open("orch_lock.json")); BSYS=Lk["builder_sys"]; USER=Lk["refiner_out"]; CODE0=Lk["builder_out"]
t=AutoTokenizer.from_pretrained(I); m=AutoModelForCausalLM.from_pretrained(I,dtype=(torch.bfloat16 if dev=="cuda" else torch.float32)).to(dev).eval()
tmpl=t.apply_chat_template([{"role":"system","content":BSYS},{"role":"user","content":USER}],tokenize=False,add_generation_prompt=True)
mm=re.search(r"```(?:python)?\s*(.*?)```",CODE0,re.S); code0=(mm.group(1) if mm else CODE0).strip()
pref=t(tmpl+"```python\n"+code0[:code0.index("return")+len("return")],return_tensors="pt").input_ids.to(dev)
with torch.no_grad(): hs=m(pref,output_hidden_states=True).hidden_states
W=m.lm_head.weight; norm=m.model.norm; out={}
with torch.no_grad():
    for L in range(1,len(hs)):   # hs[L] = po vrstvě L (hs[0] = vstup); hs[28] je už po final norm v HF, norm je idempotentní jen přibližně → použij vlastní normu na hs[L] pro L<28
        v=hs[L][0,-1]
        z=(norm(v) if L<len(hs)-1 else v)@W.T
        top=torch.topk(z.float(),3); out[str(L)]={"top":[(t.decode([int(i)]),round(float(s),2)) for s,i in zip(top.values,top.indices)]}
for L in (1,7,14,21,23,28): print(L,out[str(L)]["top"])
json.dump(out,open("lens_scores.json","w"),ensure_ascii=False); print("LENSSCOREDONE",flush=True)
