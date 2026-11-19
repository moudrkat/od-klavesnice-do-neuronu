#!/usr/bin/env python3
# "Tohle to dělá": aktivace vybraných neuronů po tokenech na hero promptu (zadání + kód). Jen dopředný průchod.
import json, re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
dev="cuda"; I="Qwen/Qwen2.5-Coder-1.5B-Instruct"
Lk=json.load(open("orch_lock.json")); BSYS=Lk["builder_sys"]; USER=Lk["refiner_out"]; CODE0=Lk["builder_out"]
t=AutoTokenizer.from_pretrained(I); m=AutoModelForCausalLM.from_pretrained(I,dtype=torch.bfloat16).to(dev).eval()
tmpl=t.apply_chat_template([{"role":"system","content":BSYS},{"role":"user","content":USER}],tokenize=False,add_generation_prompt=True)
mm=re.search(r"```(?:python)?\s*(.*?)```",CODE0,re.S); code0=(mm.group(1) if mm else CODE0).strip()
full=tmpl+"```python\n"+code0+"```"; enc=t(full,return_offsets_mapping=True); fids=torch.tensor([enc.input_ids]).to(dev); offs=enc.offset_mapping
us=full.index(USER); ue=us+len(USER); cs=full.index("def ")
user_toks=[i for i,(a,b) in enumerate(offs) if a>=us and b<=ue]; code_toks=[i for i,(a,b) in enumerate(offs) if a>=cs]
lab=lambda i: full[offs[i][0]:offs[i][1]]
NEUR={14:[2825,7845,8212],27:[8226,1661],22:[8350]}; box={}
hs=[m.model.layers[L].mlp.down_proj.register_forward_pre_hook(lambda mod,args,L=L: box.__setitem__(L,args[0][0].float().detach().cpu())) for L in NEUR]
with torch.no_grad(): m(fids)
for h in hs: h.remove()
out={"user":[lab(i) for i in user_toks],"code":[lab(i) for i in code_toks],"strips":{}}
for L,ns in NEUR.items():
    for n in ns:
        x=box[L][:,n]; out["strips"][f"{L}.{n}"]={"user":[round(float(x[i]),2) for i in user_toks],"code":[round(float(x[i]),2) for i in code_toks]}
        print(f"L{L}·{n}  zadání max {max(out['strips'][f'{L}.{n}']['user'],key=abs):.1f}  kód max {max(out['strips'][f'{L}.{n}']['code'],key=abs):.1f}",flush=True)
json.dump(out,open("token_strips.json","w"),ensure_ascii=False); print("STRIPSDONE",flush=True)
