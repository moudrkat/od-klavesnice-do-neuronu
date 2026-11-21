#!/usr/bin/env python3
# Co doopravdy jde do modelu: všech 87 tokenů × 1 536 čísel (hidden_states[0] = řádky tabulky vnoření), stejné tokeny jako full_context.py.
# Uloží celou matici (zaokrouhleno), skupiny tokenů, id a prvních 8 čísel pro Write a return. Žádné generování.
import json, re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
dev="cuda" if torch.cuda.is_available() else "cpu"; I="Qwen/Qwen2.5-Coder-1.5B-Instruct"
Lk=json.load(open("orch_lock.json")); BSYS=Lk["builder_sys"]; USER=Lk["refiner_out"]; CODE0=Lk["builder_out"]
t=AutoTokenizer.from_pretrained(I); m=AutoModelForCausalLM.from_pretrained(I,dtype=(torch.bfloat16 if dev=="cuda" else torch.float32)).to(dev).eval()
tmpl=t.apply_chat_template([{"role":"system","content":BSYS},{"role":"user","content":USER}],tokenize=False,add_generation_prompt=True)
mm=re.search(r"```(?:python)?\s*(.*?)```",CODE0,re.S); code0=(mm.group(1) if mm else CODE0).strip()
full=tmpl+"```python\n"+code0+"```"; enc=t(full,return_offsets_mapping=True); ids=enc.input_ids; offs=enc.offset_mapping
us=full.index(USER); ue=us+len(USER); ss=full.index(BSYS); se=ss+len(BSYS); cs=full.index("def ")
grp=["první token" if i==0 else ("systémový prompt" if a>=ss and b<=se else ("tvoje zadání" if a>=us and b<=ue else ("jeho kód" if a>=cs else "šablona"))) for i,(a,b) in enumerate(offs)]
toks=[full[a:b] for a,b in offs]
with torch.no_grad(): hs=m(torch.tensor([ids]).to(dev),output_hidden_states=True).hidden_states
X0=hs[0][0].float().cpu(); X28=hs[-1][0].float().cpu()
E=m.model.embed_tokens.weight
ret=[i for i in range(len(toks)) if grp[i]=="jeho kód" and toks[i].strip()=="return"][0]
same=bool(torch.allclose(X0[ret],E[ids[ret]].float().cpu(),atol=1e-3))
out={"tokens":toks,"ids":ids,"group":grp,"ret_pos":ret,"shape":[len(toks),X0.shape[1]],"table_shape":list(E.shape),
     "x0":[[round(float(v),3) for v in row] for row in X0],"x28_return_first8":[round(float(v),2) for v in X28[ret][:8]],
     "x0_is_table_row":same,"write_first8":[round(float(v),3) for v in X0[[i for i in range(len(toks)) if toks[i]=="Write"][0]][:8]],"return_first8":[round(float(v),3) for v in X0[ret][:8]],
     "x0_absmax":round(float(X0.abs().max()),3),"x28_absmax":round(float(X28.abs().max()),2)}
json.dump(out,open("input_matrix.json","w"),ensure_ascii=False)
print("tokenů",len(toks),"| vstup",out["shape"],"| tabulka",out["table_shape"],"| vstupní řádek return == řádek tabulky:",same)
print("Write id",ids[[i for i in range(len(toks)) if toks[i]=="Write"][0]],out["write_first8"]); print("return id",ids[ret],out["return_first8"]); print("|x0| max",out["x0_absmax"],"| po vrstvě 28 return prvních 8:",out["x28_return_first8"])
print("INPUTDONE",flush=True)
