#!/usr/bin/env python3
# Kam jde pozornost z tokenů kódu: šablona+systémový prompt (bez prvního tokenu), první token (odkladiště), tvoje zadání, vlastní kód.
import json, re, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
dev="cuda" if torch.cuda.is_available() else "cpu"; I="Qwen/Qwen2.5-Coder-1.5B-Instruct"
Lk=json.load(open("orch_lock.json")); BSYS=Lk["builder_sys"]; USER=Lk["refiner_out"]; CODE0=Lk["builder_out"]
t=AutoTokenizer.from_pretrained(I); m=AutoModelForCausalLM.from_pretrained(I,dtype=(torch.bfloat16 if dev=="cuda" else torch.float32),attn_implementation="eager").to(dev).eval()
tmpl=t.apply_chat_template([{"role":"system","content":BSYS},{"role":"user","content":USER}],tokenize=False,add_generation_prompt=True)
mm=re.search(r"```(?:python)?\s*(.*?)```",CODE0,re.S); code0=(mm.group(1) if mm else CODE0).strip()
full=tmpl+"```python\n"+code0+"```"; enc=t(full,return_offsets_mapping=True); fids=torch.tensor([enc.input_ids]).to(dev); offs=enc.offset_mapping
us=full.index(USER); ue=us+len(USER); ss=full.index(BSYS); se=ss+len(BSYS); cs=full.index("def ")
T=fids.shape[1]; grp=np.array(["šablona"]*T,dtype=object)
for i,(a,b) in enumerate(offs):
    if i==0: grp[i]="první token"
    elif a>=ss and b<=se: grp[i]="systémový prompt"
    elif a>=us and b<=ue: grp[i]="tvoje zadání"
    elif a>=cs: grp[i]="jeho kód"
code=[i for i in range(T) if grp[i]=="jeho kód"]; ret=[i for i in code if full[offs[i][0]:offs[i][1]].strip()=="return"][0]
with torch.no_grad(): att=m(fids,output_attentions=True).attentions
G=["první token","šablona","systémový prompt","tvoje zadání","jeho kód"]; counts={g:int((grp==g).sum()) for g in G}
print("tokens:",counts,flush=True)
per_layer=[]; tot=np.zeros(len(G)); tot_ret=np.zeros(len(G))
for L in range(len(att)):
    A=att[L][0].float().cpu().numpy().mean(0)          # průměr přes hlavy [T,T]
    row=A[code].mean(0)                                 # průměr přes tokeny kódu
    v=[float(row[grp==g].sum()) for g in G]; per_layer.append([round(x,3) for x in v]); tot+=v
    tot_ret+=[float(A[ret][grp==g].sum()) for g in G]
tot/=len(att); tot_ret/=len(att)
print("TOTAL (mean over layers and heads), from the code tokens:",{g:round(float(x)*100,1) for g,x in zip(G,tot)},flush=True)
print("z pozice return:",{g:round(float(x)*100,1) for g,x in zip(G,tot_ret)},flush=True)
print("per token (total / tokens in the group):",{g:round(float(x)*100/counts[g],2) for g,x in zip(G,tot)},flush=True)
json.dump({"groups":G,"counts":counts,"total_from_code":[float(x) for x in tot],"from_return":[float(x) for x in tot_ret],"per_layer":per_layer,"system_prompt":BSYS},open("attn_split.json","w"),ensure_ascii=False); print("ATTNSPLITDONE",flush=True)
