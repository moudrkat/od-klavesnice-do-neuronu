#!/usr/bin/env python3
# Kam se z pozice `return` dívá vrstva 20 (attentions[20]): hlava 20·6 a průměr všech 12 hlav, po skupinách.
# Skupiny opraveně: „jeho kód“ začíná už na ```python (napsal ho builder sám), šablona končí na <|im_start|>assistant\n.
import json, re, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
dev="cuda" if torch.cuda.is_available() else "cpu"; I="Qwen/Qwen2.5-Coder-1.5B-Instruct"
Lk=json.load(open("orch_lock.json")); BSYS=Lk["builder_sys"]; USER=Lk["refiner_out"]; CODE0=Lk["builder_out"]
t=AutoTokenizer.from_pretrained(I); m=AutoModelForCausalLM.from_pretrained(I,dtype=(torch.bfloat16 if dev=="cuda" else torch.float32),attn_implementation="eager").to(dev).eval()
tmpl=t.apply_chat_template([{"role":"system","content":BSYS},{"role":"user","content":USER}],tokenize=False,add_generation_prompt=True)
mm=re.search(r"```(?:python)?\s*(.*?)```",CODE0,re.S); code0=(mm.group(1) if mm else CODE0).strip()
full=tmpl+"```python\n"+code0+"```"; enc=t(full,return_offsets_mapping=True); offs=enc.offset_mapping
us=full.index(USER); ue=us+len(USER); ss=full.index(BSYS); se=ss+len(BSYS)
def groups(cs):
    g=[]
    for i,(a,b) in enumerate(offs):
        if i==0: g.append("první token")
        elif a>=ss and b<=se: g.append("systémový prompt")
        elif a>=us and b<=ue: g.append("tvoje zadání")
        elif a>=cs: g.append("jeho kód")
        else: g.append("šablona")
    return np.array(g,dtype=object)
G=["první token","šablona","systémový prompt","tvoje zadání","jeho kód"]
old=groups(full.index("def ")); new=groups(len(tmpl))
toks=[full[a:b] for a,b in offs]; ret=[i for i in range(len(toks)) if toks[i].strip()=="return" and old[i]=="jeho kód"][0]
ids=enc.input_ids[:ret+1]
with torch.no_grad(): att=m(torch.tensor([ids]).to(dev),output_attentions=True).attentions
A=att[20][0].float().cpu().numpy()[:,ret,:]          # [12, ret+1] řádek return
out={"ret_pos":ret,"n_tokens":ret+1}
for name,gr in [("staré dělení",old),("opravené dělení",new)]:
    g=gr[:ret+1]
    out[name]={"počty":{k:int((g==k).sum()) for k in G},
               "hlava 20·6":{k:round(float(A[6][g==k].sum())*100,1) for k in G},
               "12 hlav průměr":{k:round(float(A.mean(0)[g==k].sum())*100,1) for k in G}}
out["kontrola: 20·6 na 'list'"]=round(float(A[6][[i for i in range(ret+1) if toks[i].strip()=="list"][0]]),3)
out["tokeny šablony 3 (opraveně)"]=[toks[i] for i in range(ret+1) if new[i]=="šablona"][-6:]
out["kód (opraveně)"]=[toks[i] for i in range(ret+1) if new[i]=="jeho kód"]
print(json.dumps(out,ensure_ascii=False,indent=1)); json.dump(out,open("attn_return20.json","w"),ensure_ascii=False,indent=1)
