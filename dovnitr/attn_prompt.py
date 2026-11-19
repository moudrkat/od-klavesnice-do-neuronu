#!/usr/bin/env python3
# Kam se model dívá do ZADÁNÍ, když píše kód? Attention z tokenů vygenerovaného kódu na tokeny uživatelského promptu.
# Stejný kontext jako attn_map.py (orch_lock.json). Jen dopředný průchod.
import json, re, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
dev="cuda" if torch.cuda.is_available() else "cpu"; I="Qwen/Qwen2.5-Coder-1.5B-Instruct"
Lk=json.load(open("orch_lock.json")); BSYS=Lk["builder_sys"]; code=Lk["builder_out"]; USER=Lk["refiner_out"]
mm=re.search(r"```(?:python)?\s*(.*?)```",code,re.S); CODE=(mm.group(1) if mm else code).strip()
t=AutoTokenizer.from_pretrained(I); m=AutoModelForCausalLM.from_pretrained(I,dtype=(torch.bfloat16 if dev=="cuda" else torch.float32),attn_implementation="eager").to(dev).eval()
tmpl=t.apply_chat_template([{"role":"system","content":BSYS},{"role":"user","content":USER}],tokenize=False,add_generation_prompt=True)
full=tmpl+"```python\n"+CODE+"```"
enc=t(full,return_offsets_mapping=True); ids=torch.tensor([enc.input_ids]).to(dev); offs=enc.offset_mapping
us=full.index(USER); ue=us+len(USER); cs=full.index("def ")
user_toks=[i for i,(a,b) in enumerate(offs) if a>=us and b<=ue]; code_toks=[i for i,(a,b) in enumerate(offs) if a>=cs]
with torch.no_grad(): att=m(ids,output_attentions=True).attentions
NL=m.config.num_hidden_layers; H=m.config.num_attention_heads
lab=lambda i: full[offs[i][0]:offs[i][1]].replace("\n","\\n")
# 1) hlava s nejsilnější pozorností kód→zadání (průměr přes kódové tokeny z celkové hmoty na zadání)
rows=[]
for L in range(NL):
    for h in range(H):
        A=att[L][0,h].float().cpu().numpy(); sub=A[np.ix_(code_toks,user_toks)]
        rows.append((float(sub.sum(1).mean()),float(sub.max()),L,h))
rows.sort(reverse=True)
print("hlavy s největší hmotou pozornosti z kódu do zadání (podíl, max, L, h):",[(round(a,2),round(b,2),L,h) for a,b,L,h in rows[:8]],flush=True)
out={"user_labels":[lab(i) for i in user_toks],"code_labels":[lab(i) for i in code_toks],"heads":[]}
for a,b,L,h in rows[:3]:
    A=att[L][0,h].float().cpu().numpy(); sub=A[np.ix_(code_toks,user_toks)]
    out["heads"].append({"head":[L,h],"mass":round(a,3),"matrix":[[round(float(v),3) for v in r] for r in sub]})
    print(f"=== L{L}·h{h} (hmota na zadání {a:.2f})",flush=True)
    for ci,i in enumerate(code_toks):
        j=int(sub[ci].argmax()); print(f"   {lab(i)!r:16s} -> {lab(user_toks[j])!r:14s} {sub[ci][j]:.2f}",flush=True)
# 2) průměr přes všechny hlavy a vrstvy: na které slovo zadání se kód dívá nejvíc
tot=np.zeros((len(code_toks),len(user_toks)))
for L in range(NL):
    for h in range(H): tot+=att[L][0,h].float().cpu().numpy()[np.ix_(code_toks,user_toks)]
tot/= (NL*H); colsum=tot.sum(0)
order=np.argsort(-colsum); print("průměr všech hlav, slova zadání podle přijaté pozornosti:",[(lab(user_toks[j]),round(float(colsum[j]),3)) for j in order[:10]],flush=True)
out["mean_all_heads"]={"matrix":[[round(float(v),4) for v in r] for r in tot],"col_rank":[[lab(user_toks[j]),round(float(colsum[j]),3)] for j in order]}
json.dump(out,open("attn_prompt.json","w"),ensure_ascii=False); print("ATTNPROMPTDONE",flush=True)
