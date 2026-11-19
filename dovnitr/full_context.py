#!/usr/bin/env python3
# Celý vstup (všech tokenů): skupina každého tokenu; pozornost z pozice `return` (průměr hlav+vrstev, a hlava 20·6);
# aktivace neuronů 20·3545 a 27·8226 na každém tokenu. Jen dopředný průchod.
import json, re, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
dev="cuda"; I="Qwen/Qwen2.5-Coder-1.5B-Instruct"
Lk=json.load(open("orch_lock.json")); BSYS=Lk["builder_sys"]; USER=Lk["refiner_out"]; CODE0=Lk["builder_out"]
t=AutoTokenizer.from_pretrained(I); m=AutoModelForCausalLM.from_pretrained(I,dtype=torch.bfloat16,attn_implementation="eager").to(dev).eval()
tmpl=t.apply_chat_template([{"role":"system","content":BSYS},{"role":"user","content":USER}],tokenize=False,add_generation_prompt=True)
mm=re.search(r"```(?:python)?\s*(.*?)```",CODE0,re.S); code0=(mm.group(1) if mm else CODE0).strip()
full=tmpl+"```python\n"+code0+"```"; enc=t(full,return_offsets_mapping=True); fids=torch.tensor([enc.input_ids]).to(dev); offs=enc.offset_mapping
us=full.index(USER); ue=us+len(USER); ss=full.index(BSYS); se=ss+len(BSYS); cs=full.index("def ")
T=fids.shape[1]; grp=[]
for i,(a,b) in enumerate(offs):
    grp.append("první token" if i==0 else ("systémový prompt" if a>=ss and b<=se else ("tvoje zadání" if a>=us and b<=ue else ("jeho kód" if a>=cs else "šablona"))))
toks=[full[a:b] for a,b in offs]; ret=[i for i in range(T) if grp[i]=="jeho kód" and toks[i].strip()=="return"][0]
box={}
hs=[m.model.layers[L].mlp.down_proj.register_forward_pre_hook(lambda mod,args,L=L: box.__setitem__(L,args[0][0].float().detach().cpu())) for L in (20,27)]
with torch.no_grad(): out=m(fids,output_attentions=True)
for h in hs: h.remove()
att=out.attentions; allA=torch.stack([a[0].float().cpu() for a in att])   # [L,H,T,T]
from_ret_mean=allA[:,:,ret,:].mean((0,1)).numpy(); from_ret_h206=allA[20,6,ret,:].numpy()
res={"tokens":toks,"group":grp,"ret_pos":ret,"from_return_mean":[round(float(x),4) for x in from_ret_mean],"from_return_h20_6":[round(float(x),4) for x in from_ret_h206],
     "n20_3545":[round(float(x),2) for x in box[20][:,3545]],"n27_8226":[round(float(x),2) for x in box[27][:,8226]],"system_prompt":BSYS,"user":USER}
json.dump(res,open("full_context.json","w"),ensure_ascii=False)
print("tokenů:",T,"| return na pozici",ret)
for g in ("první token","šablona","systémový prompt","tvoje zadání","jeho kód"):
    idx=[i for i in range(T) if grp[i]==g]; print(f"{g:18s} {len(idx):3d} tokenů | pozornost z return: průměr {100*from_ret_mean[idx].sum():5.1f} %  hlava 20·6 {100*from_ret_h206[idx].sum():5.1f} %")
print("hlava 20·6 z return, top 6:",sorted([(round(float(from_ret_h206[i]),2),toks[i],grp[i]) for i in range(T)],reverse=True)[:6])
print("20·3545 největší:",sorted([(round(float(box[20][i,3545]),1),toks[i],grp[i]) for i in range(T)],key=lambda p:-abs(p[0]))[:6])
print("27·8226 největší:",sorted([(round(float(box[27][i,8226]),1),toks[i],grp[i]) for i in range(T)],key=lambda p:-abs(p[0]))[:6])
print("FULLDONE")
