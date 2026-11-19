#!/usr/bin/env python3
# Doslova: jak se čísla v proudu mění patro po patře na hero promptu, co z nich vyjde nahoře, a jak se to opakuje (další tokeny).
import json, re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
dev="cuda" if torch.cuda.is_available() else "cpu"; I="Qwen/Qwen2.5-Coder-1.5B-Instruct"
Lk=json.load(open("orch_lock.json")); BSYS=Lk["builder_sys"]; USER=Lk["refiner_out"]; CODE0=Lk["builder_out"]
t=AutoTokenizer.from_pretrained(I); m=AutoModelForCausalLM.from_pretrained(I,dtype=(torch.bfloat16 if dev=="cuda" else torch.float32)).to(dev).eval(); NL=len(m.model.layers)
tmpl=t.apply_chat_template([{"role":"system","content":BSYS},{"role":"user","content":USER}],tokenize=False,add_generation_prompt=True)
mm=re.search(r"```(?:python)?\s*(.*?)```",CODE0,re.S); code0=(mm.group(1) if mm else CODE0).strip()
prefix=tmpl+"```python\n"+code0[:code0.index("return")+len("return")]
ids=t(prefix,return_tensors="pt").input_ids.to(dev); toks=[t.decode([x]) for x in ids[0]]
hid={}
caps=[m.model.layers[L].register_forward_hook(lambda mod,inp,o,L=L: hid.__setitem__(L+1,(o[0] if isinstance(o,tuple) else o)[0].detach().float().cpu())) for L in range(NL)]
emb={}
m.model.embed_tokens.register_forward_hook(lambda mod,inp,o: emb.__setitem__(0,o[0].detach().float().cpu()))
with torch.no_grad(): out=m(ids)
for c in caps: c.remove()
hid[0]=emb[0]
us=prefix.index(USER); user_start=len(t(prefix[:us]).input_ids); user_end=len(t(prefix[:us+len(USER)]).input_ids)
P=ids.shape[1]-1   # pozice `return`
DIMS=[0,1,2,3,4,5,6,7]
def lens(v):
    x=m.model.norm(v.to(dev).to(m.model.norm.weight.dtype)); p=torch.softmax(m.lm_head(x).float(),-1); top=torch.topk(p,3)
    return [(t.decode([int(i)]),round(float(pv)*100,1)) for pv,i in zip(top.values,top.indices)]
res={"tokens_user":toks[user_start:user_end],"pos_token":toks[P],"layers":[]}
for L in range(NL+1):
    v=hid[L][P]; prev=hid[L-1][P] if L>0 else None
    cos=float(torch.nn.functional.cosine_similarity(v,prev,dim=0)) if prev is not None else None
    # kolik se změnilo přes celý prompt (průměr 1-cos přes všechny tokeny zadání)
    if prev is not None:
        c_all=torch.nn.functional.cosine_similarity(hid[L][user_start:user_end],hid[L-1][user_start:user_end],dim=1); chg=float((1-c_all).mean())
    else: chg=None
    res["layers"].append({"layer":L,"dims":[round(float(v[d]),2) for d in DIMS],"norm":round(float(v.norm()),1),"cos_prev":None if cos is None else round(cos,3),"change_all_user":None if chg is None else round(chg,4),"lens":lens(v)})
    print(f"L{L:2d} |v|={float(v.norm()):6.1f} prvních 8: {[round(float(v[d]),2) for d in DIMS]} cos s předchozím {'' if cos is None else round(cos,3)} lens {lens(v)[:2]}",flush=True)
# dokola: dalších 6 tokenů greedy s top-3 na každém kroku
steps=[]; cur=ids
for k in range(6):
    with torch.no_grad(): lg=m(cur).logits[0,-1].float()
    p=torch.softmax(lg,-1); top=torch.topk(p,3); nxt=int(top.indices[0])
    steps.append({"after":t.decode([int(cur[0,-1])]),"top":[(t.decode([int(i)]),round(float(pv)*100,1)) for pv,i in zip(top.values,top.indices)]})
    print(f"po {t.decode([int(cur[0,-1])])!r}: {steps[-1]['top']}",flush=True)
    cur=torch.cat([cur,torch.tensor([[nxt]],device=dev)],1)
res["autoregressive"]=steps; res["user_tokens_count"]=user_end-user_start
# embedding token id + prvních 8 čísel pro pár slov zadání (doslova "slovo → čísla")
res["embeddings_user"]=[{"tok":toks[i],"id":int(ids[0,i]),"dims":[round(float(hid[0][i][d]),2) for d in DIMS]} for i in range(user_start,min(user_start+8,user_end))]
json.dump(res,open("stream_walk.json","w"),ensure_ascii=False); print("STREAMWALKDONE",flush=True)
