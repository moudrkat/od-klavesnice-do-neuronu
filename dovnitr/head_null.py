#!/usr/bin/env python3
# Vrstva 20: vypni každou z 12 hlav zvlášť (×0) → p(dalšího tokenu po `return`) + celý kód. Najdi hlavu, která nic nezmění
# (stejný kód, nejmenší posun p(sum)). K tomu celé matice pozornosti 87×87 pro hlavu 20·6 a pro tu "línou" hlavu (eager attention).
# Greedy, teplota 0. Stejné tokeny jako full_context.py (tmpl + ```python + kód + ```). Výstup: head_null.json
import json, re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
dev="cuda"; I="Qwen/Qwen2.5-Coder-1.5B-Instruct"; LZ=20; HD=6
Lk=json.load(open("orch_lock.json")); BSYS=Lk["builder_sys"]; USER=Lk["refiner_out"]; CODE0=Lk["builder_out"]
t=AutoTokenizer.from_pretrained(I)
tmpl=t.apply_chat_template([{"role":"system","content":BSYS},{"role":"user","content":USER}],tokenize=False,add_generation_prompt=True)
ids=t(tmpl,return_tensors="pt").input_ids.to(dev)
mm=re.search(r"```(?:python)?\s*(.*?)```",CODE0,re.S); code0=(mm.group(1) if mm else CODE0).strip()
pref=t(tmpl+"```python\n"+code0[:code0.index("return")+len("return")],return_tensors="pt").input_ids.to(dev)
# --- 1) ablace po hlavách (sdpa jako v decku) ---
m=AutoModelForCausalLM.from_pretrained(I,dtype=torch.bfloat16).to(dev).eval()
H=m.config.num_attention_heads; D=m.config.hidden_size//H
def head_hook(h,f):
    def pre(mod,args):
        x=args[0].clone(); x[...,h*D:(h+1)*D]=x[...,h*D:(h+1)*D]*f; return (x,)
    return m.model.layers[LZ].self_attn.o_proj.register_forward_pre_hook(pre)
def run(h,f):
    hk=[] if f==1 else [head_hook(h,f)]
    with torch.no_grad():
        p=torch.softmax(m(pref).logits[0,-1].float(),-1)
        g=m.generate(ids,max_new_tokens=120,do_sample=False,pad_token_id=t.eos_token_id)
    for x in hk: x.remove()
    code=t.decode(g[0,ids.shape[1]:],skip_special_tokens=True).strip()
    pw=lambda w: round(float(p[t(w,add_special_tokens=False).input_ids[0]])*100,2)
    top=torch.topk(p,3)
    return {"p_sum":pw(" sum"),"p_len":pw(" len"),"p_tasks":pw(" tasks"),"top":[(t.decode([int(i)]),round(float(v)*100,1)) for v,i in zip(top.values,top.indices)],"code":code}
base=run(0,1); print("bez zásahu:",base["top"][:2],flush=True)
abl=[]
for h in range(H):
    r=run(h,0); r["head"]=h; r["same_code"]=(r["code"]==base["code"]); r["d_sum"]=round(r["p_sum"]-base["p_sum"],2); abl.append(r)
    print(f"hlava 20·{h:2d} ×0 → sum {r['p_sum']:5.1f} % (Δ {r['d_sum']:+5.1f}) kód {'stejný' if r['same_code'] else 'JINÝ'}",flush=True)
cands=[r for r in abl if r["same_code"] and r["head"]!=HD]
null=min(cands,key=lambda r: abs(r["d_sum"]))["head"] if cands else None
print("líná hlava:",null,flush=True)
del m; torch.cuda.empty_cache()
# --- 2) matice pozornosti (eager) na stejných 87 tokenech jako full_context.py ---
m=AutoModelForCausalLM.from_pretrained(I,dtype=torch.bfloat16,attn_implementation="eager").to(dev).eval()
full=tmpl+"```python\n"+code0+"```"; enc=t(full,return_offsets_mapping=True); fids=torch.tensor([enc.input_ids]).to(dev); offs=enc.offset_mapping
toks=[full[a:b] for a,b in offs]
us=full.index(USER); ue=us+len(USER); ss=full.index(BSYS); se=ss+len(BSYS); cs=full.index("def ")
grp=["první token" if i==0 else ("systémový prompt" if a>=ss and b<=se else ("tvoje zadání" if a>=us and b<=ue else ("jeho kód" if a>=cs else "šablona"))) for i,(a,b) in enumerate(offs)]
with torch.no_grad(): out=m(fids,output_attentions=True)
A=out.attentions[LZ][0].float().cpu()   # [H,T,T]
rnd=lambda M:[[round(float(x),4) for x in row] for row in M]
res={"tokens":toks,"group":grp,"ret_pos":[i for i in range(len(toks)) if grp[i]=="jeho kód" and toks[i].strip()=="return"][0],
     "base":base,"ablation":abl,"null_head":null,"matrix_20_6":rnd(A[HD]),"matrix_null":(rnd(A[null]) if null is not None else None),
     "sink_share_per_head":[round(float(A[h][:,0].mean()),3) for h in range(H)]}
json.dump(res,open("head_null.json","w"),ensure_ascii=False)
print("podíl pozornosti na první token (odkladiště) po hlavách:",res["sink_share_per_head"])
print("HEADNULLDONE",flush=True)
