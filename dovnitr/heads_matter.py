#!/usr/bin/env python3
# Hlava, která něco dělá: kumulativní vypnutí nejsilnějších hlav (podle poklesu p(sum)) → p(sum) a celý kód; attention hlavy 20·6.
import json, re, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
dev="cuda" if torch.cuda.is_available() else "cpu"; I="Qwen/Qwen2.5-Coder-1.5B-Instruct"
Lk=json.load(open("orch_lock.json")); BSYS=Lk["builder_sys"]; USER=Lk["refiner_out"]; CODE0=Lk["builder_out"]
t=AutoTokenizer.from_pretrained(I); m=AutoModelForCausalLM.from_pretrained(I,dtype=(torch.bfloat16 if dev=="cuda" else torch.float32),attn_implementation="eager").to(dev).eval()
H=m.config.num_attention_heads; D=m.config.hidden_size//H
cl=json.load(open("circuit_layer.json")); order=[(L,h) for d,L,h in cl["head_causal_all"]]
tmpl=t.apply_chat_template([{"role":"system","content":BSYS},{"role":"user","content":USER}],tokenize=False,add_generation_prompt=True)
ids=t(tmpl,return_tensors="pt").input_ids.to(dev)
mm=re.search(r"```(?:python)?\s*(.*?)```",CODE0,re.S); code0=(mm.group(1) if mm else CODE0).strip()
full=tmpl+"```python\n"+code0+"```"; enc=t(full,return_offsets_mapping=True); fids=torch.tensor([enc.input_ids]).to(dev); offs=enc.offset_mapping
us=full.index(USER); ue=us+len(USER); cs=full.index("def ")
user_toks=[i for i,(a,b) in enumerate(offs) if a>=us and b<=ue]; code_toks=[i for i,(a,b) in enumerate(offs) if a>=cs]; lab=lambda i: full[offs[i][0]:offs[i][1]]
ret=[i for i in code_toks if lab(i).strip()=="return"][0]; pref=fids[:,:ret+1]
def hooks(heads):
    by={}
    for L,h in heads: by.setdefault(L,[]).append(h)
    hs=[]
    for L,hh in by.items():
        def pre(mod,args,hh=hh):
            x=args[0].clone()
            for h in hh: x[...,h*D:(h+1)*D]=0
            return (x,)
        hs.append(m.model.layers[L].self_attn.o_proj.register_forward_pre_hook(pre))
    return hs
def p_next(hs):
    with torch.no_grad(): p=torch.softmax(m(pref).logits[0,-1].float(),-1)
    top=torch.topk(p,3); return [(t.decode([int(i)]),round(float(v)*100,1)) for v,i in zip(top.values,top.indices)]
def gen(hs):
    with torch.no_grad(): g=m.generate(ids,max_new_tokens=120,do_sample=False,pad_token_id=t.eos_token_id)
    return t.decode(g[0,ids.shape[1]:],skip_special_tokens=True).strip()
out={"order":order[:12],"cumulative":[]}
for k in (0,1,2,3,4,6,8,12):
    hs=hooks(order[:k]); r={"k":k,"heads":order[:k],"p_next":p_next(hs),"code":gen(hs)}
    for h in hs: h.remove()
    out["cumulative"].append(r); print(f"vypnuto {k:2d} nejsilnějších hlav: {r['p_next'][:2]} | {r['code'].replace(chr(10),' ⏎ ')[:140]}",flush=True)
with torch.no_grad(): att=m(fids,output_attentions=True).attentions
for (L,h) in order[:3]:
    A=att[L][0,h].float().cpu().numpy()
    su=A[np.ix_(code_toks,user_toks)]; sc=A[np.ix_(code_toks,code_toks)]
    r={"head":[L,h],"user_mass":round(float(su.sum(1).mean()),3),"code_x_user":[[round(float(v),3) for v in row] for row in su],"code_x_code":[[round(float(v),3) for v in row] for row in sc],
       "from_return":sorted([(round(float(A[ret,j]),2),lab(j)) for j in range(fids.shape[1])],reverse=True)[:5],
       "strongest_to_user":sorted([(round(float(su[ci].max()),2),lab(code_toks[ci]),lab(user_toks[int(su[ci].argmax())])) for ci in range(len(code_toks))],reverse=True)[:5]}
    out.setdefault("heads",[]).append(r); print(f"hlava {L}·{h}: hmota do zadání {r['user_mass']}, z pozice return se dívá na {r['from_return'][:3]}, nejsilnější kód→zadání {r['strongest_to_user'][:3]}",flush=True)
out["user_labels"]=[lab(i) for i in user_toks]; out["code_labels"]=[lab(i) for i in code_toks]
json.dump(out,open("heads_matter.json","w"),ensure_ascii=False); print("HEADSDONE",flush=True)
