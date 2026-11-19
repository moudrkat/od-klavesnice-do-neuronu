#!/usr/bin/env python3
# Patro 20 zblízka: hlava 20·6 → neuron v MLP 20. Sken neuronů patra 20 podle poklesu p(sum); pro top-3: kód po zhasnutí,
# aktivace po tokenech (zadání + kód), a vliv hlavy 20·6 (×0, ×2) na aktivaci neuronu. Jen dopředné průchody a greedy.
import json, re, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
dev="cuda" if torch.cuda.is_available() else "cpu"; I="Qwen/Qwen2.5-Coder-1.5B-Instruct"; LZ=20; HD=6
Lk=json.load(open("orch_lock.json")); BSYS=Lk["builder_sys"]; USER=Lk["refiner_out"]; CODE0=Lk["builder_out"]
t=AutoTokenizer.from_pretrained(I); m=AutoModelForCausalLM.from_pretrained(I,dtype=(torch.bfloat16 if dev=="cuda" else torch.float32)).to(dev).eval()
H=m.config.num_attention_heads; D=m.config.hidden_size//H
tmpl=t.apply_chat_template([{"role":"system","content":BSYS},{"role":"user","content":USER}],tokenize=False,add_generation_prompt=True)
ids=t(tmpl,return_tensors="pt").input_ids.to(dev)
mm=re.search(r"```(?:python)?\s*(.*?)```",CODE0,re.S); code0=(mm.group(1) if mm else CODE0).strip()
full=tmpl+"```python\n"+code0+"```"; enc=t(full,return_offsets_mapping=True); fids=torch.tensor([enc.input_ids]).to(dev); offs=enc.offset_mapping
us=full.index(USER); ue=us+len(USER); cs=full.index("def ")
user_toks=[i for i,(a,b) in enumerate(offs) if a>=us and b<=ue]; code_toks=[i for i,(a,b) in enumerate(offs) if a>=cs]; lab=lambda i: full[offs[i][0]:offs[i][1]]
ret=[i for i in code_toks if lab(i).strip()=="return"][0]; pref=fids[:,:ret+1]
def head_hook(f):
    def pre(mod,args):
        x=args[0].clone(); x[...,HD*D:(HD+1)*D]=x[...,HD*D:(HD+1)*D]*f; return (x,)
    return m.model.layers[LZ].self_attn.o_proj.register_forward_pre_hook(pre)
def neuron_hook(n,f):
    def pre(mod,args):
        x=args[0].clone(); x[...,n]=x[...,n]*f; return (x,)
    return m.model.layers[LZ].mlp.down_proj.register_forward_pre_hook(pre)
def p_next(hs):
    with torch.no_grad(): p=torch.softmax(m(pref).logits[0,-1].float(),-1)
    top=torch.topk(p,3); return [(t.decode([int(i)]),round(float(v)*100,1)) for v,i in zip(top.values,top.indices)]
def gen(hs):
    with torch.no_grad(): g=m.generate(ids,max_new_tokens=120,do_sample=False,pad_token_id=t.eos_token_id)
    return t.decode(g[0,ids.shape[1]:],skip_special_tokens=True).strip()
with torch.no_grad(): base=torch.softmax(m(pref).logits[0,-1].float(),-1)
sid=int(torch.argmax(base)); print("bez zásahu po return:",t.decode([sid]),round(float(base[sid])*100,1),flush=True)
nd=[]
for n in range(m.config.intermediate_size):
    hk=neuron_hook(n,0.0)
    with torch.no_grad(): p=torch.softmax(m(pref).logits[0,-1].float(),-1)
    hk.remove(); nd.append((float(base[sid]-p[sid])*100,n))
nd.sort(reverse=True); print("vrstva 20, vypnutí neuronu → pokles p(sum), top 8:",[(round(d,1),n) for d,n in nd[:8]],flush=True)
out={"scan":[(round(d,1),n) for d,n in nd[:15]],"neurons":{}}
box={}
cap=m.model.layers[LZ].mlp.down_proj.register_forward_pre_hook(lambda mod,args: box.__setitem__("x",args[0][0].float().detach().cpu()))
with torch.no_grad(): m(fids)
X0=box["x"]
for f in (0.0,2.0):
    hk=head_hook(f)
    with torch.no_grad(): m(fids)
    hk.remove(); box[f]=box["x"]
cap.remove()
for d,n in nd[:3]:
    hs=[neuron_hook(n,0.0)]; r={"drop":round(d,1),"p_next_off":p_next(hs),"code_off":gen(hs)}
    for h in hs: h.remove()
    r["user_strip"]=[round(float(X0[i,n]),2) for i in user_toks]; r["code_strip"]=[round(float(X0[i,n]),2) for i in code_toks]
    r["at_return"]={"head x1":round(float(X0[ret,n]),2),"head x0":round(float(box[0.0][ret,n]),2),"head x2":round(float(box[2.0][ret,n]),2)}
    top_u=sorted(zip(r["user_strip"],[lab(i) for i in user_toks]),key=lambda p:-abs(p[0]))[:4]; top_c=sorted(zip(r["code_strip"],[lab(i) for i in code_toks]),key=lambda p:-abs(p[0]))[:4]
    print(f"=== neuron 20·{n} (−{d:.1f} b.): po zhasnutí {r['p_next_off'][:2]} | kód: {r['code_off'].replace(chr(10),' ⏎ ')[:120]}",flush=True)
    print(f"    svítí na zadání: {top_u} | na kódu: {top_c} | na return při hlavě ×1/×0/×2: {r['at_return']}",flush=True)
    out["neurons"][str(n)]=r
out["user_labels"]=[lab(i) for i in user_toks]; out["code_labels"]=[lab(i) for i in code_toks]
json.dump(out,open("layer20.json","w"),ensure_ascii=False); print("LAYER20DONE",flush=True)
