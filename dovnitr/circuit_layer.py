#!/usr/bin/env python3
# Přiblížení jedné vrstvy na NAŠEM promptu (get_completed_tasks, orch_lock.json), pro vrstvu 27 (neuron 27·8226) i 22 (neuron 22·8350):
# (0) kauzální výběr hlavy: vypnutí které hlavy nejvíc shodí p(sum) po `return` (všech 336 + nejlepší v dané vrstvě)
# (1) pro vybranou hlavu: matice kód×kód a kód×zadání; (2) hlava ×0/×2/×4 → kód + p(sum); (3) hlava ×0/×2/×4 → aktivace neuronu
# na pozicích, kde svítí; (4) logit lens po vrstvách na pozici po `return`. Jen dopředné průchody a greedy generování.
import json, re, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
dev="cuda" if torch.cuda.is_available() else "cpu"; I="Qwen/Qwen2.5-Coder-1.5B-Instruct"
Lk=json.load(open("orch_lock.json")); BSYS=Lk["builder_sys"]; USER=Lk["refiner_out"]; CODE0=Lk["builder_out"]
t=AutoTokenizer.from_pretrained(I); m=AutoModelForCausalLM.from_pretrained(I,dtype=(torch.bfloat16 if dev=="cuda" else torch.float32),attn_implementation="eager").to(dev).eval()
H=m.config.num_attention_heads; D=m.config.hidden_size//H; NL=m.config.num_hidden_layers
tmpl=t.apply_chat_template([{"role":"system","content":BSYS},{"role":"user","content":USER}],tokenize=False,add_generation_prompt=True)
ids=t(tmpl,return_tensors="pt").input_ids.to(dev)
mm=re.search(r"```(?:python)?\s*(.*?)```",CODE0,re.S); code0=(mm.group(1) if mm else CODE0).strip()
full=tmpl+"```python\n"+code0+"```"; enc=t(full,return_offsets_mapping=True); fids=torch.tensor([enc.input_ids]).to(dev); offs=enc.offset_mapping
us=full.index(USER); ue=us+len(USER); cs=full.index("def ")
user_toks=[i for i,(a,b) in enumerate(offs) if a>=us and b<=ue]; code_toks=[i for i,(a,b) in enumerate(offs) if a>=cs]
lab=lambda i: full[offs[i][0]:offs[i][1]].replace("\n","\\n")
ret_pos=[i for i in code_toks if lab(i).strip()=="return"][0]       # pozice tokenu `return` v plném kontextu
prefix_ids=fids[:, :ret_pos+1]                                       # kontext až po `return` včetně → předpověď dalšího tokenu
def head_hook(L,h,f):
    def pre(mod,args):
        x=args[0].clone(); x[...,h*D:(h+1)*D]=x[...,h*D:(h+1)*D]*f; return (x,)
    return m.model.layers[L].self_attn.o_proj.register_forward_pre_hook(pre)
def neuron_hook(L,n,f):
    def pre(mod,args):
        x=args[0].clone(); x[...,n]=x[...,n]*f; return (x,)
    return m.model.layers[L].mlp.down_proj.register_forward_pre_hook(pre)
def p_next(hs):
    with torch.no_grad(): p=torch.softmax(m(prefix_ids).logits[0,-1].float(),-1)
    top=torch.topk(p,5); return [(t.decode([int(i)]),round(float(v)*100,1)) for v,i in zip(top.values,top.indices)]
def gen(hs):
    with torch.no_grad(): g=m.generate(ids,max_new_tokens=160,do_sample=False,pad_token_id=t.eos_token_id)
    return t.decode(g[0,ids.shape[1]:],skip_special_tokens=True).strip()
out={"user_labels":[lab(i) for i in user_toks],"code_labels":[lab(i) for i in code_toks]}
# ---------- (0) kauzální výběr hlavy
with torch.no_grad(): base=torch.softmax(m(prefix_ids).logits[0,-1].float(),-1)
sid=int(torch.argmax(base)); print("bez zásahu po `return`: %r %.1f %%"%(t.decode([sid]),float(base[sid])*100),flush=True)
drops=[]
for L in range(NL):
    for h in range(H):
        hk=head_hook(L,h,0.0)
        with torch.no_grad(): p=torch.softmax(m(prefix_ids).logits[0,-1].float(),-1)
        hk.remove(); drops.append((float(base[sid]-p[sid])*100,L,h))
drops.sort(reverse=True); out["head_causal_all"]=[(round(d,1),L,h) for d,L,h in drops[:15]]
print("vypnutí hlavy → pokles p(sum) v bodech, top 15:",out["head_causal_all"],flush=True)
# sken neuronů ve vrstvě 14 (a pro srovnání 27): vypnutí kterého nejvíc shodí p(sum) po `return`
out["neuron_scan"]={}
for LZ in (14,27):
    nd=[]
    for n in range(m.config.intermediate_size):
        hk=neuron_hook(LZ,n,0.0)
        with torch.no_grad(): p=torch.softmax(m(prefix_ids).logits[0,-1].float(),-1)
        hk.remove(); nd.append((float(base[sid]-p[sid])*100,n))
    nd.sort(reverse=True); out["neuron_scan"][str(LZ)]=[(round(d,1),n) for d,n in nd[:15]]
    print(f"vrstva {LZ}: vypnutí neuronu → pokles p(sum), top 10: {out['neuron_scan'][str(LZ)][:10]}",flush=True)
    json.dump(out,open("circuit_layer.json","w"),ensure_ascii=False)
# který neuron ve 14 svítí nejvíc na zadání (a na čem)
box={}
hk=m.model.layers[14].mlp.down_proj.register_forward_pre_hook(lambda mod,args: box.__setitem__("x",args[0][0].float().detach().cpu()))
with torch.no_grad(): m(fids)
hk.remove(); x=box["x"]
ua=x[user_toks].abs().mean(0); ca=x[code_toks].abs().mean(0)
out["l14_top_on_user"]=[(int(n),round(float(ua[n]),2),lab(user_toks[int(x[user_toks,n].abs().argmax())])) for n in ua.argsort(descending=True)[:8].tolist()]
out["l14_top_on_code"]=[(int(n),round(float(ca[n]),2),lab(code_toks[int(x[code_toks,n].abs().argmax())])) for n in ca.argsort(descending=True)[:8].tolist()]
print("vrstva 14, neurony nejaktivnější na zadání (neuron, průměr |akt|, token vrcholu):",out["l14_top_on_user"],flush=True)
print("vrstva 14, neurony nejaktivnější na kódu:",out["l14_top_on_code"],flush=True)
FOCUS={27:8226,22:8350,14:out["neuron_scan"]["14"][0][1]}; out["layers"]={}
with torch.no_grad(): att=m(fids,output_attentions=True).attentions
for LZ,NEUR in FOCUS.items():
    inlayer=sorted([(d,h) for d,L,h in drops if L==LZ],reverse=True); hbest=inlayer[0][1]
    r={"neuron":NEUR,"heads_in_layer":[(round(d,1),h) for d,h in inlayer],"head":hbest}
    print(f"=== vrstva {LZ}: hlavy podle poklesu p(sum): {r['heads_in_layer'][:6]} → beru hlavu {LZ}·{hbest}",flush=True)
    A=att[LZ][0,hbest].float().cpu().numpy()
    r["code_x_code"]=[[round(float(A[i,j]),3) for j in code_toks] for i in code_toks]
    r["code_x_user"]=[[round(float(A[i,j]),3) for j in user_toks] for i in code_toks]
    sub=A[np.ix_(code_toks,user_toks)]; r["user_mass"]=round(float(sub.sum(1).mean()),3)
    for ci,i in enumerate(code_toks):
        j=int(sub[ci].argmax()); 
    r["strongest_to_user"]=sorted([(round(float(sub[ci].max()),2),lab(code_toks[ci]),lab(user_toks[int(sub[ci].argmax())])) for ci in range(len(code_toks))],reverse=True)[:6]
    r["strongest_in_code"]=sorted([(round(float(A[i,j]),2),lab(i),lab(j)) for i in code_toks for j in code_toks if j<i],reverse=True)[:6]
    print("   kód→zadání nejsilnější:",r["strongest_to_user"],flush=True); print("   kód→kód nejsilnější:",r["strongest_in_code"],flush=True)
    # (2) hlava ×0/×2/×4
    r["head_dose"]={}
    for f in (1.0,0.0,2.0,4.0):
        hs=[] if f==1.0 else [head_hook(LZ,hbest,f)]; rr={"p_next":p_next(hs),"code":gen(hs)}
        for h in hs: h.remove()
        r["head_dose"][str(f)]=rr; print(f"   hlava {LZ}·{hbest} ×{f:g}: {rr['p_next'][:3]} | {rr['code'].replace(chr(10),' ⏎ ')[:150]}",flush=True)
    # (3) hlava → neuron: aktivace neuronu na kódových pozicích (plný kontext), při hlavě ×0/×1/×2/×4
    r["head_to_neuron"]={}
    box={}
    for f in (1.0,0.0,2.0,4.0):
        hs=([] if f==1.0 else [head_hook(LZ,hbest,f)])+[m.model.layers[LZ].mlp.down_proj.register_forward_pre_hook(lambda mod,args: box.__setitem__("x",args[0][0,:,NEUR].float().detach().cpu()))]
        with torch.no_grad(): m(fids)
        for h in hs: h.remove()
        x=box["x"]; r["head_to_neuron"][str(f)]={lab(i):round(float(x[i]),2) for i in code_toks}
        top=sorted(((float(x[i]),lab(i)) for i in code_toks),key=lambda p:-abs(p[0]))[:5]
        print(f"   hlava ×{f:g} → neuron {LZ}·{NEUR} na kódu, největší: {[(l,round(v,2)) for v,l in top]}",flush=True)
    # neuron ×0 → kód (kontrola)
    hs=[neuron_hook(LZ,NEUR,0.0)]; r["neuron_off"]={"p_next":p_next(hs),"code":gen(hs)}
    for h in hs: h.remove()
    print(f"   neuron {LZ}·{NEUR} ×0: {r['neuron_off']['p_next'][:3]} | {r['neuron_off']['code'].replace(chr(10),' ⏎ ')[:150]}",flush=True)
    out["layers"][str(LZ)]=r; json.dump(out,open("circuit_layer.json","w"),ensure_ascii=False)
# ---------- (3b) po vrstvách: kolik pozornosti jde z kódu do zadání (průměr přes hlavy a kódové tokeny)
prof=[]
for L in range(NL):
    A=att[L][0].float().cpu().numpy()          # [H, T, T]
    sub=A[:, code_toks][:, :, user_toks]        # [H, code, user]
    prof.append(round(float(sub.sum(2).mean()),3))
out["attn_to_user_by_layer"]=prof; print("pozornost kód→zadání po vrstvách (podíl hmoty):",prof,flush=True)
# ---------- (3c) exaktní stopa: kam se v každé vrstvě dívá pozice `return` (průměr přes hlavy), top 3 tokeny
trace=[]
for L in range(NL):
    row=att[L][0,:,ret_pos,:].float().mean(0).cpu().numpy()      # průměr přes hlavy, pozornost z pozice `return`
    order=row.argsort()[::-1][:3]
    trace.append([(lab(int(j)) if int(j)>=user_toks[0] else "<systém/šablona>",round(float(row[j]),3),"zadání" if int(j) in user_toks else ("kód" if int(j) in code_toks else "šablona")) for j in order])
out["return_attention_trace"]=trace
print("stopa čtení z pozice `return` po vrstvách (top token, podíl, odkud):",flush=True)
for L,r_ in enumerate(trace): print("  L%2d  %s"%(L,"  ".join(f"{tok!r} {p} ({src})" for tok,p,src in r_)),flush=True)
# ---------- (3d) kauzálně po vrstvách: vypnout celé MLP / celou attention vrstvy k → p(sum)
def layer_mlp_hook(L):
    return m.model.layers[L].mlp.register_forward_hook(lambda mod,inp,o: torch.zeros_like(o))
def layer_attn_hook(L):
    return m.model.layers[L].self_attn.o_proj.register_forward_pre_hook(lambda mod,args: (torch.zeros_like(args[0]),))
prof_mlp=[]; prof_attn=[]
for L in range(NL):
    hk=layer_mlp_hook(L)
    with torch.no_grad(): p=torch.softmax(m(prefix_ids).logits[0,-1].float(),-1)
    hk.remove(); prof_mlp.append(round(float(p[sid])*100,1))
    hk=layer_attn_hook(L)
    with torch.no_grad(): p=torch.softmax(m(prefix_ids).logits[0,-1].float(),-1)
    hk.remove(); prof_attn.append(round(float(p[sid])*100,1))
out["p_sum_without_layer_mlp"]=prof_mlp; out["p_sum_without_layer_attn"]=prof_attn
print("p(sum) %% bez MLP vrstvy k:  ",prof_mlp,flush=True); print("p(sum) %% bez attention vrstvy k:",prof_attn,flush=True)
json.dump(out,open("circuit_layer.json","w"),ensure_ascii=False)
# ---------- (4) logit lens po `return` (jen jako nápověda, je to aproximace)
hid={}
caps=[m.model.layers[L].register_forward_hook(lambda mod,inp,o,L=L: hid.__setitem__(L,(o[0] if isinstance(o,tuple) else o)[0,-1].detach())) for L in range(NL)]
with torch.no_grad(): m(prefix_ids)
for c in caps: c.remove()
lens=[]
for L in range(NL):
    x=m.model.norm(hid[L].to(m.model.norm.weight.dtype)); p=torch.softmax(m.lm_head(x).float(),-1); top=torch.topk(p,3)
    lens.append({"layer":L+1,"top":[(t.decode([int(i)]),round(float(v)*100,1)) for v,i in zip(top.values,top.indices)]})
out["lens_after_return"]=lens; print("=== logit lens po `return`:",flush=True)
for r_ in lens: print("  L%2d  %s"%(r_["layer"],"  ".join(f"{tok!r}:{p}" for tok,p in r_["top"])),flush=True)
json.dump(out,open("circuit_layer.json","w"),ensure_ascii=False); print("CIRCUITLAYERDONE",flush=True)
