#!/usr/bin/env python3
# Vrstva 20, všech 8 960 neuronů: každý zvlášť ×0 → p(sum) po `return` (teacher-forced prefix, jeden dopředný průchod na neuron; 100 s na RTX 4070 Ti SUPER).
# --fast: 64 neuronů najednou v jedné dávce, v každém řádku dávky vypnutý jiný (33 s). Hlavní výsledek stejný, drobné Δ se liší o desetiny.
# Uloží všech 8 960 rozdílů (obrázek „kolik jsem zkusila, než jsem našla 3545“) a aktivaci každého neuronu na `return`.
# K tomu neuron, který nic nedělá: na `return` svítí aspoň trochu (|akt| ≥ 1), ale ×0 i ×4 nechají p(sum) i kód skoro stejné.
# Greedy, teplota 0. Výstup: neuron_scan.json
import json, re, sys, torch, time
FAST="--fast" in sys.argv
from transformers import AutoTokenizer, AutoModelForCausalLM
dev="cuda" if torch.cuda.is_available() else "cpu"; I="Qwen/Qwen2.5-Coder-1.5B-Instruct"; LZ=20; NR=3545
Lk=json.load(open("orch_lock.json")); BSYS=Lk["builder_sys"]; USER=Lk["refiner_out"]; CODE0=Lk["builder_out"]
t=AutoTokenizer.from_pretrained(I); m=AutoModelForCausalLM.from_pretrained(I,dtype=(torch.bfloat16 if dev=="cuda" else torch.float32)).to(dev).eval()
def last_logits(x):   # skóre jen na poslední pozici (lm_head přes všech 87 pozic by byla zbytečná práce, na T4 největší brzda)
    try: return m(x,logits_to_keep=1).logits[:,-1]
    except TypeError: return m(x).logits[:,-1]
tmpl=t.apply_chat_template([{"role":"system","content":BSYS},{"role":"user","content":USER}],tokenize=False,add_generation_prompt=True)
ids=t(tmpl,return_tensors="pt").input_ids.to(dev)
mm=re.search(r"```(?:python)?\s*(.*?)```",CODE0,re.S); code0=(mm.group(1) if mm else CODE0).strip()
pref=t(tmpl+"```python\n"+code0[:code0.index("return")+len("return")],return_tensors="pt").input_ids.to(dev)
SUM=t(" sum",add_special_tokens=False).input_ids[0]
scale={"n":None,"f":1.0,"ns":None}
def pre(mod,args):
    if scale["ns"] is not None:   # dávka: řádek b má vypnutý neuron ns[b]
        x=args[0].clone(); b=torch.arange(x.shape[0],device=x.device); nn=torch.tensor(scale["ns"],device=x.device)
        x[b,:,nn]=x[b,:,nn]*scale["f"]; return (x,)
    if scale["n"] is None: return None
    x=args[0].clone(); x[...,scale["n"]]=x[...,scale["n"]]*scale["f"]; return (x,)
m.model.layers[LZ].mlp.down_proj.register_forward_pre_hook(pre)
box={}
h=m.model.layers[LZ].mlp.down_proj.register_forward_pre_hook(lambda mod,args: box.__setitem__("act",args[0][0,-1].float().detach().cpu()))
with torch.no_grad(): base=torch.softmax(last_logits(pref)[0].float(),-1)
h.remove()
act=box["act"]; p0=float(base[SUM])*100
print(f"bez zásahu p(sum) {p0:.1f} %, aktivace 3545 na return {float(act[NR]):.1f}",flush=True)
N=m.config.intermediate_size; delta=[]; t0=time.time(); B=64 if FAST else 1
with torch.no_grad():
    for s0 in range(0,N,B):
        ns=list(range(s0,min(s0+B,N))); scale["ns"]=ns; scale["f"]=0.0
        p=torch.softmax(last_logits(pref.repeat(len(ns),1)).float(),-1)[:,SUM]
        delta.extend([float(x)*100-p0 for x in p])
        if s0%1024==0: print(f"  {s0}/{N}  {time.time()-t0:.0f} s",flush=True)
scale["ns"]=None; print(f"sken {N} neuronů: {time.time()-t0:.0f} s",flush=True)
order=sorted(range(N),key=lambda n:delta[n]); print("největší propad p(sum):",[(n,round(delta[n],1)) for n in order[:6]],flush=True)
# neuron, který nic nedělá: svítí na return (|akt| >= 1), ale vypnutí skoro nic nezmění
NULLK=1507   # ten, co vyšel v přednášce; zkusí se první, pak ostatní v pořadí |Δ| (mezi Δ blízko nuly rozhoduje šum)
cands=sorted([n for n in range(N) if abs(float(act[n]))>=1.0 and n!=NR],key=lambda n:abs(delta[n]))
if NULLK in cands: cands.remove(NULLK); cands.insert(0,NULLK)
def gen(n,f):
    scale["n"]=n; scale["f"]=f
    with torch.no_grad():
        p=torch.softmax(last_logits(pref)[0].float(),-1); g=m.generate(ids,max_new_tokens=120,do_sample=False,pad_token_id=t.eos_token_id)
    scale["n"]=None
    return {"p_sum":round(float(p[SUM])*100,1),"code":t.decode(g[0,ids.shape[1]:],skip_special_tokens=True).strip()}
base_code=gen(None,1.0); null=None
for n in cands[:20]:
    r0=gen(n,0.0); r4=gen(n,4.0)
    if r0["code"]==base_code["code"] and r4["code"]==base_code["code"] and abs(r4["p_sum"]-p0)<2:
        null={"n":n,"act":round(float(act[n]),2),"off":r0,"x4":r4}; break
print("neuron, který nic nedělá:",null and (null["n"],null["act"],null["off"]["p_sum"],null["x4"]["p_sum"]),flush=True)
json.dump({"layer":LZ,"p_sum_base":round(p0,2),"delta_off":[round(x,2) for x in delta],"act_at_return":[round(float(x),2) for x in act],
           "top_drop":[(n,round(delta[n],1)) for n in order[:15]],"null_neuron":null,"base_code":base_code["code"],
           "n_tested":N,"fast":FAST,"n_with_drop_over_5":sum(1 for x in delta if x<-5),"n_with_abs_over_1":sum(1 for x in delta if abs(x)>1)},open("neuron_scan.json","w"),ensure_ascii=False)
print("NEURONSCANDONE",flush=True)
