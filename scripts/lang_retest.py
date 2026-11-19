#!/usr/bin/env python3
# Jazyk retest: neutrální prompt + směr ze SIGNATURY (ne těla) + steer jen střední vrstvy.
import json,re,torch
from transformers import AutoTokenizer, AutoModelForCausalLM
dev="cuda" if torch.cuda.is_available() else "cpu"; I="Qwen/Qwen2.5-Coder-1.5B-Instruct"
tok=AutoTokenizer.from_pretrained(I); m=AutoModelForCausalLM.from_pretrained(I,dtype=(torch.bfloat16 if dev=="cuda" else torch.float32)).to(dev).eval()
NL=m.config.num_hidden_layers; HS=m.config.hidden_size
BSYS="You are a coding agent. You receive a task and write one function. Reply with code only."
BAND=list(range(6,19))  # střední vrstvy
PAIRS=[
 ("def factorial(n):","function factorial(n) {"),
 ("def add(a, b):","function add(a, b) {"),
 ("def is_even(n):","function isEven(n) {"),
 ("def greet(name):","function greet(name) {"),
 ("def square(x):","function square(x) {"),
 ("def max_of(a, b):","function maxOf(a, b) {"),
 ("def reverse(s):","function reverse(s) {"),
 ("def count_up(n):","function countUp(n) {"),
]
def sigres(text):  # resid jen přes tokeny signatury (prvních ~8), per layer
    ids=tok(text,return_tensors="pt").input_ids.to(dev)
    with torch.no_grad(): hs=m(ids,output_hidden_states=True).hidden_states
    k=min(ids.shape[1],8)
    return [hs[L+1][0,:k].mean(0).float().cpu() for L in range(NL)]
aP=[torch.zeros(HS) for _ in range(NL)]; aJ=[torch.zeros(HS) for _ in range(NL)]
for py,js in PAIRS:
    rp=sigres(py); rj=sigres(js)
    for L in range(NL): aP[L]+=rp[L]; aJ[L]+=rj[L]
dirs={}
for L in BAND:
    d=(aJ[L]-aP[L])/len(PAIRS); d=d/(d.norm()+1e-6); dirs[L]=d.to(dev).to(m.dtype)
def hooks(coef):
    hh=[]
    for L in BAND:
        def mk(dv):
            def h(mod,inp,out):
                o=out[0] if isinstance(out,tuple) else out; o=o+coef*dv
                return (o,)+out[1:] if isinstance(out,tuple) else o
            return h
        hh.append(m.model.layers[L].register_forward_hook(mk(dirs[L])))
    return hh
def gen(user,coef=0.0,n=80):
    ids=tok(tok.apply_chat_template([{"role":"system","content":BSYS},{"role":"user","content":user}],tokenize=False,add_generation_prompt=True),return_tensors="pt").input_ids.to(dev)
    hh=hooks(coef) if coef!=0 else []
    with torch.no_grad(): g=m.generate(ids,max_new_tokens=n,do_sample=False,pad_token_id=tok.eos_token_id)
    for h in hh: h.remove()
    return tok.decode(g[0,ids.shape[1]:],skip_special_tokens=True).strip()
def is_js(o):
    o2=o.replace("```javascript","").replace("```js","").replace("```","")
    return ("function " in o2 or "=>" in o2 or "const " in o2 or "let " in o2 or "console.log" in o2) and "def " not in o2
def gib(o):  # smetí?
    return len(re.findall(r"[}\];]",o))>max(8,len(o)//6) or len(set(o.split()))<4
PROMPTS=["Write a function that returns the factorial of n.",
         "Write a function that checks whether a number is even.",
         "Write a function that reverses a string."]
for user in PROMPTS:
    print("\n### "+user,flush=True)
    b=gen(user,0.0); print("  base      :",(" ".join(b.split()))[:80],"[JS]" if is_js(b) else "[PY]",flush=True)
    for coef in [2.0,4.0,6.0,8.0]:
        o=gen(user,coef); tag="<<< JS!" if is_js(o) else ("(smetí)" if gib(o) else "")
        print(f"  coef {coef:>4}: "+(" ".join(o.split()))[:80],tag,flush=True)
print("\nLANGRETESTDONE",flush=True)
