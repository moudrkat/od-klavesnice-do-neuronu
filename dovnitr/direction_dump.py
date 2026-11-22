#!/usr/bin/env python3
# Směr, doslova: stejný výpočet jako lang_slide.py (signatury py vs js, vrstvy 6–18), uloží směr jako direction_L{L}.npy (1 536 čísel, norma 1).
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
import numpy as np
for L in BAND: np.save(f"direction_L{L}.npy", dirs[L].float().cpu().numpy())
print("DIRDONE", {L: float(dirs[L].float().abs().max()) for L in BAND}, flush=True)
