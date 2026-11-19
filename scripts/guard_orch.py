import json, re, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import roc_auc_score
dev="cuda" if torch.cuda.is_available() else "cpu"; M="Qwen/Qwen2.5-Coder-1.5B-Instruct"; L=14; MAXTOK=384
P=json.load(open("intent_probe.json")); DIR=torch.tensor(P["direction"]); BIAS=P["bias"]
BSYS=json.load(open("orch_lock.json"))["builder_sys"]  # builder context (orchestrace)
rows=json.load(open("guard_data.json"))
def sel(lang,kind,n=80): return [r["text"] for r in rows if r["split"]=="test" and r["lang"]==lang and r["kind"]==kind][:n]
tok=AutoTokenizer.from_pretrained(M); m=AutoModelForCausalLM.from_pretrained(M,dtype=(torch.bfloat16 if dev=="cuda" else torch.float32)).to(dev).eval(); dv=DIR.to(dev).float()
def uslice(text):
    pre=tok.apply_chat_template([{"role":"system","content":BSYS},{"role":"user","content":chr(0)}],tokenize=False,add_generation_prompt=True).split(chr(0))[0]
    a=tok(pre,add_special_tokens=False).input_ids; b=tok(text,add_special_tokens=False).input_ids[:MAXTOK]; return a+b,len(a)
@torch.no_grad()
def sc(text):
    ids,s=uslice(text); h=m(torch.tensor([ids],device=dev),output_hidden_states=True).hidden_states[L][0,s:].float(); return float(((h@dv)+BIAS).mean())
S={(lang,kind):[sc(t) for t in sel(lang,kind)] for lang in ["en","cs"] for kind in ["attack","normal","hardneg"]}
thr=float(np.percentile(S[("en","normal")],99))
print("prah (1% FP na EN normal, BUILDER kontext) =",round(thr,2),flush=True)
for lang in ["en","cs"]:
    a,n,hn=S[(lang,"attack")],S[(lang,"normal")],S[(lang,"hardneg")]
    auc=roc_auc_score([1]*len(a)+[0]*len(n),a+n)
    print("["+lang+"] PROBE v orchestraci: chyti utok",round(np.mean([x>thr for x in a])*100),"% | FP normal",round(np.mean([x>thr for x in n])*100),"% | FP hardneg",round(np.mean([x>thr for x in hn])*100),"% | AUC",round(auc,3),flush=True)
json.dump({"thr":thr,"ctx":"builder_system"},open("guard_orch.json","w"))
print("GUARDORCHDONE")
