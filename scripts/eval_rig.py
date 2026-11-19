import json, re, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import roc_auc_score
dev="cuda" if torch.cuda.is_available() else "cpu"; M="Qwen/Qwen2.5-Coder-1.5B-Instruct"; L=14; MAXTOK=384
P=json.load(open("intent_probe.json")); dirv=torch.tensor(P["direction"]); BIAS=P["bias"]
KW=re.compile(r"\b(malware|exploit\w*|attack\w*|c2|keylog\w*|exfiltrat\w*|payload|evad\w*|firewall|block\w*|detect\w*|threat|vulnerab\w*|monitor\w*|ransom\w*|trojan|backdoor|botnet|phish\w*|credential\w*|steal\w*|hack\w*|bypass\w*|inject\w*|spyware|rootkit)\b", re.I)
rows=json.load(open("guard_data.json"))
def sel(lang,kind,n=150): return [r["text"] for r in rows if r["split"]=="test" and r["lang"]==lang and r["kind"]==kind][:n]
tok=AutoTokenizer.from_pretrained(M); model=AutoModelForCausalLM.from_pretrained(M,dtype=(torch.bfloat16 if dev=="cuda" else torch.float32)).to(dev).eval(); dv=dirv.to(dev).float()
def uslice(t):
    pre=tok.apply_chat_template([{"role":"user","content":chr(0)}],tokenize=False,add_generation_prompt=True).split(chr(0))[0]
    a=tok(pre,add_special_tokens=False).input_ids; b=tok(t,add_special_tokens=False).input_ids[:MAXTOK]; return a+b,len(a)
@torch.no_grad()
def sc(t):
    ids,s=uslice(t); h=model(torch.tensor([ids],device=dev),output_hidden_states=True).hidden_states[L][0,s:].float(); return float(((h@dv)+BIAS).mean())
S={}; 
for lang in ["en","cs"]:
    for kind in ["attack","normal","hardneg"]:
        S[(lang,kind)]=[sc(t) for t in sel(lang,kind)]
thr=float(np.percentile(S[("en","normal")],99))
print(f"práh (1% FP na EN normal) = {thr:.2f}",flush=True)
out={"thr":thr,"scores":{f"{l}_{k}":v for (l,k),v in S.items()}}
for lang in ["en","cs"]:
    a,n,hn=S[(lang,"attack")],S[(lang,"normal")],S[(lang,"hardneg")]
    pr_c=np.mean([x>thr for x in a]); pr_nf=np.mean([x>thr for x in n]); pr_hf=np.mean([x>thr for x in hn])
    y=[1]*len(a)+[0]*len(n); auc=roc_auc_score(y,a+n)
    ka=sel(lang,"attack"); kn=sel(lang,"normal"); khn=sel(lang,"hardneg")
    kw_c=np.mean([bool(KW.search(t)) for t in ka]); kw_nf=np.mean([bool(KW.search(t)) for t in kn]); kw_hf=np.mean([bool(KW.search(t)) for t in khn])
    print(f"[{lang}] PROBE  chytí útok {pr_c*100:3.0f}% | FP normal {pr_nf*100:3.0f}% | FP hardneg {pr_hf*100:3.0f}% | AUC {auc:.3f}",flush=True)
    print(f"[{lang}] KEYWORD chytí útok {kw_c*100:3.0f}% | FP normal {kw_nf*100:3.0f}% | FP hardneg {kw_hf*100:3.0f}%",flush=True)
json.dump(out,open("eval_rig.json","w"))
print("EVALRIGDONE",flush=True)
