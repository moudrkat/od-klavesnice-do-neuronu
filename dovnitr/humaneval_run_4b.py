#!/usr/bin/env python3
# HumanEval (164 úloh) na Qwen3-4B-Instruct-2507 (refiner/reviewer), greedy, stejný prompt a vyhodnocení jako u 1,5B. Testy běží v samostatném procesu s limitem,
# po statické kontrole (žádné os/subprocess/open/síť). Ukládá kód i verdikt každé úlohy.
import json, gzip, re, os, sys, ast, subprocess, tempfile, torch, urllib.request
from transformers import AutoTokenizer, AutoModelForCausalLM
P="HumanEval.jsonl.gz"
if not os.path.exists(P): urllib.request.urlretrieve("https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz",P)
probs=[json.loads(l) for l in gzip.open(P,"rt")]
import glob
def snap(repo):
    for r in ["~/.cache/huggingface","~/projects/science/vibe-inside/hf"]:
        g=glob.glob(os.path.expanduser(r+"/hub/models--"+repo.replace("/","--")+"/snapshots/*"))
        if g: return g[0]
    return repo
dev="cuda"; I=snap("Qwen/Qwen3-4B-Instruct-2507")
tok=AutoTokenizer.from_pretrained(I); tok.padding_side="left"
m=AutoModelForCausalLM.from_pretrained(I,dtype=torch.bfloat16).to(dev).eval()
SYS="You are a coding agent. Complete the given Python function. Reply with the complete function in one ```python block."
def chat(pr): return tok.apply_chat_template([{"role":"system","content":SYS},{"role":"user","content":pr}],tokenize=False,add_generation_prompt=True)
outs=[]
for i in range(0,len(probs),4):
    batch=[chat(p["prompt"]) for p in probs[i:i+4]]
    enc=tok(batch,return_tensors="pt",padding=True).to(dev)
    with torch.no_grad(): g=m.generate(**enc,max_new_tokens=512,do_sample=False,pad_token_id=tok.pad_token_id)
    outs+=[tok.decode(s[enc.input_ids.shape[1]:],skip_special_tokens=True) for s in g]
    print(f"{len(outs)}/{len(probs)}",flush=True)
BAD={"os","sys","subprocess","shutil","socket","urllib","requests","pathlib","open","exec","eval","__import__","compile","input"}
def extract(t):
    mm=re.search(r"```python(.*?)```",t,re.S) or re.search(r"```(.*?)```",t,re.S); return (mm.group(1) if mm else t).strip()
def static_ok(code):
    try: tree=ast.parse(code)
    except Exception: return False
    for n in ast.walk(tree):
        if isinstance(n,(ast.Import,ast.ImportFrom)):
            names=[a.name.split(".")[0] for a in n.names] if isinstance(n,ast.Import) else [(n.module or "").split(".")[0]]
            if any(x in BAD for x in names): return False
        if isinstance(n,ast.Name) and n.id in BAD: return False
        if isinstance(n,ast.Attribute) and n.attr.startswith("__"): return False
    return True
res=[]
for p,t in zip(probs,outs):
    code=extract(t); verdict="static-blocked"
    if static_ok(code):
        prog=p["prompt"].split("def ")[0]+"\n"+code+"\n\n"+p["test"]+f"\n\ncheck({p['entry_point']})\n"
        with tempfile.TemporaryDirectory() as d:
            f=os.path.join(d,"t.py"); open(f,"w").write(prog)
            try:
                r=subprocess.run([sys.executable,"-I",f],cwd=d,capture_output=True,timeout=10,text=True)
                verdict="pass" if r.returncode==0 else ("fail: "+(r.stderr.strip().splitlines() or ["?"])[-1][:120])
            except subprocess.TimeoutExpired: verdict="fail: timeout"
    res.append({"task_id":p["task_id"],"entry_point":p["entry_point"],"prompt":p["prompt"],"code":code,"verdict":verdict,"canonical":p["canonical_solution"]})
n=sum(r["verdict"]=="pass" for r in res); print(f"HumanEval pass@1 greedy: {n}/{len(res)} = {100*n/len(res):.1f}%  (publikováno 70,7 %)")
print("statický filtr zablokoval:",sum(r["verdict"]=="static-blocked" for r in res))
json.dump(res,open("humaneval_qwen3_4b.json","w"),ensure_ascii=False,indent=0); print("HEDONE")
