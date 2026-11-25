#!/usr/bin/env python3
# Revizor (4B) nad všemi 164 kódy drobka z HumanEval (humaneval_qwen15.json): CORRECT/WRONG.
# U WRONG: senior (4B) poradí jednou větou, drobek opraví, oprava jde na skutečné testy. Greedy, stejné prompty jako orchestrate.py.
import json, gzip, re, os, sys, ast, subprocess, tempfile, torch, glob
from transformers import AutoTokenizer, AutoModelForCausalLM
dev="cuda"
def snap(repo):
    for r in ["~/.cache/huggingface","~/projects/science/vibe-inside/hf"]:
        g=glob.glob(os.path.expanduser(r+"/hub/models--"+repo.replace("/","--")+"/snapshots/*"))
        if g: return g[0]
    return repo
def load(n): t=AutoTokenizer.from_pretrained(n); return t,AutoModelForCausalLM.from_pretrained(n,dtype=torch.bfloat16).to(dev).eval()
t4,m4=load(snap("Qwen/Qwen3-4B-Instruct-2507")); tC,mC=load(snap("Qwen/Qwen2.5-Coder-1.5B-Instruct"))
def gen(t,m,msgs,n):
    ids=t(t.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True),return_tensors="pt").input_ids.to(dev)
    with torch.no_grad(): o=m.generate(ids,max_new_tokens=n,do_sample=False,pad_token_id=t.eos_token_id)
    return t.decode(o[0,ids.shape[1]:],skip_special_tokens=True).strip()
VSYS="You are a reviewer. You get a task and a Python function. Reply CORRECT or WRONG and one short sentence why."
SSYS="You are a senior engineer. The junior's Python function is wrong for the task. In ONE short sentence, say concretely what to fix."
BSYS="You are a coding agent. Complete the given Python function. Reply with the complete function in one ```python block."
probs={json.loads(l)["task_id"]:json.loads(l) for l in gzip.open("HumanEval.jsonl.gz","rt")}
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
def test(p,code):
    if not static_ok(code): return "static-blocked"
    prog=p["prompt"].split("def ")[0]+"\n"+code+"\n\n"+p["test"]+f"\n\ncheck({p['entry_point']})\n"
    with tempfile.TemporaryDirectory() as d:
        f=os.path.join(d,"t.py"); open(f,"w").write(prog)
        try:
            r=subprocess.run([sys.executable,"-I",f],cwd=d,capture_output=True,timeout=10,text=True)
            return "pass" if r.returncode==0 else "fail: "+(r.stderr.strip().splitlines() or ["?"])[-1][:120]
        except subprocess.TimeoutExpired: return "fail: timeout"
res=json.load(open("humaneval_qwen15.json")); out=[]
for k,r in enumerate(res):
    p=probs[r["task_id"]]; code="```python\n"+r["code"]+"\n```"
    rev=gen(t4,m4,[{"role":"system","content":VSYS},{"role":"user","content":"Task:\n"+p["prompt"]+"\n\nCode:\n"+code}],60)
    o={"task_id":r["task_id"],"verdict":r["verdict"],"code":r["code"],"review":rev,"flag":not rev.upper().startswith("CORRECT")}
    if o["flag"]:
        hint=gen(t4,m4,[{"role":"system","content":SSYS},{"role":"user","content":"Task:\n"+p["prompt"]+"\n\nCode:\n"+code}],60)
        fix=gen(tC,mC,[{"role":"system","content":BSYS},{"role":"user","content":p["prompt"]},{"role":"assistant","content":code},{"role":"user","content":"Reviewer: "+hint+" Fix it. Code only."}],512)
        fc=extract(fix); o.update(hint=hint,fix=fc,fix_verdict=test(p,fc))
        rev2=gen(t4,m4,[{"role":"system","content":VSYS},{"role":"user","content":"Task:\n"+p["prompt"]+"\n\nCode:\n```python\n"+fc+"\n```"}],60)
        o["review2"]=rev2
    out.append(o); print(k,r["task_id"],r["verdict"][:4],"FLAG" if o["flag"] else "ok",o.get("fix_verdict","")[:30],flush=True)
json.dump(out,open("review_humaneval.json","w"),ensure_ascii=False,indent=1)
fails=[o for o in out if o["verdict"]!="pass"]; passes=[o for o in out if o["verdict"]=="pass"]
print("chyb:",len(fails),"revizor chytil:",sum(o["flag"] for o in fails),"z toho oprava prošla testy:",sum(o["flag"] and o.get("fix_verdict")=="pass" for o in fails))
print("dobrých:",len(passes),"planý poplach:",sum(o["flag"] for o in passes),"oprava pak spadla:",sum(o["flag"] and o.get("fix_verdict")!="pass" for o in passes))
print("REVDONE")
