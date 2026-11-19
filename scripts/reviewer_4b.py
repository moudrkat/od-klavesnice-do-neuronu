import json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
dev="cuda"; I="Qwen/Qwen3-4B-Instruct-2507"
L=json.load(open("orch_lock.json")); REF=L["refiner_out"]; CODE=L["builder_out"]
RSYS="You are a code reviewer. You get a task and a Python function. Decide if the function correctly does the task. Reply with CORRECT or WRONG and one short sentence why."
t=AutoTokenizer.from_pretrained(I); m=AutoModelForCausalLM.from_pretrained(I,dtype=torch.bfloat16).to(dev).eval()
user="Task: "+REF+"\n\nCode:\n"+CODE
ids=t(t.apply_chat_template([{"role":"system","content":RSYS},{"role":"user","content":user}],tokenize=False,add_generation_prompt=True),return_tensors="pt").input_ids.to(dev)
with torch.no_grad(): o=m.generate(ids,max_new_tokens=80,do_sample=False,pad_token_id=t.eos_token_id)
rev=t.decode(o[0,ids.shape[1]:],skip_special_tokens=True).strip()
json.dump({"reviewer_model":"Qwen3-4B-Instruct-2507","reviewer_sys":RSYS,"reviewer_out_4b":rev},open("reviewer_4b.json","w"),ensure_ascii=False,indent=1)
print("REVIEWER 4B:",rev,flush=True); print("REV4BDONE",flush=True)
