import json, re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
dev="cuda" if torch.cuda.is_available() else "cpu"; M="Qwen/Qwen2.5-Coder-1.5B-Instruct"
L=json.load(open("orch_lock.json")); BSYS=L["builder_sys"]; REF=L["refiner_out"]
tok=AutoTokenizer.from_pretrained(M); m=AutoModelForCausalLM.from_pretrained(M,dtype=(torch.bfloat16 if dev=="cuda" else torch.float32)).to(dev).eval()
PROMPT=tok.apply_chat_template([{"role":"system","content":BSYS},{"role":"user","content":REF}],tokenize=False,add_generation_prompt=True)
ids=tok(PROMPT,return_tensors="pt").input_ids.to(dev)
with torch.no_grad(): o=m.generate(ids,max_new_tokens=120,do_sample=False,pad_token_id=tok.eos_token_id)
gen=tok.decode(o[0,ids.shape[1]:],skip_special_tokens=True)
mm=re.search(r"return",gen); prefix=PROMPT+gen[:mm.end()]
pids=tok(prefix,return_tensors="pt").input_ids.to(dev)
with torch.no_grad(): lg=torch.softmax(m(pids).logits[0,-1].float(),-1)
top=torch.topk(lg,6); TOP=[(tok.decode([t]),round(p*100,1)) for p,t in zip(top.values.tolist(),top.indices.tolist())]
ut=tok(REF,add_special_tokens=False)
json.dump({"prompt":PROMPT,"code":gen,"n_user_tokens":len(ut.input_ids),"user_tokens":[tok.decode([t]) for t in ut.input_ids],"top_after_return":TOP},open("rem_core.json","w"),ensure_ascii=False,indent=1)
print("CODE:",re.search(r"def .*",gen).group(0) if re.search(r"def .*",gen) else gen[:60],flush=True)
print("USER TOKENS:",len(ut.input_ids),"| TOP after return:",TOP,flush=True)
