#!/usr/bin/env python3
# Pokračování ZA `return` (greedy) při zásahu: stejný prefix jako pravděpodobnosti na slidu Hýbu (prompt + kód až po `return`), čtyři stavy: bez zásahu, hlava ×0, neuron ×0, neuron ×4. Výstup: play20_continue_heads.json
import json, re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
dev="cuda" if torch.cuda.is_available() else "cpu"; I="Qwen/Qwen2.5-Coder-1.5B-Instruct"; LZ=20; HD=6; NR=3545
Lk=json.load(open("orch_lock.json")); BSYS=Lk["builder_sys"]; USER=Lk["refiner_out"]; CODE0=Lk["builder_out"]
t=AutoTokenizer.from_pretrained(I); m=AutoModelForCausalLM.from_pretrained(I,dtype=(torch.bfloat16 if dev=="cuda" else torch.float32)).to(dev).eval()
H=m.config.num_attention_heads; D=m.config.hidden_size//H
tmpl=t.apply_chat_template([{"role":"system","content":BSYS},{"role":"user","content":USER}],tokenize=False,add_generation_prompt=True)
ids=t(tmpl,return_tensors="pt").input_ids.to(dev)
mm=re.search(r"```(?:python)?\s*(.*?)```",CODE0,re.S); code0=(mm.group(1) if mm else CODE0).strip()
pref=t(tmpl+"```python\n"+code0[:code0.index("return")+len("return")],return_tensors="pt").input_ids.to(dev)
def head_hook(f):
    def pre(mod,args):
        x=args[0].clone(); x[...,HD*D:(HD+1)*D]=x[...,HD*D:(HD+1)*D]*f; return (x,)
    return m.model.layers[LZ].self_attn.o_proj.register_forward_pre_hook(pre)
def neuron_hook(f):
    def pre(mod,args):
        x=args[0].clone(); x[...,NR]=x[...,NR]*f; return (x,)
    return m.model.layers[LZ].mlp.down_proj.register_forward_pre_hook(pre)
out=[]
for name,hf,nf in (("bez zásahu",1,1),("hlava ×0",0,1),("hlava ×2",2,1),("hlava ×4",4,1),("hlava ×0 + neuron ×4",0,4)):
    hs=([] if hf==1 else [head_hook(hf)])+([] if nf==1 else [neuron_hook(nf)])
    with torch.no_grad():
        lg=m(pref).logits[0,-1].float(); p=torch.softmax(lg,-1); top=torch.topk(p,4)
        g=m.generate(pref,max_new_tokens=40,do_sample=False,pad_token_id=t.eos_token_id)
    for h in hs: h.remove()
    cont=t.decode(g[0,pref.shape[1]:],skip_special_tokens=True); cont=cont.split("```")[0].rstrip()
    out.append({"state":name,"head":hf,"neuron":nf,"p":[(t.decode([int(i)]),round(float(v)*100,1)) for v,i in zip(top.values,top.indices)],"after_return":cont})
    print(name,"| p:",out[-1]["p"][:3],"| return"+cont.replace("\n","⏎"),flush=True)
json.dump({"prefix_ends_with":"return","items":out},open("play20_continue_heads.json","w"),ensure_ascii=False,indent=1); print("CONTDONE",flush=True)
