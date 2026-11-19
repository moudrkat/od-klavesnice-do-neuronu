#!/usr/bin/env python3
# Čísla na slide přímo z modelu: config, počet parametrů (celkem / bez embeddingu / MLP / attention), tokeny/s na této kartě.
import json, time, torch, glob
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
I="Qwen/Qwen2.5-Coder-1.5B-Instruct"; dev="cuda" if torch.cuda.is_available() else "cpu"
cfg=AutoConfig.from_pretrained(I); c=cfg.to_dict()
keys=["num_hidden_layers","hidden_size","intermediate_size","num_attention_heads","num_key_value_heads","vocab_size","max_position_embeddings","tie_word_embeddings","torch_dtype","rope_theta"]
print("config:",{k:c.get(k) for k in keys},flush=True)
snap=glob.glob(f"hf/hub/models--Qwen--Qwen2.5-Coder-1.5B-Instruct/snapshots/*/")[0]
print("snapshot:",snap, "| soubory:",sorted(p.split('/')[-1] for p in glob.glob(snap+'*')),flush=True)
tok=AutoTokenizer.from_pretrained(I); m=AutoModelForCausalLM.from_pretrained(I,dtype=(torch.bfloat16 if dev=="cuda" else torch.float32)).to(dev).eval()
tot=sum(p.numel() for p in m.parameters()); emb=m.model.embed_tokens.weight.numel()
mlp=sum(p.numel() for n,p in m.named_parameters() if ".mlp." in n); attn=sum(p.numel() for n,p in m.named_parameters() if ".self_attn." in n)
head=0 if c.get("tie_word_embeddings") else m.lm_head.weight.numel()
print(f"parametry: celkem {tot:,} | embedding {emb:,} | lm_head {'sdílená s embeddingem' if head==0 else format(head,',')} | MLP {mlp:,} ({100*mlp/tot:.0f} %) | attention {attn:,} ({100*attn/tot:.0f} %)",flush=True)
print(f"MLP neuronů: {c['num_hidden_layers']} × {c['intermediate_size']} = {c['num_hidden_layers']*c['intermediate_size']:,} | attention hlav: {c['num_hidden_layers']} × {c['num_attention_heads']} = {c['num_hidden_layers']*c['num_attention_heads']}",flush=True)
print("velikost vah v bf16: %.2f GB"%(tot*2/1e9),flush=True)
print("GPU:",torch.cuda.get_device_name(0),"| VRAM %.0f GB"%(torch.cuda.get_device_properties(0).total_memory/1e9),flush=True)
# tokeny/s: jeden prompt, greedy, 256 nových tokenů, 3 opakování
SYS="You are a coding agent. Reply with Python code only, in one ```python block."
ids=tok(tok.apply_chat_template([{"role":"system","content":SYS},{"role":"user","content":"Write a function that parses a CSV line into a list of fields, handling quoted fields with commas."}],tokenize=False,add_generation_prompt=True),return_tensors="pt").input_ids.to(dev)
with torch.no_grad(): m.generate(ids,max_new_tokens=16,do_sample=False,pad_token_id=tok.eos_token_id)
rates=[]
for _ in range(3):
    torch.cuda.synchronize(); t0=time.time()
    with torch.no_grad(): g=m.generate(ids,max_new_tokens=256,min_new_tokens=256,do_sample=False,pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize(); rates.append((g.shape[1]-ids.shape[1])/(time.time()-t0))
print("tokeny/s (bf16, HF generate, 256 tokenů, 3 běhy): %s → medián %.0f"%([round(r) for r in rates],sorted(rates)[1]),flush=True)
json.dump({"config":{k:c.get(k) for k in keys},"params_total":tot,"params_embedding":emb,"params_mlp":mlp,"params_attn":attn,"tok_s":rates,"gpu":torch.cuda.get_device_name(0)},open("model_facts.json","w"))
print("MODELFACTSDONE",flush=True)
