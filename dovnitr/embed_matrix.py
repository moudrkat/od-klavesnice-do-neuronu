#!/usr/bin/env python3
# Celá embedding matice drobka (embed_tokens.weight, 151 936 × 1 536), ilustrativně: každý 200. řádek, skutečná čísla, uloženo jako float16 .npy.
# K tomu: je svázaná s výstupem (lm_head)? a řádky 470 (return) a 7985 (Write) celé.
import json, numpy as np, torch
from transformers import AutoModelForCausalLM
I="Qwen/Qwen2.5-Coder-1.5B-Instruct"; m=AutoModelForCausalLM.from_pretrained(I,dtype=torch.bfloat16)
E=m.model.embed_tokens.weight.detach().float()
tied=bool(m.lm_head.weight.data_ptr()==E.data_ptr()) or bool(torch.equal(m.lm_head.weight.detach().float(),E))
S=200; sub=E[::S].numpy().astype(np.float16); np.save("embed_matrix_sub.npy",sub)
json.dump({"shape":list(E.shape),"stride":S,"sub_shape":list(sub.shape),"tied":tied,"absmax":round(float(E.abs().max()),4),"std":round(float(E.std()),5),
           "row_470":[round(float(x),4) for x in E[470]],"row_7985":[round(float(x),4) for x in E[7985]]},open("embed_matrix.json","w"))
print("tabulka",tuple(E.shape),"| svázaná s výstupem:",tied,"| každý",S,". řádek →",sub.shape,"| |E| max",round(float(E.abs().max()),4),"std",round(float(E.std()),5)); print("EMBMATDONE",flush=True)
