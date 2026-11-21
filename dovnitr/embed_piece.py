#!/usr/bin/env python3
# Kousek embedding matice doslova: řádky 464–475 (kolem ` return` = 470), prvních 10 z 1 536 sloupců, s tokeny vedle. Výstup: embed_piece.json
import json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
I="Qwen/Qwen2.5-Coder-1.5B-Instruct"; t=AutoTokenizer.from_pretrained(I); m=AutoModelForCausalLM.from_pretrained(I,dtype=torch.bfloat16)
E=m.model.embed_tokens.weight.detach().float(); rows=list(range(464,476))
out={"rows":[{"id":r,"tok":t.decode([r]),"vals":[round(float(x),3) for x in E[r][:10]]} for r in rows],"cols":10,"shape":list(E.shape)}
json.dump(out,open("embed_piece.json","w"),ensure_ascii=False)
for r in out["rows"]: print(r["id"],repr(r["tok"]),r["vals"][:5])
print("PIECEDONE",flush=True)
