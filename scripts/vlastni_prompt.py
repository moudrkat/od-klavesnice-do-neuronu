#!/usr/bin/env python3
# Tvoje zadání → kód od buildera (1,5B, teplota 0) → zapsáno do orch_lock.json (refiner_out, builder_out).
# Pak lekce 3, 4, 5 běží na tvém promptu. Zadání piš anglicky a chtěj funkci, která něco vrací (lekce koukají na slovo `return`).
import json, re, sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
request = " ".join(sys.argv[1:]).strip()
assert request, 'použití: python vlastni_prompt.py "Write a Python function ..."'
dev = "cuda" if torch.cuda.is_available() else "cpu"
L = json.load(open("orch_lock.json"))
I = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
t = AutoTokenizer.from_pretrained(I); m = AutoModelForCausalLM.from_pretrained(I, dtype=(torch.bfloat16 if dev == "cuda" else torch.float32)).to(dev).eval()
tmpl = t.apply_chat_template([{"role": "system", "content": L["builder_sys"]}, {"role": "user", "content": request}], tokenize=False, add_generation_prompt=True)
ids = t(tmpl, return_tensors="pt").input_ids.to(dev)
with torch.no_grad():
    g = m.generate(ids, max_new_tokens=160, do_sample=False, pad_token_id=t.eos_token_id)
code = t.decode(g[0, ids.shape[1]:], skip_special_tokens=True).strip()
print(code)
body = re.search(r"```(?:python)?\s*(.*?)```", code, re.S); body = body.group(1) if body else code
assert "return" in body, "lekce se dívají na slovo `return`; chtěj funkci, která něco vrací"
L["refiner_out"] = request; L["builder_out"] = code
json.dump(L, open("orch_lock.json", "w"), ensure_ascii=False, indent=1)
print("\nV orch_lock.json je teď tvůj prompt.")
