import json, re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
dev="cuda" if torch.cuda.is_available() else "cpu"
def load(m,cache):
    import os; os.environ["HF_HOME"]=cache
    t=AutoTokenizer.from_pretrained(m); mm=AutoModelForCausalLM.from_pretrained(m,dtype=(torch.bfloat16 if dev=="cuda" else torch.float32)).to(dev).eval(); return t,mm
def gen(t,m,msgs,n=140):
    ids=t(t.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True),return_tensors="pt").input_ids.to(dev)
    with torch.no_grad(): o=m.generate(ids,max_new_tokens=n,do_sample=False,pad_token_id=t.eos_token_id)
    return t.decode(o[0,ids.shape[1]:],skip_special_tokens=True).strip()
OUT={}
WISH="Udělej mi appku na úkoly, ať vidím kolik jich mám hotových."
# 1) REFINER = 4B
RSYS="Jsi agent, který z přání uživatele vytáhne to hlavní, co chce vidět, a navrhne JEDNU Python funkci, která to spočítá. Odpověz přesně jedním řádkem: Napiš Python funkci nazev(argumenty): vrať ... . Název funkce česky a výstižně."
t4,m4=load("Qwen/Qwen3-4B-Instruct-2507","/home/kachna/projects/science/instruct-steer/hf-cache")
ref=gen(t4,m4,[{"role":"system","content":RSYS},{"role":"user","content":"chci vidět průměr známek"},{"role":"assistant","content":"Napiš Python funkci prumer_znamek(znamky): vrať průměr čísel v seznamu."},{"role":"user","content":WISH}],60)
OUT["refiner_sys"]=RSYS; OUT["wish"]=WISH; OUT["refiner_out"]=ref
del m4; torch.cuda.empty_cache()
# 2) BUILDER = 1.5B coder, s coding-agent promptem
BSYS="Jsi coding agent. Dostaneš zadání a napíšeš jednu Python funkci. Odpověz POUZE kódem v bloku ```python, bez vysvětlení."
tc,mc=load("Qwen/Qwen2.5-Coder-1.5B-Instruct","/home/kachna/projects/science/vibe-inside/hf")
build=gen(tc,mc,[{"role":"system","content":BSYS},{"role":"user","content":ref}])
OUT["builder_sys"]=BSYS; OUT["builder_out"]=build
# 3) REVIEWER = 1.5B
VSYS="Jsi reviewer. Tohle je kód od kodéra. Znáš zadání, ale neznáš očekávaný výstup. Rozhodni, jestli kód dělá to, co zadání zadá. Odpověz SPRÁVNĚ nebo ŠPATNĚ a jednou větou proč."
rev=gen(tc,mc,[{"role":"system","content":VSYS},{"role":"user","content":"ZADÁNÍ: "+ref+"\n\nKÓD:\n"+build}],60)
OUT["reviewer_sys"]=VSYS; OUT["reviewer_out"]=rev.replace(chr(10)," ")
# 4) DISSEKČNÍ PROMPT (minimální wrapper, code-only) + funkce
DTASK="Napiš Python funkci pocet_hotovych_ukolu(ukoly): vrať počet úkolů, které jsou označeny jako hotové. Odpověz POUZE kódem v bloku ```python, bez vysvětlení, bez textu."
code=gen(tc,mc,[{"role":"user","content":DTASK}])
OUT["dissect_task"]=DTASK; OUT["dissect_code"]=code
json.dump(OUT,open("orch_lock.json","w"),ensure_ascii=False,indent=1)
print("REFINER:",ref,flush=True); print("BUILDER:",build[:120],flush=True); print("REVIEWER:",OUT["reviewer_out"][:120],flush=True); print("DISSECT:",code[:120],flush=True); print("ORCHLOCKDONE",flush=True)
