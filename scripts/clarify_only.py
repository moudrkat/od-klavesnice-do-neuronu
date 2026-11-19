#!/usr/bin/env python3
# Jen clarify terminál, konzistentní upřesnění (coder only). Fialové téma.
import torch, glob, os
from transformers import AutoTokenizer, AutoModelForCausalLM
from ansitoimg.render import ansiToSVG
dev="cuda"; OUT="term_shots"
def snap(repo):
    for r in ["~/.cache/huggingface","~/projects/science/vibe-inside/hf"]:
        g=glob.glob(os.path.expanduser(r+"/hub/models--"+repo.replace("/","--")+"/snapshots/*"))
        if g: return g[0]
    return repo
sgr=lambda n:(lambda s:f"\x1b[{n}m{s}\x1b[0m")
C=sgr("36");Y=sgr("33");D=sgr("90");CB=sgr("1;36");DB=sgr("1;90")
I=snap("Qwen/Qwen2.5-Coder-1.5B-Instruct")
tok=AutoTokenizer.from_pretrained(I); m=AutoModelForCausalLM.from_pretrained(I,dtype=torch.bfloat16).to(dev).eval()
def gen(msgs,n=320):
    ids=tok(tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True),return_tensors="pt").input_ids.to(dev)
    with torch.no_grad(): o=m.generate(ids,max_new_tokens=n,do_sample=False,pad_token_id=tok.eos_token_id)
    return tok.decode(o[0,ids.shape[1]:],skip_special_tokens=True).strip()
WISH="Udělej mi appku na úkoly, ať vidím kolik jich mám hotových."
clars=["Prostě chci spočítat svoje úkoly.","Každej úkol je buď hotovej nebo nehotovej.","Chci vědět, kolik jich mám hotovejch. Ať to funguje!"]
turns=[{"role":"user","content":WISH}]
L=[DB("╭─ ")+CB("ruční upřesňování — kolo za kolem")+DB(" "+"─"*52), DB("│ ")+Y("vibecoder ▸ ")+WISH]
for cl in clars:
    ans=gen(turns); turns.append({"role":"assistant","content":ans})
    L.append(DB("│ ")+C("model ▸ ")+" ".join(ans.split()))
    L.append(DB("│ ")+Y("vibecoder ▸ ")+cl); turns.append({"role":"user","content":cl})
ans=gen(turns,320); L.append(DB("│ ")+C("model ▸ ")+" ".join(ans.split()))
L.append(DB("│ ")+D("# čtyři kola a pořád žádná čistá funkce — ruční upřesňování je otravné"))
L.append(DB("╰"+"─"*88))
open(f"{OUT}/clarify.ansi","w").write("\n".join(L))
ansiToSVG("\n".join(L),f"{OUT}/clarify.svg",theme="ubuntu.yml",width=100,title="orchestrate.py — aorus  ·  vibecoder bez orchestrace")
print("SAVED clarify (konzistentní)",flush=True)
