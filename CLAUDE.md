# Pro agenty (Claude Code, OpenCode, Cursor, Codex…)

Tohle repo jsou materiály k přednášce „Od klávesnice do neuronu“ jako sedm lekcí. Uživatel chce lekce
projít a pochopit, co se děje uvnitř malého modelu na kód. Tvoje role: spouštět, číst výstupy, vysvětlovat
mu čísla jeho slovy, a nic nevymýšlet. Všechno tady je naměřené při teplotě 0; když číslo nevyjde stejně,
řekni to, nezaokrouhluj to k číslu v README.

## Jak to pustit

- Jedna lekce = jeden příkaz z kořene repa: `./lekce 3`. Seznam: `./lekce` bez argumentu.
- Instalace: `./lekce setup` (torch, transformers, scikit-learn, numpy). Modely `Qwen/Qwen2.5-Coder-1.5B-Instruct`
  (3 GB) a pro lekci 1 `Qwen/Qwen3-4B-Instruct-2507` (8 GB) se stáhnou z Hugging Face při prvním spuštění.
- GPU není nutné: skripty se samy přepnou na CPU (pomalé, float32, poslední číslice se můžou lišit). Lekce 2
  (164 úloh HumanEval) a lekce 1 (model 4B) jsou na CPU na hodiny; řekni to uživateli předem a nabídni Colab
  (`lessons.ipynb`) nebo čtení hotových JSON.
- Skripty čtou vstupy z vlastní složky (`scripts/`, `dovnitr/`) a zapisují JSON vedle sebe. `./lekce` to řeší.
- Výstup každého skriptu z přednášky už je v repu jako JSON. Když uživatel nemá čas nebo hardware, čti JSON
  a vysvětli mu, co v něm je. Jmenuje se stejně jako skript.

## Mapa lekcí

| lekce | skripty | výsledek, který má uživatel vidět |
|---|---|---|
| 1 Parta | `scripts/orchestrate.py -i` | refiner (4B) zpřesní přání → builder (1,5B) napíše kód → reviewer (4B) zkontroluje |
| 2 Drobek | `dovnitr/model_facts.py`, `humaneval_run.py` | 28 vrstev, 1 536, 8 960 neuronů, 12 hlav; HumanEval 110/164 |
| 3 Co jde dovnitř | `dovnitr/token_strips.py`, `attn_split.py` | 87 tokenů, 23 uživatelovy; pozornost z kódu 17/18/17/18/30 % |
| 4 Vrstvy | `dovnitr/stream_walk.py` | prvních 8 čísel proudu po vrstvách; `不了` → `NotImplementedError` → `count` → `sum` 79 % |
| 5 Vrstva 20 | `dovnitr/heads_matter.py`, `layer20.py`, `play20_grid.py` | hlava 20·6 (72 % na zadání, 0,46 na `list`), neuron 20·3545 (17 na `return`); dávky → `len` / `sum` / `tasks.count` |
| 6 Směr | `scripts/lang_retest.py` | směr `function` − `def`, vrstvy 6–18, koef 6 → JavaScript 3/3 |
| 7 Probe | `scripts/pertoken.py`, `guard_orch.py`, `eval_rig.py` | TOR/C2 požadavek bez zlého slova: filtr 51/100, probe ~98; potřebuje `scripts/intent_probe.json` z repa hidden-intent |

## Vlastní prompt uživatele

`./lekce prompt "Write a Python function longest_word(words): return the longest string in the list."`
Builder napíše kód, zapíše se do `orch_lock.json` (`refiner_out`, `builder_out`) v obou složkách, a lekce
3, 4, 5 proběhnou na tom promptu. Zadání musí být anglicky a funkce musí něco vracet (lekce koukají na slovo
`return`). Zpátky na session z přednášky: `./lekce reset`.

## Úkoly „Zkus“ z README, kde co změnit

- Lekce 2: jméno modelu je řetězec `I="Qwen/Qwen2.5-Coder-1.5B-Instruct"` nahoře v `dovnitr/humaneval_run.py`.
- Lekce 3: `builder_sys` a `refiner_out` v `orch_lock.json` (obě složky, nebo `./lekce prompt`).
- Lekce 4: pozice se bere z `code0.index("return")` ve `dovnitr/stream_walk.py`; pro `def` nahraď oba výskyty `"return"` v tom řádku.
- Lekce 5: dávky jsou dva cykly `for hf in (0,0.5,1,2,4)` / `for nf in (0,0.5,1,2,4)` v `dovnitr/play20_grid.py`; přidej hodnoty do cyklu `nf`. Pozor: `deck/vrstva20.html` má mřížku 5×5 napevno, nové dávky se tam neobjeví.
- Lekce 6: seznam koeficientů `for coef in [2.0,4.0,6.0,8.0]` ke konci `scripts/lang_retest.py`.
- Lekce 7: požadavek je řetězec `TXT=` nahoře v `scripts/pertoken.py`.

## Čeho se nedotýkat

- JSONy s výsledky z přednášky (`*.json` vedle skriptů, `deck/`, `REPRODUKCE.md`): když uživatel chce svůj
  běh, ulož ho pod jiným jménem nebo ho nech přepsat vědomě. Původní jde vrátit přes `git checkout`.
- Neměň teplotu ani sampling: všechno je greedy, proto jsou čísla opakovatelná.
- Nic nedopisuj do čísel. Když se výsledek liší od README (jiná karta, CPU, jiná verze transformers), ukaž obojí.
