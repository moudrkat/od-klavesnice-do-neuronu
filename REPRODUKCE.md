# Reprodukce — slide → skript → data (stav 9. 11. 2026)

Kořen: toto repo. Skripty běží na aorusu v `~/projects/science/vibe-inside/` (venv `~/tmp/vllm-lens-test/.venv`), model Qwen2.5-Coder-1.5B-Instruct, teplota 0. Deck za běhu nečte žádný JSON; načítá jen `assets/term/clarify.png`, `orch.png`, `fix.png`, `tor.png`, `assets/defense.gif`, `assets/pendulum_one_cz.gif`. Čísla jsou v HTML vepsaná; tahle tabulka říká, odkud jsou.

| slide | číslo / tvrzení | skript | data |
|---|---|---|---|
| Vágní přání → model tápe | 4 kola ručního upřesňování | `scripts/clarify_only.py`, render `scripts/term_capture_v3.py` | `deck/assets/term/clarify.svg` → `clarify.png` |
| Zvládne to parta | refiner → builder → reviewer CORRECT | `scripts/orch_lock.py`, `scripts/reviewer_4b.py` | `scripts/orch_lock.json` (reviewer 1,5B tam říká WRONG), `scripts/reviewer_4b.json` (4B: CORRECT, to je na slidu) |
| A takhle to běží | skutečný běh | `scripts/orchestrate.py`, render `scripts/term_final.py` | `scripts/orch_cli.log` → `scripts/term_orch.html` → `orch.png` |
| Builder chybuje → oprava | int_to_roman WRONG → CORRECT | `scripts/orchestrate.py` | `scripts/orch_fix3.log` → `scripts/term_fix.html` → `fix.png` |
| Drobeček | 110/164 HumanEval; 96 tok/s; 28 vrstev, 1 536, 8 960, 12 hlav, 151 936 | `dovnitr/humaneval_run.py`, `dovnitr/model_facts.py`; tok/s bez skriptu (měřeno ručně) | `dovnitr/humaneval_qwen15.json`, `dovnitr/model_facts.json` |
| Co umí a co ne | 110/164, 45/6/2/1, tři ukázky | `dovnitr/humaneval_run.py` | `dovnitr/humaneval_qwen15.json` |
| Tvůj prompt neletí sám | chat template, token id | tokenizer (`apply_chat_template`, `encode`) | `scripts/rem_core.json` (tokeny) |
| 28 vrstev · jedna vrstva | 28, 1 536, 12, 8 960 | config modelu | — |
| Uhodni další token | sum 79 % · len 14 % · tasks 7 % | `scripts/rem_core_fix.py` | `scripts/rem_core.json` (78,8 / 13,7 / 7,3) |
| Co jde dovnitř (87 tokenů) | skupiny tokenů; pozornost z kódu: 17/18/17/18/30 % | `dovnitr/attn_split.py`, `dovnitr/full_context.py` | `dovnitr/attn_split.json`, `dovnitr/full_context.json` |
| Jak se z čísel stane slovo | vektor, matice 151 936 × 1 536, skóre, pravděpodobnosti, smyčka | `dovnitr/play20_grid.py` (next_word), `dovnitr/stream_walk.py` | `dovnitr/play20_grid.json`, `dovnitr/stream_walk.json` |
| Vrstva 20: hlava 20·6 | pozornost z return přes 87 tokenů; 72 % do zadání; list 0,46; vypnutí 79→73 | `dovnitr/full_context.py`, `dovnitr/heads_matter.py`, `dovnitr/play20_grid.py` | `dovnitr/full_context.json`, `dovnitr/heads_matter.json`, `dovnitr/play20_grid.json` |
| Neuron 20·3545, krok 2 (sken) | všech 8 960 neuronů ×0: jen 3545 −41,1; 5191 −6,5; neuron 1507 (akt −1,0) ×0 i ×4 → 78,8 %, kód stejný | `dovnitr/neuron_scan.py` | `dovnitr/neuron_scan.json`, obrázek `deck/assets/neuron_scan.png` |
| Umí jedinou věc · teplota (4 kroky) | po `return` při T 0,1/1/1,5/2: sum 100/79/63/34 %, entropie 0,00/0,96/1,93/7,70 b; vzorky kódu seed 0–4 | `dovnitr/temperature.py` | `dovnitr/temperature.json` |
| Vrstva 20 zblízka, krok 2 | líná hlava 20·2 (×0: sum 78,8 → 79,5 %, kód stejný; 27 % pozornosti na první token), 20·6 ×0: 72,7 %, 20·9 ×0: 72,5 %; matice 87×87 pro 20·6 a 20·2 | `dovnitr/head_null.py` | `dovnitr/head_null.json`, obrázky `deck/assets/attn_20_6.png`, `attn_20_2.png` (jas = pozornost, strop 0,5; pruhy = skupiny tokenů) |
| Neuron 20·3545 | aktivace přes 87 tokenů; sken vrstvy 20; −41 b. | `dovnitr/full_context.py`, `dovnitr/layer20.py` | `dovnitr/full_context.json`, `dovnitr/layer20.json` |
| Hýbu hlavou a neuronem | mřížka dávek 5×5, p + kód | `dovnitr/play20_grid.py` | `dovnitr/play20_grid.json`, interaktivně `deck/vrstva20.html` |
| Výzkum 2025–2026: sáhnout do kódu | obrázky z paperů: arXiv 2604.16697 (Surgical Repair), 2606.22673 (Agent MechSuits, NeurIPS 2026), 2507.09508 (MoC), 2510.02917 (SAE code correctness, ICLR 2026); ústně 2610.07722 (SteerScope) | — | `deck/assets/papers/` |
| Filtr na slova (51 ze 100) | regex 24 bezpečnostních slov (malware, exploit, attack, c2, keylog…, viz `KW` v eval_rig.py) proti probe | `scripts/eval_rig.py` (aorus) | `scripts/eval_rig.json` |
| HumanEval 4B parťák (142/164) | stejný běh na Qwen3-4B-Instruct-2507 | `dovnitr/humaneval_run_4b.py` | `dovnitr/humaneval_qwen3_4b.json` |
| Tohle to dělá (pásy tokenů) | aktivace 14·2825/7845/8212 na zadání, 22·8350 na kódu | `dovnitr/token_strips.py` | `dovnitr/token_strips.json` |
| Četl si moje zadání? hlava 16·4 | 69 % do zadání; `['`→completed 0,51; vypnutí nic; max hlava 7,6 b. | `dovnitr/attn_prompt.py`, `dovnitr/circuit_layer.py` | `dovnitr/attn_prompt.json`, `dovnitr/circuit_layer.json` |
| Přičti směr → JavaScript | 3/3, coef 6, vrstvy 6–18, chybí i++ | `scripts/lang_retest.py` (směr + 3 prompty), `scripts/lang_slide.py` (faktoriál na slidu) | `scripts/lang_slide.json` |
| Probe: pravítko na proudu | vrstva 14, práh = 1 planý poplach ze 100 | `scripts/eval_rig.py` | `scripts/eval_rig.json` (thr 1,40) |
| Guard naživo (TOR) | ÚTOK (0,3), builder se nespustí | `scripts/orchestrate.py`, práh `scripts/guard_orch.py` | `scripts/guard_orch.json` (thr −0,24) → `term/tor.png`; vyžaduje probe `results/probe.json` z repa hidden-intent (na aorusu, 2 MB) |
| Probe chytí → rozmělní (GIF) | gated steering, model se rozkecá | `scripts/capture_deflect.py` (builder GIFu je ztracený, popis ve VYSLEDKY §7) | `scripts/deflect.json`, `deck/assets/defense.gif` |
| Refusal je tenký štít | 38 % → 0 %; probe ~98 % | `scripts/direction_causal.py` | `scripts/direction_causal.json`, `scripts/eval_rig.json` |
| Nevypadá rizikově (heatmap) | skóre po tokenech; 51 / ~98 / ~1; 40 % → 80 % | `scripts/pertoken.py`; `scripts/eval_rig.py`; adversariální test popsán ve VYSLEDKY §6 (skript v2/v3 probe na aorusu: trénink v repu hidden-intent (`pipeline/3_train.py`), ) | `scripts/pertoken.json`, `scripts/eval_rig.json` (51 % jen v stdout a v poznámky v hidden-intent README) |
| Co si můžete zkusit zjistit | sort_array([]) pád | `dovnitr/humaneval_run.py` | `dovnitr/humaneval_qwen15.json` |

## Mimo deck (VYSLEDKY §4: rodiny neuronů)
`dovnitr/n249_study.py`, `dovnitr/n249_layers.py`, `dovnitr/n249_tokens.py`, `dovnitr/n249_pendulum.py`, `dovnitr/capture_249_other.py`, `dovnitr/family_ablate.py`, `dovnitr/family_rep.py`, `dovnitr/family_rep2.py`, `dovnitr/bench_ablate.py` a jejich JSONy.

## Známé díry
- probe `results/probe.json` z repa hidden-intent, `guard_data*.json` jsou jen na aorusu (velké / trénovací data).
- Builder `defense.gif` se ztratil s pracovní složkou; data (`scripts/deflect.json`) a postup jsou.
- 96 tok/s je ruční měření bez skriptu.
