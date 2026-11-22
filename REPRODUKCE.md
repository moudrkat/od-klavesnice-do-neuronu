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
| Co jde dovnitř, kroky 2–3 (slovník) | kousek řádků 464–475 × 10 sloupců; celá matice každý 200. řádek; svázaná s lm_head: True; std 0,0173, max 0,295 | `dovnitr/embed_piece.py`, `dovnitr/embed_matrix.py` | `dovnitr/embed_piece.json`, `dovnitr/embed_matrix.json`, `deck/assets/embed_matrix.png` |
| Co jde dovnitř, krok 4 (matice 87 × 1 536) | řádek return == řádek 470 slovníku (ověřeno), Write řádek 7985; |x0| max 0,098 | `dovnitr/input_matrix.py` | `dovnitr/input_matrix.json`, `deck/assets/input_matrix.png` |
| Co se ve vrstvě děje (2 kroky) | rozměry z configu: hidden 1 536, 12 hlav × 128, 2 KV hlavy (256), MLP 8 960; obrázky = hlava 20·6 (head_null.json) a neuron 20·3545 (full_context.json) | `dovnitr/model_facts.py` | `deck/assets/attn_20_6.png`, `deck/assets/neuron_strip.png` |
| …a teplota (4 kroky) | po `return` při T 0/1/1,5/2 (krok „teplota 0“ = greedy; sloupce spočítané při T = 0,1 jako limit, sum 100,0 %): sum 100/79/63/34 %, top 15 slov + zbytek slovníku, entropie 0,00/0,96/1,93/7,70 b; vzorky kódu seed 0–4 | `dovnitr/temperature.py` | `dovnitr/temperature.json` |
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

| Bubliny u vrstev: skóre vítěze | top-3 skóre (logity) po vrstvách 1–28 na pozici return | `dovnitr/lens_scores.py` | `dovnitr/lens_scores.json` |
| Hraní: za return napíše | greedy pokračování za `return` při 4 stavech, stejný prefix jako p | `dovnitr/play20_continue.py` | `dovnitr/play20_continue.json` |
| Teplota 3 / 5 / 10, celý vektor skóre, proud po vrstvě 28 | `temperature.json` 9 teplot + vzorky; `temperature_logits.npy` 151 936 skóre; `stream28_return.npy` | `dovnitr/temperature.py` | `dovnitr/temperature.json`, `.npy` |
| Směr doslova | směr function − def, vrstvy 6–18, norma 1 | `dovnitr/direction_dump.py` | `dovnitr/direction_L6..18.npy` |
