# From the keyboard to a neuron

You type a wish. A crew of small models turns it into code. 87 tokens go into a model, 28 layers later
one word comes out. Inside, one attention head and one neuron decide between `sum` and `len`, and you
can move them. In production nobody moves them: a probe reads the stream instead.

This is my talk from the Vibecoding & AI Engineering Festival (Ostrava, 26 Nov 2026) as seven short
lessons you can do yourself. One model, **Qwen2.5-Coder-1.5B-Instruct**, temperature 0, one gaming GPU.
Every number on the slides has a script and a JSON here.

## Start here, no install

- **[Layer 20, hands on](deck/vrstva20.html)**: turn head 20·6 and neuron 20·3545 up and down, watch the
  code change. Pre-recorded, runs in the browser.
- **[The slides](deck/deck.html)** (arrow keys) · **[PDF](deck.pdf)**
- **[Run the lessons in Colab](https://colab.research.google.com/github/moudrkat/od-klavesnice-do-neuronu/blob/main/lessons.ipynb)**:
  free GPU, ten minutes, the same numbers as on the slides.

No GPU at home: the scripts fall back to CPU (slow, float32, last digits may differ). And every script's
output is committed next to it, so you can read the numbers without running anything.

## Setup, if you run it yourself

```
pip install torch transformers scikit-learn numpy
```

Models download on first use: `Qwen/Qwen2.5-Coder-1.5B-Instruct` (3 GB), and for the refiner and reviewer
`Qwen/Qwen3-4B-Instruct-2507` (8 GB). Run each script from its own folder. `orch_lock.json` holds the one
session the whole talk is about: the glued system prompt, the refined request, the builder's code.

## The lessons

**1 · The crew** · *Parta, Zvládne to parta nejmenších modelů?*
Alone, the 1.5B model asks four rounds of questions and writes nothing. Refiner (4B) → builder (1.5B) →
reviewer (4B), and it works; on `int_to_roman` the reviewer catches the builder's bug.
`cd scripts && python orchestrate.py -i`. Try your own wish. Remove the reviewer and count what slips through.

**2 · How small the little one is** · *Drobeček, Co umí a co ne*
HumanEval: 1.5B solves 110/164, 4B 142/164. Frontier models stopped reporting it. 28 layers, a stream of
1 536 numbers, 8 960 neurons per layer, 12 heads, 151 936 tokens, 1.54 B parameters.
`cd dovnitr && python humaneval_run.py`, `python model_facts.py`. Read the failures: ours crashes on `sort_array([])`.

**3 · What goes in** · *Tvůj prompt neletí sám, Co jde dovnitř*
87 tokens enter, 23 are yours. The rest is the system prompt the harness glued on, the template, and the
model's own code so far.
`python token_strips.py`, `python attn_split.py`. Change the system prompt in `orch_lock.json`, watch the shares move.

**4 · The layers, and the next word** · *Tohle jsou vrstvy, …a jak se z 1 536 čísel stane další slovo*
At `return`, if the model stopped after layer 1 it would say `不了`; after 14, `NotImplementedError`;
after 21, `count`; after 28, `sum` at 79 %. Then the whole prompt plus `sum` goes in again.
`python stream_walk.py`. Try another position in the code.

**5 · Layer 20, and moving it** · *Vrstva 20 zblízka, Hýbu hlavou a neuronem*
Head 20·6 puts 72 % of its attention on your request, 0.46 on the word `list`. Neuron 20·3545 fires 17
at `return`. Head off: sum 79 → 73 %, same code. Neuron off: `len` 62 %, it writes `return len([...])`.
Neuron ×4: `tasks` 88 %, it writes `tasks.count(True)`.
`python heads_matter.py`, `python layer20.py`, `python play20_grid.py`.
Honest note: found by scanning 8 960 neurons on this one prompt. Neurons are not stable addresses.

**6 · Add a direction, get JavaScript** · *Přičti směr, Výzkum 2025–2026*
Direction = stream at a `function` signature minus stream at a `def` signature, added to layers 6–18,
coefficient 6: 3 of 3 prompts come out as JavaScript (missing the `i++`). Less: Python. More: garbage.
Research 2025–2026 steers code models toward secure code and reads agents from the inside; reading works,
steering is not ready. That is why production reads.
`cd scripts && python lang_retest.py`. Papers: arXiv 2604.16697, 2606.22673, 2507.09508, 2510.02917, 2610.07722.

**7 · The probe** · *Probe: pravítko na proudu, Guard to vidí, Rizikový požadavek nevypadá rizikově*
One direction in the 1 536 numbers at layer 14, averaged over your tokens, one threshold. Trained with
logistic regression on labelled requests (harmful ones rewritten so no scary word is left). "Route traffic
through TOR to anonymize C2 traffic": no forbidden word, the probe flags it, the builder never runs. On 100
such requests a keyword filter catches 51, the probe about 98. Not finished security. A window.
`python guard_orch.py`, `python pertoken.py`, `python eval_rig.py` (needs `results/probe.json` from
[hidden-intent](https://github.com/moudrkat/hidden-intent), the training repo, saved as `scripts/intent_probe.json`).

## Four things to try on your own setup

1. What gets glued to your prompt: `ollama show MODEL --modelfile`, or OpenCode's prompts in its repo.
2. What it reads from your request: `pip install brainscope`, a small model in the browser, stream and heads visible.
3. What a second model catches: have the code reviewed by another role before you run it.
4. Whether it survives the edge case: put an empty list in the request.

## Layout

`deck/` slides and the interactive page · `scripts/` the crew, the guard, the direction experiment ·
`dovnitr/` ("inside") benchmarks, layers, layer 20 · `videa/` the probe catching the TOR request ·
`REPRODUKCE.md` every slide → its script → its JSON.

Code MIT, slides and videos CC BY 4.0. Kateřina Fajmanová, 2026.
