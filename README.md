# scaleladder-context-degradation

Starter repository for Rakesh Reddy, Vizuara SLM Bootcamp.

Two published results about long-context degradation disagree. One reports
Llama-3-8B losing 47.1% of its score while a frontier model loses 1.2%. Another
reports every model, small and frontier alike, losing a comparable fraction
(starting around 13% and going much higher) even when retrieval is perfect.
This repository is the experiment that decides between them.

The core idea: those two studies compared models that differ in everything at
once (size, data, context extension, tokenizer, serving stack), and both divided
the accuracy drop by a short-context baseline that differs by tens of points
across the models being compared. Here we vary parameter count **within** a
model family, hold the task instances byte-identical across conditions, and
report degradation both the conventional way and normalised by the headroom the
model actually had.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Datasets (HotpotQA, GSM8K, WikiText-103) download from the Hugging Face Hub the
first time you run. Nothing large is committed here. Gated checkpoints such as
Llama and Gemma need `huggingface-cli login` first.

## The first three things to run

```bash
# 1. Prove the plumbing works. One 135M model, two lengths, ten items.
bash scripts/smoke_test.sh

# 2. Look at what the model actually saw and said, with your own eyes.
head -n 1 runs/smoke/results.jsonl | python -m json.tool | head -40

# 3. Run one real ladder. Start with SmolLM2 (three rungs, small enough to be honest about).
bash scripts/run_sweep.sh configs/default.yaml
```

Then read `runs/default/scale_fit.txt`. The coefficient on `log10(params)` with
its confidence interval is the result the whole project turns on.

## Layout

```
configs/
  default.yaml      the real sweep: ladders, tasks, the four stress dials
  smoke.yaml        the ten-minute version
src/
  config.py         typed config loading
  data.py           HotpotQA, GSM8K and the filler corpus
  stressor.py       the context stress generator: four independent dials
  models.py         local Hugging Face runner plus an API runner for reference curves
  metrics.py        answer extraction, exact match, retrieval probe, D_rel and D_hn
  evaluate.py       the sweep; writes runs/<name>/results.jsonl (resumable)
  analyze.py        aggregation, bootstrap CIs, and the regression on log10(params)
  plots.py          the figures that replace Figure 2 of the draft
scripts/
  smoke_test.sh     run this first
  run_sweep.sh      the real thing, resumable, run it under tmux
experiments/
  log_template.md   one block per run; the rows map onto the paper's tables
```

## The four stress dials

Everything in `stressor.py` exists to move exactly one of these at a time while
the question, the evidence sentences and the instruction wording stay
byte-identical:

| Dial | Values | Why it is separate |
| --- | --- | --- |
| input length `L` | 1K, 4K, 16K, 32K, 128K | the input-length reading of the contradiction |
| filler type | whitespace, in-domain prose, hard distractors | distinguishes length from interference |
| evidence position | start, middle, end | position effects may themselves depend on size |
| output length `G` | 1, 8, 32, 128 | the generation-length reading of the contradiction |

Plus two controls that the analysis cannot do without, both produced by the same
sweep: a **retrieval probe** (recite the evidence, then answer) and a
**no-evidence floor** (the same prompt with the answer-bearing sentences
deleted). The floor is the denominator of `D_hn`. Measure it, never assume it.

## The two metrics

```
D_rel = (a_short - a_long) / a_short                      what prior work reports
D_hn  = (a_short - a_long) / (a_short - a_floor)          what we add
```

`D_rel` is a ratio whose denominator varies by 40 accuracy points across a
single ladder. A small model that starts near the floor cannot show a big
relative drop; a strong model has a long way to fall. `D_hn` asks the question
that is actually comparable: what fraction of the accuracy this model *had* did
length destroy?

## Compute notes

The default config is deliberately small. Scale up in this order, checking the
results file between each step: more items per cell, then more seeds, then more
lengths, then more rungs. Going straight to 128K on a 32B model is how week two
disappears. A 24GB GPU comfortably handles everything up to 7B at 32K; past
that, use a rented A100 for a single focused sweep rather than a long one.
