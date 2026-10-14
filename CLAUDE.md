# Instructions for the coding agent

You are helping Rakesh Reddy run a controlled experiment for a Vizuara SLM
Bootcamp research project. Read this before touching anything.

## What this project is, and is not

**Is:** a careful measurement that settles a disagreement in the literature.
Two published results conflict about whether small language models degrade more
than large ones with long context. The deliverable is one coefficient with a
confidence interval, plus the harness and the reporting recipe that make it
checkable.

**Is not:** a new method. Do not propose architecture changes, new attention
mechanisms, fine-tuning recipes or prompt tricks as the contribution. If a
mitigation gets tested (for example recite-then-answer), it is tested as an
*ablation across the ladder*, never as the headline.

The single most important property of this codebase is that a condition changes
exactly one thing. If a change you are about to make causes two things to vary
at once, stop and say so.

## Layout

| Path | Role |
| --- | --- |
| `configs/*.yaml` | the whole experiment definition; nothing experimental is hardcoded in `src/` |
| `src/stressor.py` | builds prompts; the four dials live here |
| `src/metrics.py` | scoring and the two degradation formulas |
| `src/evaluate.py` | the sweep loop; append-only, resumable results |
| `src/analyze.py` | aggregation, bootstrap CIs, the regression |
| `src/plots.py` | figures from real results |
| `experiments/log_template.md` | one block per run; rows map onto the paper tables |

## Conventions

- Python 3.10+, type hints on function signatures, no classes where a function
  will do.
- Every experimental knob goes in a config file, never a literal in `src/`. If
  you need a new knob, add it to the dataclasses in `config.py` as well.
- `results.jsonl` is append-only and keyed by
  `(model, task, condition, seed, uid)`. Never rewrite it in place; never
  reorder its columns. Resumability depends on that key.
- Accuracies are on a 0 to 100 scale everywhere outside `metrics.py`.
- Any number that reaches the paper must come with an interval. A point
  estimate alone is not a result.
- Comments explain *why*, not *what*. Match the density already in the files.
- No new dependencies without saying what breaks if we skip it.

## Rules specific to this experiment

1. **Byte-identical invariance.** The question, the evidence sentences and the
   instruction wording must be identical across every condition for a given
   item. Only the padding changes. If a refactor risks breaking this, add an
   assertion instead of trusting it.
2. **Never assume the floor.** `a_floor` comes from the no-evidence control
   rows of the same sweep. If `floor_source` says `assumed_zero`, `D_hn` is not
   reportable; fix the run rather than the metric.
3. **Frontier API models never enter the size regression.** Their parameter
   counts are unknown. They draw a reference curve and nothing else.
4. **Report both metrics, always.** `D_rel` is what prior work reports, so it
   is how we connect to them; `D_hn` is what makes the comparison fair. Showing
   only one invites the exact confusion we are trying to resolve.
5. **Tokenizer-matched lengths.** Target lengths are realised with each model's
   own tokenizer. Check `actual_tokens` in the results when a curve looks odd.

## Current state

Implemented and import-clean: config, data loaders, stressor, HF and API
runners, metrics, the sweep loop, aggregation with bootstrap CIs, the OLS fit,
and plotting.

Not yet implemented, roughly in priority order:

1. **Prompt paraphrases.** `stressor.INSTRUCTION` is a single template. The
   design calls for three. Add a paraphrase index to `Condition`, thread it
   through `record_key`, and treat it as a grouping variable in `analyze.py`.
2. **Long-form constrained generation task.** The `output_length > 1` path
   exists in `stressor.py` but the scorer only reads the first `ANSWER:` line.
   It needs per-item scoring to reproduce the generation-length result.
3. **Paired per-item tests.** `analyze.py` bootstraps over items but does not
   yet do the paired comparison between two specific rungs.
4. **Headroom matching.** The subset analysis restricted to items where two
   models have equal short-context accuracy.
5. **Code completion task.** Third task in the paper's design, not started.

## How to help well

- When asked for a change, first say which dial or control it affects and
  whether it breaks a comparison already in `results.jsonl`.
- Prefer small, runnable steps: a change that cannot be smoke-tested in ten
  minutes is too big.
- After any change to `stressor.py` or `metrics.py`, run
  `bash scripts/smoke_test.sh` and read the first raw generation before saying
  it works.
- If a result looks surprising, your first hypothesis should be a bug in the
  harness, not a discovery about language models. Suggest the check that would
  distinguish them.
