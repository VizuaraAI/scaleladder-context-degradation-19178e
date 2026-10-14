# Experiment log

One block per run. Copy the template, fill it in the same day you run it, and
never overwrite an old block. The columns map one to one onto Table 1 and
Table 3 of the paper draft, so a complete log turns into the results section
with almost no extra work.

A run that you cannot describe in this format is a run you cannot put in a
paper. If a field is "I am not sure", that is the thing to fix next.

---

## Run template

**Run ID:** `<run_name>` (matches `runs/<run_name>/`)
**Date:**
**Commit:** `git rev-parse --short HEAD`
**Why this run exists (one sentence):**
**What I expected to see before looking:**

### Setup

| Field | Value |
| --- | --- |
| Models | e.g. SmolLM2 135M / 360M / 1.7B |
| Task | multihop_qa / math_wordproblems |
| Input lengths | 1K, 4K, 16K, 32K |
| Filler type | whitespace / in_domain_shuffled / hard_distractor |
| Evidence position | start / middle / end |
| Output length G | 1 |
| Seeds | 1234, 1235, 1236 |
| Prompt paraphrases | 1 of 3 |
| Items per cell | |
| GPU and wall clock | |

### Results, paper Table 1 rows

| Model | Params | 1K | 4K | 16K | 32K | a_floor | D_rel | D_hn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |  |  |

### Results, paper Table 2 row

| Score | Dial | beta(log10 params) | 95% CI | adj R^2 | Verdict |
| --- | --- | --- | --- | --- | --- |
| d_rel | input length |  |  |  |  |
| d_hn | input length |  |  |  |  |

### Retrieval decomposition

| Model | retrieval hit rate at 1K | at 32K | reasoning failures despite retrieval |
| --- | --- | --- | --- |
|  |  |  |  |

### Reading the run

- What actually happened:
- Did it match what I expected? If not, which assumption broke:
- Confounds I can rule out after this run:
- Confounds still open:
- Next run this implies:

### Sanity checks (tick before trusting anything above)

- [ ] `actual_tokens` is within 5% of the requested length for every condition
- [ ] every cell has the full number of items (no silent truncation)
- [ ] `floor_source` is `measured`, not `assumed_zero`
- [ ] at least one condition was eyeballed: I read 5 raw generations myself
- [ ] short-context accuracy is sane (a 1.7B model should not be at 2% on 1K)
- [ ] the CI width is small enough that the comparison I am making is possible

---

## Run 001

(copy the template above and start here)
