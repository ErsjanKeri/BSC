# Generation outputs (canonical sweep)

Deterministic generation outputs from the canonical thesis sweep, captured as evidence that the wall-clock numbers in `thesis/chapters/06_evaluation.tex` correspond to actual coherent inference and that the same `seed=42` produces bit-identical generation across all uring code paths.

| File | Source log | Model | Tokens | Eval time |
|---|---|---|---|---|
| `20b_multitopic_seed42.txt` | `time-tracking/results/cgroup_20260501_191023/lazy_20b_7g_run1.log` | GPT-OSS-20B-F16 | 1999 | 1387.40 s |
| `120b_multitopic_seed42.txt` | `time-tracking/results/cgroup_20260502_140144/lazy_120b_22g_run1.log` | GPT-OSS-120B-F16 | 1999 | 571.58 s |

**Prompt** (identical for both):

> Write a long detailed essay covering these five topics in sequence: (1) the history of computer science, (2) the history of mathematics, (3) the history of physics, (4) the history of astronomy, (5) the history of biology. For each topic, cover ancient origins, medieval developments, 19th century advances, 20th century breakthroughs, and modern applications. Include specific dates, scientist names, and their contributions.

**Run flags** (identical for both, modulo per-config flags):

```
-n 2000 -ngl 0 -no-cnv --no-warmup --eager-compute --seed 42
```

The 1999 token count comes from the model emitting an end-of-generation token at index 1999 (the 2000th tokens-to-generate request is preempted). Both models produce the full essay; neither early-stops with a refusal under this prompt.

The MD5 of the generation region (post-prompt-echo, pre-perf-summary) matches across `--uring-projection-overlap`, `--uring-async-projection-overlap`, and `--uring-async-experts` runs at the same seed for each model. This is the byte-level pipeline correctness guarantee asserted in `thesis/chapters/05_implementation.tex`.
