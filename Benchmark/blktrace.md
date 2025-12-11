# Blktrace Findings for 20251210_114450_llama-2-7b_50percent

What happened
- Blktrace did run and produced ~12GB of per-CPU files in `results/20251210_114450_llama-2-7b_50percent/blktrace/`.
- `blkparse` produced a huge `blktrace_parsed.txt` (~17GB).
- `access_pattern.json` was never created, so SUMMARY showed “No blktrace data captured”.

Root cause
- The analyzer read the wrong columns from `blkparse` output. It inspected column 5 (action `A/Q/M`) instead of column 6 (RWBS), so it discarded every line and thought there were zero reads.

Fix applied
- `utils/analyze_blktrace.py` now:
  - Reads the correct columns (RWBS=column 6, sector=7, size=9).
  - Streams input (does not load entire file into RAM).
  - Accepts `-` to read from stdin.
- `experiments.py` now pipes `blkparse` output directly into the analyzer to avoid giant intermediate files.

How to regenerate access_pattern.json for this run (may take several minutes)
```
cd /home/keri/BSC/9DecemberExperiments/results/20251210_114450_llama-2-7b_50percent
blkparse -i blktrace/trace -o - | python3 ../../utils/analyze_blktrace.py - access_pattern.json
```

After that completes, SUMMARY will have data (or you can re-run the experiment with the updated scripts).***
