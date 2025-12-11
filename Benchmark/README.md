# SSD-Backed LLM Inference Experiments

Measure LLM inference performance with models on SSD vs RAM.

## Usage

```bash
sudo python3 experiments.py
```

That's it! One command does everything:
- Creates swap automatically
- Compiles tools
- Runs all experiments
- Generates SUMMARY.md for each experiment

## Customize Experiments

Edit the array in `experiments.py` (line 42):

```python
EXPERIMENTS = [
    # (model, ssd_percent, swappiness, tokens, name)
    ("llama-2-7b", 0, 0, 100, "baseline"),
    ("llama-2-7b", 50, 100, 100, "50percent"),
    ("llama-2-7b", 100, 100, 100, "100percent"),
]
```

Add/remove/modify experiments as needed.

## Structure

```
experiments.py       # Main script - everything in one place
utils/
  ├── monitor_system.sh      # System monitoring
  └── analyze_blktrace.py    # I/O pattern analysis
results/
  └── TIMESTAMP_model_scenario/
      ├── SUMMARY.md          # Human-readable summary
      ├── config.json         # Experiment parameters
      ├── inference_metrics.json
      ├── blktrace/
      ├── page_faults.log
      └── memory.csv, cpu.csv, etc.
```

## Requirements

- Python 3
- Sudo access
- blktrace, bpftrace, iostat, bc
- llama.cpp compiled
