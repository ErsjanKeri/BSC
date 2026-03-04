# MAP_POPULATE Impact Study

Systematic measurement of the impact of `MAP_POPULATE` flag on LLM loading time and inference speed.

## Research Question

Does disabling `MAP_POPULATE` (lazy on-demand paging) improve:
1. Model loading time?
2. Memory efficiency for models that exceed available RAM?
3. Overall inference performance?

## Experimental Design

### 2x2 Matrix

| Model Size | MAP_POPULATE=ON | MAP_POPULATE=OFF |
|------------|-----------------|------------------|
| **Small (20B, 13GB)** <br/> Fits in 30GB RAM | Exp 1: Baseline | Exp 2: Lazy load |
| **Large (120B, 65GB)** <br/> Exceeds 30GB RAM | Exp 3: May thrash | Exp 4: On-demand |

### Predictions

**Small Model (Fits in RAM)**:
- Exp 1: ~2s load, fast inference
- Exp 2: ~1.7s load (faster!), same inference speed

**Large Model (Exceeds RAM)**:
- Exp 3: Very long load or OOM/thrashing
- Exp 4: Fast load, slower inference but usable

## Structure

```
time-tracking/
├── run_experiments.py    # Main experiment runner
├── utils.py              # Helper functions (cleanup, parsing, stats)
├── settings.json         # Configuration
├── results/              # Timestamped experiment results
│   └── experiment_YYYYMMDD_HHMMSS/
│       ├── config.json
│       ├── exp1_small_prefetch_on.csv
│       ├── exp2_small_prefetch_off.csv
│       ├── exp3_large_prefetch_on.csv
│       ├── exp4_large_prefetch_off.csv
│       └── run_*.log
└── README.md
```

## Usage

### Run All Experiments (Small + Large)

```bash
cd ~/BSC/time-tracking
python3 run_experiments.py
```

### Run Only Small Model Experiments

```bash
python3 run_experiments.py --small-only
```

### Run Only Large Model Experiments

```bash
python3 run_experiments.py --large-only
```

## Configuration

Edit `settings.json` to configure:

- **Models**: Which models to test
- **Iterations**: Number of runs per experiment (default: 10)
- **Prompt**: Test prompt
- **Tokens**: Number of tokens to generate (default: 100)
- **Cleanup**: Whether to drop page cache between runs
- **Cooldown**: Seconds to wait between runs (default: 5)

## Prerequisites

1. **Two llama.cpp builds** with different MAP_POPULATE settings:
   - `~/llama.cpp/build_prefetch_on/` - Compiled with `init_mappings(true, ...)`
   - `~/llama.cpp/build_prefetch_off/` - Compiled with `init_mappings(false, ...)`

2. **Models**:
   - Small: `~/llama.cpp/models/gpt-oss-20b-F16.gguf` (12.85 GB)
   - Large: `~/llama.cpp/models/gpt-oss-120b/gpt-oss-120b-F16.gguf` (65.4 GB)

## Output

Each experiment produces:

- **CSV file**: Raw timing data for all runs
- **Log files**: Full llama-completion output for each run
- **Statistics**: Mean, median, stdev, min, max for load/eval times

## Analysis

The script automatically calculates:
- Mean load time ± standard deviation
- Mean eval time ± standard deviation
- Min/max values
- Number of successful runs

## Expected Results

**For small model (13GB < 30GB RAM)**:
- MAP_POPULATE=OFF should be ~20% faster at loading
- Inference speed should be identical

**For large model (65GB > 30GB RAM)**:
- MAP_POPULATE=ON should fail, thrash, or be extremely slow
- MAP_POPULATE=OFF should work via on-demand paging

## Technical Details

### What is MAP_POPULATE?

`MAP_POPULATE` is a flag to `mmap()` that causes:
- **ON**: Eager page faulting - entire model loaded into RAM at startup (blocking)
- **OFF**: Lazy paging - pages loaded on-demand as accessed (non-blocking)

### Code Location

**File**: `llama.cpp/src/llama-model.cpp`, line 6797

```cpp
// MAP_POPULATE=ON (default)
ml.init_mappings(true, use_mlock ? &pimpl->mlock_mmaps : nullptr);

// MAP_POPULATE=OFF (modified)
ml.init_mappings(false, use_mlock ? &pimpl->mlock_mmaps : nullptr);
```

This calls `llama-mmap.cpp:388`:
```cpp
if (prefetch) { flags |= MAP_POPULATE; }
addr = mmap(NULL, file->size(), PROT_READ, flags, fd, 0);
```
