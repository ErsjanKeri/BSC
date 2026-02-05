# Expert Activation Pattern Analysis - Experimental Plan

## Objective

Collect tensor traces across multiple prompt domains to analyze:
1. **Temporal correlation**: Do expert selections show time-series patterns?
2. **Domain clustering**: Do different domains (code, math, creative) activate different expert subsets?
3. **Stickiness**: Are certain experts repeatedly activated (autocorrelation)?

## Data Collection Plan

### Prompts (5 Domains)

All prompts configured in `prompts.json`:

1. **Code Generation**: Binary search tree implementation (Python)
2. **Mathematical Reasoning**: Train distance problem with step-by-step solution
3. **Creative Writing**: AI discovering friendship (narrative with sensory details)
4. **Factual Knowledge**: CRISPR gene editing explanation
5. **Mixed Domain**: Algorithm comparison + code implementation

Each prompt generates **100 tokens** → Total: **500 tokens**

### Expected Data Volume

- **Per prompt**: ~170 MB (100 tokens × 1.7 MB/token)
- **Total**: ~850 MB raw data
- **Compressed**: ~400-500 MB (for transfer)

### Server Execution (TUM Server)

```bash
cd ~/BSC/tensor-tracing

# Run all 5 experiments sequentially
./run_expert_experiments.sh

# This will:
# 1. Read prompts from prompts.json
# 2. Run llama-completion for each (100 tokens each)
# 3. Parse binary traces → JSON
# 4. Organize by domain:
#    experiments/expert-analysis-2026-01-26/
#      domain-1-code/
#      domain-2-math/
#      domain-3-creative/
#      domain-4-factual/
#      domain-5-mixed/
# 5. Create tarball: expert-analysis-2026-01-26.tar.gz

# Estimated time: 30-60 minutes
```

### Transfer to Local

```bash
# On server
scp expert-analysis-2026-01-26.tar.gz your-local-machine:~/Downloads/

# On local machine
cd ~/Downloads
tar -xzf expert-analysis-2026-01-26.tar.gz
mv expert-analysis-2026-01-26 ~/Public/LLAMA/BSC/desktopui/data/

# Now you can analyze with Desktop UI
```

## Analysis Plan (After Data Collection)

### Phase 1: Basic Statistics

Extract expert IDs from all traces:

```python
import json
from pathlib import Path
from collections import Counter

# Load all traces
all_expert_data = []

for domain_dir in Path('experiments/expert-analysis-2026-01-26').glob('domain-*'):
    domain_name = domain_dir.name

    for token_file in (domain_dir / 'traces').glob('token-*.json'):
        with open(token_file) as f:
            trace = json.load(f)

        for entry in trace['entries']:
            if entry['operation_type'] == 'MUL_MAT_ID' and entry['expert_ids']:
                all_expert_data.append({
                    'domain': domain_name,
                    'token_id': entry['token_id'],
                    'layer_id': entry['layer_id'],
                    'expert_ids': entry['expert_ids'][:4],  # Top-4
                    'timestamp_ms': entry['timestamp_relative_ms']
                })

# Basic stats
print(f"Total MoE operations: {len(all_expert_data)}")
print(f"Domains: {set(d['domain'] for d in all_expert_data)}")
```

### Phase 2: Domain × Expert Frequency Matrix

```python
# Build matrix: Domain × Expert → Activation Count
import numpy as np

domains = ['domain-1-code', 'domain-2-math', 'domain-3-creative',
           'domain-4-factual', 'domain-5-mixed']
n_experts = 32

freq_matrix = np.zeros((len(domains), n_experts))

for data in all_expert_data:
    domain_idx = domains.index(data['domain'])
    for expert_id in data['expert_ids']:
        freq_matrix[domain_idx, expert_id] += 1

# Normalize by domain (percentage)
freq_matrix = freq_matrix / freq_matrix.sum(axis=1, keepdims=True) * 100

# Show heatmap (in Desktop UI or matplotlib)
```

### Phase 3: Temporal Autocorrelation

```python
from scipy.stats import pearsonr

# For each layer, build expert sequence across tokens
for layer in range(24):
    expert_sequence = []

    for token_id in range(100):
        # Get expert IDs used at this token for this layer
        experts_at_token = [d['expert_ids'] for d in all_expert_data
                            if d['token_id'] == token_id and d['layer_id'] == layer]
        expert_sequence.append(experts_at_token[0] if experts_at_token else [])

    # Compute autocorrelation with lag=1
    # If expert at token t predicts expert at token t+1: high correlation
```

### Phase 4: Visualization in Desktop UI

Add new view: **Expert Analysis Panel**
- Heatmap: Domain (rows) × Expert ID (columns)
- Time series: Expert activation over token sequence
- Transition matrix: P(expert_t+1 | expert_t)

## Output Structure

```
experiments/expert-analysis-2026-01-26/
├── domain-1-code/
│   ├── tensor_trace.bin
│   ├── traces/
│   │   ├── token-00000.json
│   │   ├── token-00001.json
│   │   └── ... (100 files)
│   ├── graphs/
│   └── buffer-timeline.json
├── domain-2-math/
│   └── ... (same structure)
├── domain-3-creative/
├── domain-4-factual/
├── domain-5-mixed/
└── summary.json  ← Metadata about experiment
```

## Next Steps

1. ✅ Created `prompts.json` (5 domains, 100 tokens each)
2. ✅ Created `run_expert_experiments.sh` (automated data collection)
3. ⏸️ Run on server: `./run_expert_experiments.sh`
4. ⏸️ Transfer data locally
5. ⏸️ Analyze expert patterns (Phase 1-4 above)

---

**Ready to run on server!** Just execute `./run_expert_experiments.sh` and wait ~60 minutes.