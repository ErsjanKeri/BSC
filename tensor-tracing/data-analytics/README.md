# Expert Activation Data Analytics

Statistical analysis of MoE expert activation patterns across multiple domains.

## Dataset

**Source**: `../expert-analysis-2026-01-26/`
- **5 domains**: code, math, creative, factual, mixed
- **100 tokens per domain** = 500 tokens total
- **72 MoE operations per token** (24 layers × 3 components)
- **4 experts selected per operation** (top-4 routing)
- **Total**: ~144,000 expert activations

## Structure

```
data-analytics/
├── notebooks/
│   └── 01_expert_activation_analysis.ipynb  # Main analysis notebook
├── outputs/
│   ├── figures/                              # PNG/PDF for thesis
│   └── tables/                               # CSV results
└── README.md                                 # This file
```

## Quick Start

```bash
cd data-analytics/notebooks

# Install dependencies (if needed)
pip install jupyter pandas numpy matplotlib seaborn scipy scikit-learn

# Launch Jupyter
jupyter notebook 01_expert_activation_analysis.ipynb
```

## Key Analyses

### 1. Domain × Expert Frequency Heatmap
Shows which experts are used by which domains (per layer, per component).

**Research Question**: Do domains activate different expert subsets?

### 2. Statistical Significance Testing
Chi-square test to verify domain clustering is not random.

### 3. Expert Classification
Identify specialist experts (domain-specific) vs generalist experts (used everywhere).

### 4. Layer-wise Analysis
Compare early layers (0-5) vs late layers (18-23) for pattern differences.

### 5. Temporal Autocorrelation
Test if expert at token t predicts expert at token t+1 (stickiness).

## Model Architecture

**GPT-OSS-20B MoE Structure**:
- 24 transformer layers
- Each layer has 3 FFN components:
  - `ffn_down_exps`: 32 experts (2880×2880 each)
  - `ffn_gate_exps`: 32 experts
  - `ffn_up_exps`: 32 experts
- Top-4 routing: Selects 4 out of 32 experts per operation
- Each component's experts are INDEPENDENT (different parameters)

**Total expert parameters**: 2,304 expert weight tensors (24 × 3 × 32)

## Data Schema

**DataFrame columns** (after extraction):
- `domain`: str (domain-1-code, domain-2-math, ...)
- `token_id`: int (0-99)
- `layer_id`: int (0-23)
- `component`: str (ffn_down, ffn_gate, ffn_up)
- `expert_id`: int (0-31, which expert was selected)
- `position`: int (0-3, ranking in top-4)

## Expected Insights

**If domain clustering exists**:
- Different domains will show different expert usage patterns
- Chi-square test will show p < 0.001
- Specialist experts can be identified

**If temporal correlation exists**:
- Expert at token t will predict expert at t+1
- Lag-1 autocorrelation > 20% (baseline: 3.125% if random)

**If layer differences exist**:
- Early layers may show different patterns than late layers
- Some layers may have stronger domain clustering than others
