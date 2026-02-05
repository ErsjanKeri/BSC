#!/bin/bash
#
# Master script to collect expert activation data across multiple domains
#
# Usage: ./run_expert_experiments.sh
#
# This script will:
# 1. Read prompts from prompts.json
# 2. Run llama-completion for each prompt (100 tokens each)
# 3. Parse all traces (binary → JSON)
# 4. Organize by domain
# 5. Create tarball for transfer
#
# Estimated time: 30-60 minutes for 500 tokens total
#

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load configuration
echo "===================================================================="
echo "Expert Activation Pattern Data Collection"
echo "===================================================================="
echo ""

# Read experiment name from prompts.json
EXPERIMENT_NAME=$(python3 -c "import json; print(json.load(open('prompts.json'))['experiment_name'])")
OUTPUT_BASE="experiments/${EXPERIMENT_NAME}"

echo "Experiment: $EXPERIMENT_NAME"
echo "Output directory: $OUTPUT_BASE"
echo ""

# Create output directory structure
mkdir -p "$OUTPUT_BASE"

# Get number of prompts
NUM_PROMPTS=$(python3 -c "import json; print(len(json.load(open('prompts.json'))['prompts']))")

echo "Total prompts to process: $NUM_PROMPTS"
echo ""

# Process each prompt
for i in $(seq 0 $(($NUM_PROMPTS - 1))); do
    echo "===================================================================="
    echo "Prompt $((i+1))/$NUM_PROMPTS"
    echo "===================================================================="

    # Extract prompt data
    PROMPT_ID=$(python3 -c "import json; print(json.load(open('prompts.json'))['prompts'][$i]['id'])")
    DOMAIN=$(python3 -c "import json; print(json.load(open('prompts.json'))['prompts'][$i]['domain'])")
    PROMPT_TEXT=$(python3 -c "import json; print(json.load(open('prompts.json'))['prompts'][$i]['prompt'])")
    N_PREDICT=$(python3 -c "import json; print(json.load(open('prompts.json'))['prompts'][$i]['n_predict'])")

    echo "Domain: $DOMAIN"
    echo "Prompt ID: $PROMPT_ID"
    echo "Tokens to generate: $N_PREDICT"
    echo "Prompt: ${PROMPT_TEXT:0:80}..."
    echo ""

    # Create domain output directory
    DOMAIN_DIR="$OUTPUT_BASE/$PROMPT_ID"
    mkdir -p "$DOMAIN_DIR"

    # Run experiment for this prompt
    echo "Running inference..."
    python3 run_experiment.py \
        --prompt "$PROMPT_TEXT" \
        --n-predict $N_PREDICT \
        --output-dir "$DOMAIN_DIR" \
        --domain "$DOMAIN"

    echo ""
    echo "✓ Completed: $PROMPT_ID"
    echo ""
    echo "--------------------------------------------------------------------"
    echo ""
done

echo "===================================================================="
echo "All experiments completed!"
echo "===================================================================="
echo ""

# Create summary
echo "Creating summary..."
python3 << 'PYTHON'
import json
import os
from pathlib import Path

exp_name = json.load(open('prompts.json'))['experiment_name']
base_dir = f'experiments/{exp_name}'

summary = {
    'experiment': exp_name,
    'date': '2026-01-26',
    'domains': []
}

for domain_dir in Path(base_dir).glob('domain-*'):
    domain_info = {
        'id': domain_dir.name,
        'trace_files': len(list((domain_dir / 'traces').glob('*.json'))) if (domain_dir / 'traces').exists() else 0,
        'size_mb': sum(f.stat().st_size for f in domain_dir.rglob('*') if f.is_file()) / (1024*1024)
    }
    summary['domains'].append(domain_info)

with open(f'{base_dir}/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"Summary written to {base_dir}/summary.json")
PYTHON

# Create tarball for transfer
echo ""
echo "Creating tarball for transfer..."
TARBALL="${EXPERIMENT_NAME}.tar.gz"
cd experiments
tar -czf "../$TARBALL" "$EXPERIMENT_NAME"
cd ..

TARBALL_SIZE=$(du -h "$TARBALL" | cut -f1)
echo "✓ Created: $TARBALL ($TARBALL_SIZE)"
echo ""

echo "===================================================================="
echo "Data collection complete!"
echo "===================================================================="
echo ""
echo "To transfer data:"
echo "  scp $TARBALL your-local-machine:~/Downloads/"
echo ""
echo "To extract locally:"
echo "  tar -xzf $TARBALL"
echo "  # Data will be in: $EXPERIMENT_NAME/"
echo ""