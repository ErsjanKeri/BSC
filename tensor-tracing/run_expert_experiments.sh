#!/bin/bash
#
# Automated expert activation data collection
# Uses existing run_experiment.py + settings.json workflow
#
# Usage: ./run_expert_experiments.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "===================================================================="
echo "Expert Activation Pattern Data Collection"
echo "===================================================================="
echo ""

# Backup original settings.json
cp settings.json settings.json.backup
echo "✓ Backed up settings.json → settings.json.backup"
echo ""

# Create experiment directory
EXPERIMENT_NAME="expert-analysis-2026-01-26"
OUTPUT_BASE="experiments/${EXPERIMENT_NAME}"
mkdir -p "$OUTPUT_BASE"

echo "Output directory: $OUTPUT_BASE"
echo ""

# Read prompts from prompts.json
NUM_PROMPTS=$(python3 -c "import json; print(len(json.load(open('prompts.json'))['prompts']))")

echo "Total prompts: $NUM_PROMPTS"
echo ""

# Process each prompt
for i in $(seq 0 $(($NUM_PROMPTS - 1))); do
    echo "===================================================================="
    echo "Experiment $((i+1))/$NUM_PROMPTS"
    echo "===================================================================="

    # Extract prompt data using Python (pass $i as argument)
    PROMPT_ID=$(python3 -c "import json; print(json.load(open('prompts.json'))['prompts'][$i]['id'])")
    DOMAIN=$(python3 -c "import json; print(json.load(open('prompts.json'))['prompts'][$i]['domain'])")
    PROMPT_TEXT=$(python3 -c "import json; print(json.load(open('prompts.json'))['prompts'][$i]['prompt'])")
    N_PREDICT=$(python3 -c "import json; print(json.load(open('prompts.json'))['prompts'][$i]['n_predict'])")

    echo "Domain: $DOMAIN"
    echo "Prompt ID: $PROMPT_ID"
    echo "Tokens: $N_PREDICT"
    echo ""

    # Update settings.json for this prompt
    python3 << PYTHON
import json

# Load current settings
with open('settings.json', 'r') as f:
    settings = json.load(f)

# Update prompt and n_predict
settings['prompt'] = """$PROMPT_TEXT"""
settings['n_predict'] = $N_PREDICT

# Write updated settings
with open('settings.json', 'w') as f:
    json.dump(settings, f, indent=2)

print("✓ Updated settings.json")
PYTHON

    echo ""
    echo "Running inference..."

    # Run experiment (uses updated settings.json)
    python3 run_experiment.py

    # Create domain directory
    DOMAIN_DIR="$OUTPUT_BASE/$PROMPT_ID"
    mkdir -p "$DOMAIN_DIR"

    # Move output to domain folder
    echo ""
    echo "Moving data to $DOMAIN_DIR..."

    # Move generated data
    if [ -f "/tmp/tensor_trace.bin" ]; then
        mv /tmp/tensor_trace.bin "$DOMAIN_DIR/"
    fi

    if [ -d "webui/public/data/traces" ]; then
        mv webui/public/data/traces "$DOMAIN_DIR/"
    fi

    if [ -d "webui/public/data/graphs" ]; then
        mv webui/public/data/graphs "$DOMAIN_DIR/"
    fi

    if [ -f "webui/public/data/buffer-timeline.json" ]; then
        mv webui/public/data/buffer-timeline.json "$DOMAIN_DIR/"
    fi

    if [ -f "webui/public/data/memory-map.json" ]; then
        cp webui/public/data/memory-map.json "$DOMAIN_DIR/"  # Copy, not move (same for all)
    fi

    echo "✓ Data moved to $DOMAIN_DIR"
    echo ""

    # Summary
    TRACE_COUNT=$(find "$DOMAIN_DIR/traces" -name "*.json" 2>/dev/null | wc -l | tr -d ' ')
    echo "Collected: $TRACE_COUNT token traces"
    echo ""

done

# Restore original settings.json
mv settings.json.backup settings.json
echo "✓ Restored original settings.json"
echo ""

# Create summary
python3 << 'PYTHON'
import json
from pathlib import Path

exp_dir = Path('experiments/expert-analysis-2026-01-26')

summary = {
    'experiment': 'expert-analysis-2026-01-26',
    'date': '2026-01-26',
    'model': 'gpt-oss-20b-F16',
    'domains': []
}

for domain_dir in sorted(exp_dir.glob('domain-*')):
    traces_dir = domain_dir / 'traces'
    n_traces = len(list(traces_dir.glob('*.json'))) if traces_dir.exists() else 0
    size_mb = sum(f.stat().st_size for f in domain_dir.rglob('*') if f.is_file()) / (1024*1024)

    summary['domains'].append({
        'id': domain_dir.name,
        'tokens': n_traces,
        'size_mb': round(size_mb, 2)
    })

with open(exp_dir / 'summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("✓ Created summary.json")

# Print summary
print("\nExperiment Summary:")
print(f"  Total domains: {len(summary['domains'])}")
print(f"  Total tokens: {sum(d['tokens'] for d in summary['domains'])}")
print(f"  Total size: {sum(d['size_mb'] for d in summary['domains']):.1f} MB")
PYTHON

echo ""
echo "Creating tarball..."
cd experiments
tar -czf "../${EXPERIMENT_NAME}.tar.gz" "$EXPERIMENT_NAME"
cd ..

TARBALL_SIZE=$(du -h "${EXPERIMENT_NAME}.tar.gz" | cut -f1)
echo "✓ Created: ${EXPERIMENT_NAME}.tar.gz ($TARBALL_SIZE)"
echo ""

echo "===================================================================="
echo "Data collection complete!"
echo "===================================================================="
echo ""
echo "To transfer to local machine:"
echo "  scp ${EXPERIMENT_NAME}.tar.gz local:~/Downloads/"
echo ""
echo "To extract locally:"
echo "  cd ~/Downloads"
echo "  tar -xzf ${EXPERIMENT_NAME}.tar.gz"
echo "  mv ${EXPERIMENT_NAME} ~/Public/LLAMA/BSC/desktopui/data/"
echo ""