#!/bin/bash
#
# Collect expert activation traces for 2000 tokens across 5 domains.
# Uses --trace-mode experts for lightweight tracing (~295 MB per domain).
#
# Output: experiments/expert-traces-2k/<domain>/traces/token-NNNNN.json
#
# Usage: ./run_expert_trace_2k.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LLAMA="$HOME/llama.cpp/build/bin/llama-completion"
MODEL="$HOME/llama.cpp/models/gpt-oss-20b-F16.gguf"
EXPERIMENT_DIR="experiments/expert-traces-2k"
N_PREDICT=2000
SEED=42

if [ ! -f "$LLAMA" ]; then
    echo "ERROR: llama-completion not found at $LLAMA"
    exit 1
fi

mkdir -p "$EXPERIMENT_DIR"

echo "===================================================================="
echo "Expert Trace Collection (2000 tokens, experts-only mode)"
echo "===================================================================="
echo "Output: $EXPERIMENT_DIR"
echo ""

# Prompts array (matching prompts.json domains)
declare -A PROMPTS
PROMPTS["domain-1-code"]="Write a Python function that implements a binary search tree with insert, search, and delete operations. Include detailed docstrings explaining each method."
PROMPTS["domain-2-math"]="Solve this problem step by step: If a train travels at 80 km/h for 2.5 hours, then increases speed to 120 km/h for another 1.5 hours, what is the total distance traveled? Show your reasoning clearly."
PROMPTS["domain-3-creative"]="Write a short story about an AI system discovering the concept of friendship through observing humans in a café. Use vivid sensory descriptions and focus on the AI's internal perspective."
PROMPTS["domain-4-factual"]="Explain how CRISPR gene editing works, including the role of guide RNA, Cas9 protein, and the DNA repair mechanisms involved. Describe both the mechanism and potential applications."
PROMPTS["domain-5-mixed"]="Compare the time complexity of quicksort and mergesort algorithms. Explain when each is preferred, then write a Python implementation of whichever is better for sorting nearly-sorted arrays."

DOMAINS=("domain-1-code" "domain-2-math" "domain-3-creative" "domain-4-factual" "domain-5-mixed")

for domain in "${DOMAINS[@]}"; do
    echo "===================================================================="
    echo "Domain: $domain"
    echo "===================================================================="

    DOMAIN_DIR="$EXPERIMENT_DIR/$domain"
    TRACES_DIR="$DOMAIN_DIR/traces"
    mkdir -p "$TRACES_DIR"

    # Clean previous trace
    rm -f /tmp/tensor_trace.bin

    echo "Running inference ($N_PREDICT tokens)..."
    $LLAMA \
        -m "$MODEL" \
        -p "${PROMPTS[$domain]}" \
        -n "$N_PREDICT" \
        -ngl 0 -no-cnv --no-warmup \
        --trace-mode experts \
        --seed "$SEED" \
        2>&1 | tail -3

    echo ""

    # Parse the binary trace into per-token JSON using existing parser
    if [ -f /tmp/tensor_trace.bin ]; then
        echo "Parsing trace..."
        python3 tools/parse_trace.py --export-json "$TRACES_DIR"

        N_TOKENS=$(ls "$TRACES_DIR"/token-*.json 2>/dev/null | wc -l)
        TRACE_SIZE=$(du -sh "$TRACES_DIR" | cut -f1)
        echo "  Collected: $N_TOKENS token traces ($TRACE_SIZE)"

        # Copy raw binary too (for re-parsing if needed)
        cp /tmp/tensor_trace.bin "$DOMAIN_DIR/tensor_trace.bin"
    else
        echo "WARNING: No trace file generated!"
    fi

    echo ""
done

# Run analysis
echo "===================================================================="
echo "Running expert pattern analysis..."
echo "===================================================================="

for domain in "${DOMAINS[@]}"; do
    TRACES_DIR="$EXPERIMENT_DIR/$domain/traces"
    if [ -d "$TRACES_DIR" ]; then
        echo ""
        echo "--- $domain ---"
        python3 tools/analyze_expert_patterns.py "$TRACES_DIR" \
            --cache-sizes 100,200,250,288,350,500,740 \
            > "$EXPERIMENT_DIR/$domain/analysis.txt" 2>&1
        # Print summary lines
        grep -E "^(Loaded|Mean|Global|Total unique|Saturation|  cache=)" "$EXPERIMENT_DIR/$domain/analysis.txt"
    fi
done

echo ""
echo "===================================================================="
echo "Done! Results in: $EXPERIMENT_DIR"
echo "===================================================================="
