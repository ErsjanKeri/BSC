#!/bin/bash
# Launch all 5 domain analysis windows

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATA_DIR="../tensor-tracing/expert-analysis-2026-01-26"

# Check if data exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory not found: $DATA_DIR"
    echo "Please run the expert experiments first."
    exit 1
fi

echo "Launching 5 domain analysis windows..."
echo ""

# Launch each domain in background
./build/bin/tensor-trace-analyzer "$DATA_DIR/domain-1-code" &
sleep 0.5

./build/bin/tensor-trace-analyzer "$DATA_DIR/domain-2-math" &
sleep 0.5

./build/bin/tensor-trace-analyzer "$DATA_DIR/domain-3-creative" &
sleep 0.5

./build/bin/tensor-trace-analyzer "$DATA_DIR/domain-4-factual" &
sleep 0.5

./build/bin/tensor-trace-analyzer "$DATA_DIR/domain-5-mixed" &

echo "✓ Launched 5 windows"
echo "Press Ctrl+C to close all windows"

wait
