#!/bin/bash
# Investigation: What happens during the 5-second "load" phase?
#
# This script monitors I/O activity during model loading to understand
# WHY it takes 5+ seconds even without MAP_POPULATE
#
# NOTE: Run this on the TUM server (cli-hiwi-02), not on macOS

set -e

# Detect if on server or local
if [[ $(hostname) == *"cli-hiwi"* ]]; then
    LLAMA_BIN="$HOME/llama.cpp/build_prefetch_off/bin/llama-completion"
    MODEL="$HOME/llama.cpp/models/gpt-oss-20b-F16.gguf"
else
    echo "ERROR: This script should be run on the TUM server (cli-hiwi-02)"
    echo "Local model path: /Users/ersibesi/Public/LLAMA/gpt-oss-20b-F16.gguf"
    exit 1
fi

OUTPUT_DIR="./io_investigation"

mkdir -p "$OUTPUT_DIR"

echo "======================================"
echo "I/O Investigation: Model Load Phase"
echo "======================================"
echo ""
echo "Model: $(basename $MODEL)"
echo "Size: $(du -h $MODEL | cut -f1)"
echo "Binary: MAP_POPULATE=OFF"
echo ""

# Clean page cache
echo "Dropping page cache..."
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sync
    sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
else
    sudo purge  # macOS
fi

# Start iostat in background
echo "Starting iostat monitoring..."
iostat -d 1 > "$OUTPUT_DIR/iostat.log" &
IOSTAT_PID=$!

# Start monitoring
echo "Starting inference with I/O monitoring..."
echo "Watch htop in another terminal to see RAM usage"
echo ""

# Run inference and capture timing
/usr/bin/time -l "$LLAMA_BIN" \
    -m "$MODEL" \
    -p "Hello" \
    -n 10 \
    -no-cnv \
    2>&1 | tee "$OUTPUT_DIR/llama_output.log"

# Stop iostat
kill $IOSTAT_PID 2>/dev/null || true

echo ""
echo "======================================"
echo "Results saved to: $OUTPUT_DIR/"
echo "======================================"
echo ""
echo "Check these files:"
echo "  - llama_output.log: Inference timing"
echo "  - iostat.log: Disk I/O during run"
echo ""
echo "KEY QUESTIONS:"
echo "  1. How much data was actually read from disk?"
echo "  2. When did the reads happen? (during load or inference?)"
echo "  3. How long did mmap() call take?"
