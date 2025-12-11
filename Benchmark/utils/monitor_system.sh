#!/bin/bash
# monitor_system.sh - Continuously monitor system metrics during inference
# Usage: ./monitor_system.sh <output_dir> <interval_seconds>

OUTPUT_DIR="$1"
INTERVAL="${2:-1}"  # Default 1 second

if [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 <output_dir> [interval_seconds]"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Starting system monitoring (interval: ${INTERVAL}s)"
echo "Output directory: $OUTPUT_DIR"

# Start iostat in background
iostat -x -t $INTERVAL > "$OUTPUT_DIR/iostat.log" &
IOSTAT_PID=$!

# Start continuous memory monitoring
(
    echo "timestamp,total_mb,used_mb,free_mb,shared_mb,cache_mb,available_mb,swap_total_mb,swap_used_mb,swap_free_mb"
    while true; do
        TIMESTAMP=$(date +%s)
        # Parse free -m output
        free -m | awk -v ts="$TIMESTAMP" '
        NR==2 {total=$2; used=$3; free=$4; shared=$5; cache=$6; available=$7}
        NR==3 {swap_total=$2; swap_used=$3; swap_free=$4}
        END {print ts","total","used","free","shared","cache","available","swap_total","swap_used","swap_free}
        '
        sleep $INTERVAL
    done
) > "$OUTPUT_DIR/memory.csv" &
MEMORY_PID=$!

# Start vmstat for page faults
vmstat -t $INTERVAL > "$OUTPUT_DIR/vmstat.log" &
VMSTAT_PID=$!

# Start CPU monitoring
(
    echo "timestamp,cpu_user,cpu_system,cpu_idle,cpu_iowait"
    while true; do
        TIMESTAMP=$(date +%s)
        mpstat $INTERVAL 1 | awk -v ts="$TIMESTAMP" '/Average/ && !/CPU/ {print ts","$3","$5","$12","$6}'
    done
) > "$OUTPUT_DIR/cpu.csv" &
CPU_PID=$!

# Save PIDs for cleanup
echo "$IOSTAT_PID" > "$OUTPUT_DIR/monitor_pids.txt"
echo "$MEMORY_PID" >> "$OUTPUT_DIR/monitor_pids.txt"
echo "$VMSTAT_PID" >> "$OUTPUT_DIR/monitor_pids.txt"
echo "$CPU_PID" >> "$OUTPUT_DIR/monitor_pids.txt"

echo "Monitoring started. PIDs:"
echo "  iostat: $IOSTAT_PID"
echo "  memory: $MEMORY_PID"
echo "  vmstat: $VMSTAT_PID"
echo "  cpu: $CPU_PID"
echo ""
echo "To stop: kill \$(cat $OUTPUT_DIR/monitor_pids.txt)"

# Keep script running
wait
