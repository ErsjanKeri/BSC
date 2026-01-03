#!/usr/bin/env python3
"""
Tensor Trace Binary Parser

Parses /tmp/tensor_trace.bin and displays trace entries in human-readable format.
Supports both Path A (direct tensor names) and Path B (tensor_idx lookup) validation.

Usage:
    python3 parse_trace.py                          # Show all entries
    python3 parse_trace.py --limit 20               # Show first 20 entries
    python3 parse_trace.py --layer 5                # Filter by layer 5
    python3 parse_trace.py --csv tinyllama_structure.csv  # Correlate with CSV
"""

import struct
import sys
import argparse
import csv
from pathlib import Path

# TensorAccessLog struct layout (128 bytes total)
# uint64_t timestamp_ns;        // 0-8
# uint32_t token_id;            // 8-12
# uint16_t layer_id;            // 12-14
# uint16_t thread_id;           // 14-16
# uint8_t  operation_type;      // 16-17
# uint8_t  phase;               // 17-18
# uint16_t padding1;            // 18-20
# uint32_t padding1b;           // 20-24
# uint32_t tensor_idx;          // 24-28
# uint64_t tensor_ptr;          // 28-36
# uint64_t file_offset;         // 36-44
# uint32_t size_bytes;          // 44-48
# uint8_t  attention_head;      // 48-49
# uint8_t  qkv_type;            // 49-50
# uint16_t padding2;            // 50-52
# uint8_t  expert_id;           // 52-53
# uint8_t  expert_rank;         // 53-54
# uint16_t routing_score;       // 54-56
# uint32_t padding3;            // 56-60
# uint32_t padding4;            // 60-64
# char tensor_name[64];         // 64-128

ENTRY_SIZE = 128
OPERATION_TYPES = {
    1: "MUL_MAT",
    2: "ADD",
    3: "ROPE",
    # Add more as needed
}

def parse_entry(data):
    """Parse a single 128-byte trace entry."""
    if len(data) < ENTRY_SIZE:
        return None

    # Parse fixed fields
    timestamp_ns = struct.unpack('<Q', data[0:8])[0]
    if timestamp_ns == 0:  # Empty entry
        return None

    token_id = struct.unpack('<I', data[8:12])[0]
    layer_id = struct.unpack('<H', data[12:14])[0]
    thread_id = struct.unpack('<H', data[14:16])[0]
    operation_type = struct.unpack('<B', data[16:17])[0]
    phase = struct.unpack('<B', data[17:18])[0]

    tensor_idx = struct.unpack('<I', data[24:28])[0]
    tensor_ptr = struct.unpack('<Q', data[28:36])[0]
    file_offset = struct.unpack('<Q', data[36:44])[0]
    size_bytes = struct.unpack('<I', data[44:48])[0]

    # Parse tensor name (bytes 64-128)
    tensor_name_bytes = data[64:128]
    tensor_name = tensor_name_bytes.split(b'\x00')[0].decode('utf-8', errors='ignore')

    return {
        'timestamp_ns': timestamp_ns,
        'token_id': token_id,
        'layer_id': layer_id,
        'thread_id': thread_id,
        'operation_type': operation_type,
        'phase': phase,
        'tensor_idx': tensor_idx,
        'tensor_ptr': tensor_ptr,
        'file_offset': file_offset,
        'size_bytes': size_bytes,
        'tensor_name': tensor_name,
    }

def load_csv(csv_path):
    """Load CSV structure for correlation."""
    tensors = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['tensor_name']
            tensors[name] = {
                'file_offset': int(row['file_offset']),
                'size_bytes': int(row['size_bytes']),
                'layer_id': int(row['layer_id']),
                'component_type': row['component_type'],
            }
    return tensors

def format_size(size_bytes):
    """Format size in human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f}KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f}MB"

def format_layer(layer_id):
    """Format layer ID (65535 = N/A)."""
    return "N/A" if layer_id == 65535 else str(layer_id)

def main():
    parser = argparse.ArgumentParser(description='Parse tensor trace binary')
    parser.add_argument('trace_file', nargs='?', default='/tmp/tensor_trace.bin',
                        help='Path to trace binary (default: /tmp/tensor_trace.bin)')
    parser.add_argument('--limit', type=int, default=0,
                        help='Limit number of entries to display (0 = all)')
    parser.add_argument('--layer', type=int, default=None,
                        help='Filter by layer ID')
    parser.add_argument('--csv', type=str, default=None,
                        help='Path to CSV structure for correlation')
    parser.add_argument('--validate', action='store_true',
                        help='Validate Path A vs Path B correlation')
    parser.add_argument('--stats', action='store_true',
                        help='Show statistics instead of entries')

    args = parser.parse_args()

    # Load trace file
    trace_path = Path(args.trace_file)
    if not trace_path.exists():
        print(f"Error: Trace file not found: {trace_path}", file=sys.stderr)
        return 1

    # Load CSV if provided
    csv_data = None
    if args.csv:
        csv_data = load_csv(args.csv)
        print(f"Loaded {len(csv_data)} tensors from CSV\n")

    # Parse entries
    entries = []
    with open(trace_path, 'rb') as f:
        entry_num = 0
        while True:
            data = f.read(ENTRY_SIZE)
            if len(data) < ENTRY_SIZE:
                break

            entry = parse_entry(data)
            if entry is None:
                break

            entry['entry_num'] = entry_num
            entries.append(entry)
            entry_num += 1

    print(f"Parsed {len(entries)} trace entries\n")

    # Apply filters
    filtered = entries
    if args.layer is not None:
        filtered = [e for e in entries if e['layer_id'] == args.layer]
        print(f"Filtered to {len(filtered)} entries for layer {args.layer}\n")

    # Show statistics
    if args.stats:
        show_statistics(entries, csv_data)
        return 0

    # Validate Path A vs Path B
    if args.validate:
        validate_paths(entries)
        return 0

    # Display entries
    display_entries(filtered, args.limit, csv_data)

    return 0

def show_statistics(entries, csv_data):
    """Show statistics about trace entries."""
    from collections import Counter

    print("=== Trace Statistics ===\n")
    print(f"Total entries: {len(entries)}")

    # Layer distribution
    layer_counts = Counter(e['layer_id'] for e in entries)
    print("\nEntries per layer:")
    for layer_id in sorted(layer_counts.keys()):
        layer_str = format_layer(layer_id)
        print(f"  Layer {layer_str:>3}: {layer_counts[layer_id]:>4} entries")

    # Operation types
    op_counts = Counter(e['operation_type'] for e in entries)
    print("\nEntries per operation:")
    for op_type, count in op_counts.items():
        op_name = OPERATION_TYPES.get(op_type, f"UNKNOWN({op_type})")
        print(f"  {op_name}: {count} entries")

    # Size distribution
    total_bytes = sum(e['size_bytes'] for e in entries)
    avg_size = total_bytes / len(entries) if entries else 0
    print(f"\nTotal data accessed: {format_size(total_bytes)}")
    print(f"Average access size: {format_size(int(avg_size))}")

    # Unique tensors accessed
    unique_names = set(e['tensor_name'] for e in entries if e['tensor_name'])
    print(f"\nUnique tensors accessed: {len(unique_names)}")

    if csv_data:
        # Coverage: what % of model was accessed?
        coverage = len(unique_names) / len(csv_data) * 100 if csv_data else 0
        print(f"Model coverage: {coverage:.1f}% ({len(unique_names)}/{len(csv_data)} tensors)")

def validate_paths(entries):
    """Validate that Path A (tensor_name) matches Path B (tensor_idx)."""
    print("=== Validating Path A vs Path B ===\n")

    # Build reverse lookup: tensor_idx → expected_name
    # We'll use the first occurrence of each tensor_idx to build the table
    idx_to_name = {}
    mismatches = []

    for entry in entries:
        tensor_idx = entry['tensor_idx']
        tensor_name = entry['tensor_name']

        # Skip entries with no tensor_idx (lookup failed)
        if tensor_idx == 0xFFFFFFFF:  # UINT32_MAX
            continue

        if tensor_idx not in idx_to_name:
            # First time seeing this idx, store it
            idx_to_name[tensor_idx] = tensor_name
        else:
            # Verify it matches
            expected_name = idx_to_name[tensor_idx]
            if tensor_name != expected_name:
                mismatches.append({
                    'entry_num': entry['entry_num'],
                    'tensor_idx': tensor_idx,
                    'path_a_name': tensor_name,
                    'path_b_name': expected_name,
                })

    print(f"Validated {len(entries)} entries")
    print(f"Unique tensor indices: {len(idx_to_name)}")

    if mismatches:
        print(f"\n⚠️  Found {len(mismatches)} mismatches:")
        for m in mismatches[:10]:  # Show first 10
            print(f"  Entry {m['entry_num']}: idx={m['tensor_idx']}, "
                  f"Path A='{m['path_a_name']}', Path B='{m['path_b_name']}'")
    else:
        print("\n✅ All entries match! Path A and Path B are consistent.")

def display_entries(entries, limit, csv_data):
    """Display trace entries in table format."""
    if limit > 0:
        entries = entries[:limit]

    # Header
    print(f"{'#':>4} {'Time(ms)':>10} {'Tok':>4} {'Lay':>3} {'Op':>7} "
          f"{'Size':>8} {'TIdx':>5} {'Tensor Name':<40}")
    print("-" * 95)

    # Entries
    for entry in entries:
        time_ms = entry['timestamp_ns'] / 1_000_000
        layer = format_layer(entry['layer_id'])
        op_name = OPERATION_TYPES.get(entry['operation_type'], f"OP{entry['operation_type']}")
        size = format_size(entry['size_bytes'])

        # Handle UINT32_MAX (not found in registry)
        tensor_idx_str = "N/A" if entry['tensor_idx'] == 0xFFFFFFFF else str(entry['tensor_idx'])

        name = entry['tensor_name'] if entry['tensor_name'] else "<anonymous>"

        # Truncate long names
        if len(name) > 40:
            name = name[:37] + "..."

        print(f"{entry['entry_num']:>4} {time_ms:>10.2f} {entry['token_id']:>4} "
              f"{layer:>3} {op_name:>7} {size:>8} {tensor_idx_str:>5} {name:<40}")

        # If CSV provided, show correlation
        if csv_data and entry['tensor_name'] in csv_data:
            csv_entry = csv_data[entry['tensor_name']]
            csv_layer = format_layer(csv_entry['layer_id'])
            print(f"     {'':>10} {'':>4} {csv_layer:>3} {'CSV':>7} "
                  f"{format_size(csv_entry['size_bytes']):>8} {'':>5} "
                  f"{csv_entry['component_type']:<40}")

if __name__ == '__main__':
    sys.exit(main())
