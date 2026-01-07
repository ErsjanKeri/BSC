#!/usr/bin/env python3
"""
Tensor Trace Binary Parser - Phase 1 (256-byte format)

Parses /tmp/tensor_trace.bin with the NEW 256-byte format:
- ONE entry per operation
- ALL sources embedded (up to 4)
- Memory source detection (DISK vs BUFFER)
- Proper GGUF offsets

Usage:
    python3 parse_trace_v2.py                       # Show all entries
    python3 parse_trace_v2.py --limit 20            # Show first 20 entries
    python3 parse_trace_v2.py --stats               # Show statistics
    python3 parse_trace_v2.py --verify              # Verify format
"""

import struct
import sys
import argparse
from pathlib import Path
from collections import Counter

# === Structure Definitions (Phase 1) ===

# TensorAccessLog: 256 bytes total
# - Header: 24 bytes (metadata + destination)
# - Sources: 208 bytes (4 × 52 bytes)

ENTRY_SIZE = 256
SOURCE_SIZE = 52

# Header format (24 bytes)
HEADER_FORMAT = '<QIHHBBB5s24s'  # timestamp, token_id, layer_id, thread_id, op_type, phase, num_sources, padding, dst_name

# Source format (52 bytes)
SOURCE_FORMAT = '<20sQIHBBQI4s'  # name[20], tensor_ptr, size_bytes, layer_id, memory_source, pad1, disk_offset_or_buffer_id, tensor_idx, pad2[4]

# Operation type names (ggml_op enum)
OPERATION_TYPES = {
    0: "NONE",
    1: "DUP",
    2: "ADD",
    3: "MUL",
    4: "DIV",
    5: "SQR",
    6: "SQRT",
    7: "LOG",
    8: "SIN",
    9: "COS",
    10: "SUM",
    11: "SUM_ROWS",
    12: "MEAN",
    13: "ARGMAX",
    14: "COUNT_EQUAL",
    15: "REPEAT",
    16: "REPEAT_BACK",
    17: "CONCAT",
    18: "SILU_BACK",
    19: "NORM",
    20: "RMS_NORM",
    21: "RMS_NORM_BACK",
    22: "GROUP_NORM",
    23: "MUL_MAT",
    24: "MUL_MAT_ID",
    25: "OUT_PROD",
    26: "SCALE",
    27: "SET",
    28: "CPY",
    29: "CONT",
    30: "RESHAPE",
    31: "VIEW",
    32: "PERMUTE",
    33: "TRANSPOSE",
    34: "GET_ROWS",
    35: "GET_ROWS_BACK",
    36: "DIAG",
    37: "DIAG_MASK_INF",
    38: "DIAG_MASK_ZERO",
    39: "SOFT_MAX",
    40: "SOFT_MAX_BACK",
    41: "ROPE",
    42: "ROPE_BACK",
    43: "CLAMP",
    44: "CONV_TRANSPOSE_1D",
    45: "IM2COL",
    46: "IM2COL_BACK",
    47: "CONV_TRANSPOSE_2D",
    48: "POOL_1D",
    49: "POOL_2D",
    50: "POOL_2D_BACK",
    51: "UPSCALE",
    52: "PAD",
    53: "ARANGE",
    54: "TIMESTEP_EMBEDDING",
    55: "ARGSORT",
    56: "LEAKY_RELU",
    57: "FLASH_ATTN_EXT",
    58: "FLASH_ATTN_BACK",
    59: "SSM_CONV",
    60: "SSM_SCAN",
    61: "WIN_PART",
    62: "WIN_UNPART",
    63: "GET_REL_POS",
    64: "ADD_REL_POS",
    65: "RWKV_WKV6",
    66: "UNARY",
    67: "MAP_UNARY",
    68: "MAP_BINARY",
    69: "MAP_CUSTOM1_F32",
    70: "MAP_CUSTOM2_F32",
    71: "MAP_CUSTOM3_F32",
    72: "MAP_CUSTOM1",
    73: "MAP_CUSTOM2",
    74: "MAP_CUSTOM3",
    75: "CROSS_ENTROPY_LOSS",
    76: "CROSS_ENTROPY_LOSS_BACK",
    77: "OPT_STEP_ADAMW",
}


def parse_source(data, offset):
    """Parse a single SourceTensorInfo (52 bytes)."""
    try:
        unpacked = struct.unpack_from(SOURCE_FORMAT, data, offset)

        return {
            'name': unpacked[0].decode('utf-8', errors='ignore').rstrip('\x00'),
            'tensor_ptr': unpacked[1],
            'size_bytes': unpacked[2],
            'layer_id': unpacked[3] if unpacked[3] != 65535 else None,
            'memory_source': 'DISK' if unpacked[4] == 0 else 'BUFFER',
            'disk_offset_or_buffer_id': unpacked[6],
            'tensor_idx': unpacked[7] if unpacked[7] != 0xFFFFFFFF else None,
        }
    except Exception as e:
        print(f"Error parsing source at offset {offset}: {e}", file=sys.stderr)
        return None


def parse_entry(data, entry_num):
    """Parse a complete TensorAccessLog (256 bytes)."""
    if len(data) < ENTRY_SIZE:
        return None

    try:
        # Parse header (24 bytes)
        header = struct.unpack_from(HEADER_FORMAT, data, 0)

        timestamp_ns = header[0]
        if timestamp_ns == 0:  # Empty entry
            return None

        token_id = header[1]
        layer_id = header[2] if header[2] != 65535 else None
        thread_id = header[3]
        operation_type = header[4]
        phase = header[5]
        num_sources = header[6]
        dst_name = header[8].decode('utf-8', errors='ignore').rstrip('\x00')

        # Parse sources (4 × 52 bytes, starting at offset 48)
        sources = []
        source_offset = struct.calcsize(HEADER_FORMAT)

        for i in range(4):
            src = parse_source(data, source_offset + i * SOURCE_SIZE)
            if src and i < num_sources:  # Only include valid sources
                sources.append(src)

        return {
            'entry_num': entry_num,
            'timestamp_ns': timestamp_ns,
            'token_id': token_id,
            'layer_id': layer_id,
            'thread_id': thread_id,
            'operation_type': operation_type,
            'operation_name': OPERATION_TYPES.get(operation_type, f'UNKNOWN_{operation_type}'),
            'phase': 'PROMPT' if phase == 0 else 'GENERATE',
            'num_sources': num_sources,
            'dst_name': dst_name,
            'sources': sources,
        }
    except Exception as e:
        print(f"Error parsing entry {entry_num}: {e}", file=sys.stderr)
        return None


def format_size(size_bytes):
    """Format size in human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f}KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f}MB"


def display_entries(entries, limit=0):
    """Display trace entries in detailed format."""
    if limit > 0:
        entries = entries[:limit]

    for entry in entries:
        time_ms = entry['timestamp_ns'] / 1_000_000
        layer_str = f"L{entry['layer_id']}" if entry['layer_id'] is not None else "N/A"

        print(f"\n=== Entry #{entry['entry_num']} ===")
        print(f"  Time: {time_ms:.3f} ms")
        print(f"  Token: {entry['token_id']}, Layer: {layer_str}, Thread: {entry['thread_id']}")
        print(f"  Operation: {entry['operation_name']}, Phase: {entry['phase']}")
        print(f"  Destination: '{entry['dst_name']}'")
        print(f"  Sources ({entry['num_sources']}):")

        for i, src in enumerate(entry['sources']):
            size_str = format_size(src['size_bytes'])
            layer_str = f"L{src['layer_id']}" if src['layer_id'] is not None else "N/A"

            print(f"    [{i}] '{src['name']}' - {size_str} - {src['memory_source']}")

            if src['memory_source'] == 'DISK':
                print(f"        Disk offset: 0x{src['disk_offset_or_buffer_id']:X}")
            else:
                print(f"        Buffer ID: 0x{src['disk_offset_or_buffer_id']:X}")

            if src['tensor_idx'] is not None:
                print(f"        Tensor idx: {src['tensor_idx']}")


def show_statistics(entries):
    """Show comprehensive statistics."""
    if not entries:
        print("No entries to analyze")
        return

    print("\n=== Trace Statistics ===\n")
    print(f"Total entries: {len(entries)}")

    # Time range
    first_ts = entries[0]['timestamp_ns']
    last_ts = entries[-1]['timestamp_ns']
    duration_ms = (last_ts - first_ts) / 1_000_000
    print(f"Duration: {duration_ms:.3f} ms")

    # Token distribution
    token_counts = Counter(e['token_id'] for e in entries)
    print(f"\nTokens: {list(token_counts.keys())}")

    # Layer distribution
    layer_counts = Counter(e['layer_id'] for e in entries if e['layer_id'] is not None)
    print(f"\nLayers accessed: {sorted(layer_counts.keys())}")
    print(f"Unique layers: {len(layer_counts)}")

    # Operation types
    op_counts = Counter(e['operation_name'] for e in entries)
    print(f"\nOperation distribution:")
    for op_name, count in op_counts.most_common():
        pct = (count / len(entries)) * 100
        print(f"  {op_name:<20}: {count:>4} ({pct:>5.1f}%)")

    # Source analysis
    total_sources = sum(e['num_sources'] for e in entries)
    avg_sources = total_sources / len(entries) if entries else 0
    print(f"\nSource tensors:")
    print(f"  Total: {total_sources}")
    print(f"  Average per operation: {avg_sources:.2f}")

    # Memory source distribution
    all_sources = [src for e in entries for src in e['sources']]
    if all_sources:
        mem_source_counts = Counter(src['memory_source'] for src in all_sources)
        print(f"\nMemory source distribution:")
        for mem_type, count in mem_source_counts.items():
            pct = (count / len(all_sources)) * 100
            print(f"  {mem_type}: {count} ({pct:.1f}%)")

    # Unique tensors
    unique_sources = set(src['name'] for src in all_sources if src['name'])
    print(f"\nUnique source tensors: {len(unique_sources)}")

    # Data volume
    total_bytes = sum(src['size_bytes'] for src in all_sources)
    print(f"\nTotal data accessed: {format_size(total_bytes)}")


def verify_format(entries):
    """Verify the trace format is correct."""
    print("\n=== Format Verification ===\n")

    if not entries:
        print("❌ No entries found")
        return False

    issues = []

    # Check entry count
    print(f"✓ Found {len(entries)} entries")

    # Check if entries have valid sources
    entries_with_sources = [e for e in entries if e['num_sources'] > 0]
    print(f"✓ Entries with sources: {len(entries_with_sources)} / {len(entries)}")

    # Check destination names
    entries_with_dst = [e for e in entries if e['dst_name']]
    print(f"✓ Entries with destination names: {len(entries_with_dst)} / {len(entries)}")

    # Check source names
    all_sources = [src for e in entries for src in e['sources']]
    sources_with_names = [src for src in all_sources if src['name']]
    print(f"✓ Sources with names: {len(sources_with_names)} / {len(all_sources)}")

    # Check memory source detection
    disk_sources = [src for src in all_sources if src['memory_source'] == 'DISK']
    buffer_sources = [src for src in all_sources if src['memory_source'] == 'BUFFER']
    print(f"✓ DISK sources: {len(disk_sources)}")
    print(f"✓ BUFFER sources: {len(buffer_sources)}")

    # Check disk offsets
    disk_with_offsets = [src for src in disk_sources if src['disk_offset_or_buffer_id'] > 0]
    if disk_sources:
        offset_pct = (len(disk_with_offsets) / len(disk_sources)) * 100
        print(f"✓ DISK sources with non-zero offsets: {len(disk_with_offsets)} / {len(disk_sources)} ({offset_pct:.1f}%)")
        if offset_pct < 50:
            issues.append(f"⚠️  Only {offset_pct:.1f}% of DISK sources have non-zero offsets")

    # Check operation types
    valid_ops = [e for e in entries if e['operation_name'] != f"UNKNOWN_{e['operation_type']}"]
    print(f"✓ Valid operation types: {len(valid_ops)} / {len(entries)}")

    # Check layers
    entries_with_layer = [e for e in entries if e['layer_id'] is not None]
    if entries_with_layer:
        unique_layers = set(e['layer_id'] for e in entries_with_layer)
        print(f"✓ Unique layers: {len(unique_layers)} (expecting 22 for TinyLlama)")
        if len(unique_layers) < 22:
            issues.append(f"⚠️  Only {len(unique_layers)} layers captured (expected 22)")

    # Summary
    if issues:
        print(f"\n❌ Found {len(issues)} issues:")
        for issue in issues:
            print(f"  {issue}")
        return False
    else:
        print("\n✅ Format verification passed!")
        return True


def main():
    parser = argparse.ArgumentParser(description='Parse tensor trace binary (256-byte format)')
    parser.add_argument('trace_file', nargs='?', default='/tmp/tensor_trace.bin',
                        help='Path to trace binary (default: /tmp/tensor_trace.bin)')
    parser.add_argument('--limit', type=int, default=0,
                        help='Limit number of entries to display (0 = all)')
    parser.add_argument('--stats', action='store_true',
                        help='Show statistics')
    parser.add_argument('--verify', action='store_true',
                        help='Verify format correctness')

    args = parser.parse_args()

    # Load trace file
    trace_path = Path(args.trace_file)
    if not trace_path.exists():
        print(f"Error: Trace file not found: {trace_path}", file=sys.stderr)
        return 1

    # Get file size
    file_size = trace_path.stat().st_size
    expected_entries = file_size // ENTRY_SIZE
    print(f"Trace file: {trace_path}")
    print(f"File size: {file_size / (1024*1024):.2f} MB ({file_size} bytes)")
    print(f"Expected entries: {expected_entries} (at 256 bytes each)\n")

    # Parse entries
    entries = []
    with open(trace_path, 'rb') as f:
        entry_num = 0
        while True:
            data = f.read(ENTRY_SIZE)
            if len(data) < ENTRY_SIZE:
                break

            entry = parse_entry(data, entry_num)
            if entry is None:
                break

            entries.append(entry)
            entry_num += 1

    print(f"✓ Parsed {len(entries)} entries\n")

    if not entries:
        print("No valid entries found", file=sys.stderr)
        return 1

    # Execute requested action
    if args.verify:
        verify_format(entries)
    elif args.stats:
        show_statistics(entries)
    else:
        display_entries(entries, args.limit if args.limit > 0 else 10)
        if args.limit == 0:
            print(f"\n... ({len(entries) - 10} more entries, use --limit to show more)")

    return 0


if __name__ == '__main__':
    sys.exit(main())
