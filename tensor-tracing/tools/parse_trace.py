#!/usr/bin/env python3
"""
Tensor Trace Binary Parser - 1024-byte format with 128-byte names

Parses /tmp/tensor_trace.bin with the 1024-byte format:
- ONE entry per operation
- ALL sources embedded (up to 4)
- Memory source detection (DISK vs BUFFER)
- 128-byte names for full tensor names without truncation

Usage:
    python3 parse_trace.py                       # Show all entries
    python3 parse_trace.py --limit 20            # Show first 20 entries
    python3 parse_trace.py --stats               # Show statistics
    python3 parse_trace.py --verify              # Verify format
"""

import struct
import sys
import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict

# === Structure Definitions (1024-byte format) ===

# TensorAccessLog: 1024 bytes total
# - Metadata: 24 bytes
# - Destination name: 128 bytes
# - Sources: 640 bytes (4 × 160 bytes)
# - Expert IDs: 64 bytes (16 × int32)
# - num_experts: 1 byte
# - Padding: 167 bytes

ENTRY_SIZE = 1024
SOURCE_SIZE = 160
METADATA_SIZE = 24
DST_NAME_SIZE = 128
SOURCES_TOTAL_SIZE = 640
EXPERT_IDS_SIZE = 64  # 16 × 4 bytes
NUM_EXPERTS_SIZE = 1
PADDING_SIZE = 167

# Metadata format (24 bytes)
METADATA_FORMAT = '<QIHHBBB5s'  # timestamp, token_id, layer_id, thread_id, op_type, phase, num_sources, padding[5]

# Source format (160 bytes)
SOURCE_FORMAT = '<128sQIHBBQI4s'  # name[128], tensor_ptr, size_bytes, layer_id, memory_source, pad1, disk_offset_or_buffer_id, tensor_idx, pad2[4]

# Operation type names (ggml_op enum)
# IMPORTANT: This MUST match the exact order in ggml/include/ggml.h
# Last synced: 2026-01-08 from llama.cpp commit
OPERATION_TYPES = {
    0: "NONE",
    1: "DUP",
    2: "ADD",
    3: "ADD_ID",        # FIXED: Was missing, causing all subsequent ops to shift
    4: "ADD1",          # FIXED: Was missing
    5: "ACC",           # FIXED: Was missing
    6: "SUB",           # FIXED: Was missing
    7: "MUL",           # Was at position 3 (WRONG)
    8: "DIV",
    9: "SQR",
    10: "SQRT",
    11: "LOG",          # Was at position 7 (WRONG)
    12: "SIN",
    13: "COS",
    14: "SUM",
    15: "SUM_ROWS",
    16: "CUMSUM",       # FIXED: Was missing
    17: "MEAN",
    18: "ARGMAX",
    19: "COUNT_EQUAL",
    20: "REPEAT",
    21: "REPEAT_BACK",
    22: "CONCAT",
    23: "SILU_BACK",
    24: "NORM",         # FIXED: Was at position 19
    25: "RMS_NORM",     # Was at position 20
    26: "RMS_NORM_BACK",
    27: "GROUP_NORM",
    28: "L2_NORM",      # FIXED: Was missing
    29: "MUL_MAT",      # Was at position 23 (WRONG)
    30: "MUL_MAT_ID",
    31: "OUT_PROD",     # Was at position 25 (WRONG)
    32: "SCALE",
    33: "SET",
    34: "CPY",
    35: "CONT",         # Was at position 29 (WRONG)
    36: "RESHAPE",
    37: "VIEW",
    38: "PERMUTE",
    39: "TRANSPOSE",
    40: "GET_ROWS",     # Was at position 34
    41: "GET_ROWS_BACK",
    42: "SET_ROWS",     # FIXED: Was missing
    43: "DIAG",
    44: "DIAG_MASK_INF",
    45: "DIAG_MASK_ZERO",
    46: "SOFT_MAX",
    47: "SOFT_MAX_BACK",
    48: "ROPE",
    49: "ROPE_BACK",
    50: "CLAMP",
    51: "CONV_TRANSPOSE_1D",
    52: "IM2COL",
    53: "IM2COL_BACK",
    54: "IM2COL_3D",    # FIXED: Was missing
    55: "CONV_2D",      # FIXED: Was missing
    56: "CONV_3D",      # FIXED: Was missing
    57: "CONV_2D_DW",   # FIXED: Was missing
    58: "CONV_TRANSPOSE_2D",  # Was at position 47
    59: "POOL_1D",      # Was at position 48
    60: "POOL_2D",
    61: "POOL_2D_BACK",
    62: "UPSCALE",
    63: "PAD",
    64: "PAD_REFLECT_1D",  # FIXED: Was missing
    65: "ROLL",         # FIXED: Was missing
    66: "ARANGE",
    67: "TIMESTEP_EMBEDDING",
    68: "ARGSORT",
    69: "TOP_K",        # FIXED: Was missing
    70: "LEAKY_RELU",
    71: "TRI",          # FIXED: Was missing
    72: "FILL",         # FIXED: Was missing
    73: "FLASH_ATTN_EXT",
    74: "FLASH_ATTN_BACK",
    75: "SSM_CONV",
    76: "SSM_SCAN",
    77: "WIN_PART",
    78: "WIN_UNPART",
    79: "GET_REL_POS",
    80: "ADD_REL_POS",
    81: "RWKV_WKV6",
    82: "GATED_LINEAR_ATTN",  # FIXED: Was missing
    83: "RWKV_WKV7",    # FIXED: Was missing
    84: "SOLVE_TRI",    # FIXED: Was missing
    85: "UNARY",
    86: "MAP_CUSTOM1",  # Was split into MAP_CUSTOM1_F32, etc. (WRONG)
    87: "MAP_CUSTOM2",
    88: "MAP_CUSTOM3",
    89: "CUSTOM",       # FIXED: Was missing
    90: "CROSS_ENTROPY_LOSS",
    91: "CROSS_ENTROPY_LOSS_BACK",
    92: "OPT_STEP_ADAMW",
    93: "OPT_STEP_SGD",
    94: "GLU",
}


def parse_source(data, offset):
    """Parse a single SourceTensorInfo (160 bytes)."""
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
    """Parse a complete TensorAccessLog (1024 bytes)."""
    if len(data) < ENTRY_SIZE:
        return None

    try:
        offset = 0

        # Parse metadata (24 bytes)
        metadata = struct.unpack_from(METADATA_FORMAT, data, offset)
        offset += METADATA_SIZE

        timestamp_ns = metadata[0]
        if timestamp_ns == 0:  # Empty entry
            return None

        token_id = metadata[1]
        layer_id = metadata[2] if metadata[2] != 65535 else None
        thread_id = metadata[3]
        operation_type = metadata[4]
        phase = metadata[5]
        num_sources = metadata[6]
        # metadata[7] is padding[5] - ignored

        # Parse dst_name (128 bytes)
        dst_name = data[offset:offset+DST_NAME_SIZE].decode('utf-8', errors='ignore').rstrip('\x00')
        offset += DST_NAME_SIZE

        # Parse sources (4 × 160 bytes)
        sources = []
        for i in range(4):
            src = parse_source(data, offset)
            if src and i < num_sources:  # Only include valid sources
                sources.append(src)
            offset += SOURCE_SIZE

        # Parse expert IDs (64 bytes = 16 × int32)
        expert_ids_format = '<16i'  # 16 int32 values
        expert_ids_raw = struct.unpack_from(expert_ids_format, data, offset)
        offset += EXPERT_IDS_SIZE

        # Parse num_experts (1 byte)
        num_experts = struct.unpack_from('<B', data, offset)[0]
        offset += NUM_EXPERTS_SIZE

        # Extract only valid expert IDs
        expert_ids = list(expert_ids_raw[:num_experts]) if num_experts > 0 else []

        # Remaining padding (167 bytes) - skip

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
            'expert_ids': expert_ids,       # NEW
            'num_experts': num_experts,     # NEW
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

        # Show expert IDs if present
        if entry.get('num_experts', 0) > 0:
            top4 = entry['expert_ids'][:4]  # Show top-4 (actually used)
            all_ids = entry['expert_ids']
            print(f"  Experts: TOP-4: {top4}, All: {all_ids}")

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


def export_to_json_per_token(entries, output_dir):
    """
    Export trace entries as JSON files, one per token.

    Args:
        entries: List of parsed trace entries
        output_dir: Directory to write JSON files (e.g., 'webui/public/data/traces/')

    Returns:
        Number of token files written
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Group entries by token_id
    entries_by_token = defaultdict(list)
    for entry in entries:
        token_id = entry['token_id']
        entries_by_token[token_id].append(entry)

    # Write one JSON file per token
    num_files = 0
    for token_id in sorted(entries_by_token.keys()):
        token_entries = entries_by_token[token_id]

        # Compute metadata for this token
        timestamps = [e['timestamp_ns'] for e in token_entries]
        timestamp_start = min(timestamps)
        timestamp_end = max(timestamps)
        duration_ms = (timestamp_end - timestamp_start) / 1_000_000  # ns -> ms

        # Format entries for JSON export
        formatted_entries = []
        for i, entry in enumerate(token_entries):
            # Format sources with proper field names
            formatted_sources = []
            for src in entry['sources']:
                formatted_src = {
                    'name': src['name'],
                    'tensor_ptr': f"0x{src['tensor_ptr']:x}",  # Format as hex string
                    'size_bytes': src['size_bytes'],
                    'layer_id': src['layer_id'],
                    'memory_source': src['memory_source']
                }

                # Add disk_offset or buffer_id based on memory source
                if src['memory_source'] == 'DISK':
                    formatted_src['disk_offset'] = src['disk_offset_or_buffer_id']
                else:
                    formatted_src['buffer_id'] = src['disk_offset_or_buffer_id']

                formatted_sources.append(formatted_src)

            formatted_entry = {
                'entry_id': i,
                'timestamp_ns': entry['timestamp_ns'],
                'timestamp_relative_ms': round((entry['timestamp_ns'] - timestamp_start) / 1_000_000, 3),
                'token_id': token_id,
                'layer_id': entry['layer_id'],
                'thread_id': entry['thread_id'],
                'phase': entry['phase'],
                'operation_type': entry['operation_name'],
                'dst_name': entry['dst_name'],
                'num_sources': entry['num_sources'],
                'sources': formatted_sources,
                'expert_ids': entry.get('expert_ids', []),      # NEW
                'num_experts': entry.get('num_experts', 0)      # NEW
            }
            formatted_entries.append(formatted_entry)

        # Build JSON structure
        token_json = {
            'token_id': token_id,
            'metadata': {
                'total_entries': len(token_entries),
                'duration_ms': round(duration_ms, 3),
                'timestamp_start_ns': timestamp_start,
                'format_version': '1024-byte'
            },
            'entries': formatted_entries
        }

        # Write to file
        output_file = output_path / f"token-{token_id:05d}.json"
        with open(output_file, 'w') as f:
            json.dump(token_json, f, indent=2)

        file_size_kb = output_file.stat().st_size / 1024
        print(f"✓ Token {token_id:5d}: {len(token_entries):4d} entries → {output_file} ({file_size_kb:.1f} KB)")
        num_files += 1

    return num_files


def main():
    parser = argparse.ArgumentParser(description='Parse tensor trace binary (1024-byte format with 128-byte names)')
    parser.add_argument('trace_file', nargs='?', default='/tmp/tensor_trace.bin',
                        help='Path to trace binary (default: /tmp/tensor_trace.bin)')
    parser.add_argument('--limit', type=int, default=0,
                        help='Limit number of entries to display (0 = all)')
    parser.add_argument('--stats', action='store_true',
                        help='Show statistics')
    parser.add_argument('--verify', action='store_true',
                        help='Verify format correctness')
    parser.add_argument('--export-json', type=str, metavar='OUTPUT_DIR',
                        help='Export entries to JSON files per token (e.g., webui/public/data/traces/)')

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
    print(f"Expected entries: {expected_entries} (at 1024 bytes each)\n")

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
    if args.export_json:
        print(f"\nExporting to JSON (grouped by token)...")
        num_files = export_to_json_per_token(entries, args.export_json)
        print(f"\n✓ Exported {num_files} token files to {args.export_json}")
    elif args.verify:
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
