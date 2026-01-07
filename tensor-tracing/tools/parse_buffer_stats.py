#!/usr/bin/env python3
"""
Parse Buffer Stats JSONL to Timeline JSON

Parses /tmp/buffer_stats.jsonl and generates a buffer occupancy timeline
suitable for visualization in the WebUI.

Input format (JSONL):
    {"timestamp_ms":0.125,"event":"alloc","buffer_id":123,"buffer_ptr":456,"size":1024,"name":"KVCache","backend":"CPU","usage":2,"layer":65535}
    {"timestamp_ms":5.450,"event":"dealloc","buffer_id":123}

Output format (JSON):
    {
      "metadata": {...},
      "buffers": [{id, name, size, usage, backend, ...}, ...],
      "timeline": [{timestamp_ms, event, buffer_id, ...}, ...]
    }

Usage:
    python3 parse_buffer_stats.py /tmp/buffer_stats.jsonl --output data/buffer-timeline.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any


# Buffer usage type names
BUFFER_USAGE_NAMES = {
    0: "ANY",
    1: "WEIGHTS",
    2: "COMPUTE"
}


def parse_jsonl(jsonl_path: Path) -> List[Dict[str, Any]]:
    """
    Parse JSONL file and return list of events.

    Args:
        jsonl_path: Path to JSONL file

    Returns:
        List of event dictionaries
    """
    events = []

    with open(jsonl_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                event = json.loads(line)
                events.append(event)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}", file=sys.stderr)
                continue

    return events


def build_buffer_registry(events: List[Dict]) -> Dict[int, Dict]:
    """
    Build registry of all buffers from allocation events.

    Args:
        events: List of events

    Returns:
        Dictionary mapping buffer_id -> buffer_info
    """
    registry = {}

    for event in events:
        if event.get('event') == 'alloc':
            buffer_id = event['buffer_id']
            registry[buffer_id] = {
                'id': buffer_id,
                'name': event.get('name', 'unnamed'),
                'size': event.get('size', 0),
                'backend': event.get('backend', 'unknown'),
                'usage': event.get('usage', 0),
                'usage_name': BUFFER_USAGE_NAMES.get(event.get('usage', 0), 'UNKNOWN'),
                'layer': event.get('layer', 65535),
                'alloc_time_ms': event['timestamp_ms'],
                'dealloc_time_ms': None  # Will be filled in later
            }

    return registry


def compute_occupancy_timeline(events: List[Dict], registry: Dict[int, Dict]) -> List[Dict]:
    """
    Compute buffer occupancy over time.

    Args:
        events: List of events
        registry: Buffer registry

    Returns:
        Timeline with cumulative occupancy
    """
    timeline = []
    active_buffers = {}  # buffer_id -> size
    cumulative_size = 0

    for event in events:
        timestamp_ms = event['timestamp_ms']
        buffer_id = event['buffer_id']
        event_type = event.get('event')

        if event_type == 'alloc':
            size = event.get('size', 0)
            active_buffers[buffer_id] = size
            cumulative_size += size

            timeline.append({
                'timestamp_ms': timestamp_ms,
                'event': 'alloc',
                'buffer_id': buffer_id,
                'buffer_name': event.get('name', 'unnamed'),
                'size': size,
                'cumulative_size': cumulative_size,
                'num_active_buffers': len(active_buffers)
            })

        elif event_type == 'dealloc':
            if buffer_id in active_buffers:
                size = active_buffers.pop(buffer_id)
                cumulative_size -= size

                # Update registry with dealloc time
                if buffer_id in registry:
                    registry[buffer_id]['dealloc_time_ms'] = timestamp_ms

                timeline.append({
                    'timestamp_ms': timestamp_ms,
                    'event': 'dealloc',
                    'buffer_id': buffer_id,
                    'buffer_name': registry.get(buffer_id, {}).get('name', 'unknown'),
                    'size': size,
                    'cumulative_size': cumulative_size,
                    'num_active_buffers': len(active_buffers)
                })
            else:
                print(f"Warning: Dealloc for unknown buffer {buffer_id}", file=sys.stderr)

    return timeline


def compute_metadata(events: List[Dict], registry: Dict, timeline: List[Dict]) -> Dict[str, Any]:
    """
    Compute metadata about buffer usage.

    Args:
        events: List of events
        registry: Buffer registry
        timeline: Occupancy timeline

    Returns:
        Metadata dictionary
    """
    if not timeline:
        return {
            'total_events': 0,
            'total_buffers': 0,
            'peak_occupancy_bytes': 0,
            'peak_occupancy_mb': 0,
            'duration_ms': 0
        }

    # Peak occupancy
    peak_occupancy = max((t['cumulative_size'] for t in timeline), default=0)

    # Duration
    first_ts = timeline[0]['timestamp_ms']
    last_ts = timeline[-1]['timestamp_ms']
    duration_ms = last_ts - first_ts

    # Count by usage type
    usage_counts = {}
    for buf in registry.values():
        usage_name = buf['usage_name']
        usage_counts[usage_name] = usage_counts.get(usage_name, 0) + 1

    return {
        'total_events': len(events),
        'total_buffers': len(registry),
        'peak_occupancy_bytes': peak_occupancy,
        'peak_occupancy_mb': round(peak_occupancy / (1024 * 1024), 2),
        'duration_ms': duration_ms,
        'usage_breakdown': usage_counts
    }


def format_size(size_bytes: int) -> str:
    """Format size in human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f}KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f}MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f}GB"


def generate_output(events: List[Dict], output_path: Path = None) -> None:
    """
    Generate output JSON from events.

    Args:
        events: List of events from JSONL
        output_path: Output file path (None for stdout)
    """
    # Build buffer registry
    registry = build_buffer_registry(events)

    # Compute occupancy timeline
    timeline = compute_occupancy_timeline(events, registry)

    # Compute metadata
    metadata = compute_metadata(events, registry, timeline)

    # Convert registry to list
    buffers = sorted(registry.values(), key=lambda b: b['alloc_time_ms'])

    # Build output structure
    output = {
        'metadata': metadata,
        'buffers': buffers,
        'timeline': timeline
    }

    # Write output
    if output_path:
        output_dir = output_path.parent
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
            print(f"Created directory: {output_dir}")

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        file_size = output_path.stat().st_size
        print(f"✓ Buffer timeline written to: {output_path}")
        print(f"  File size: {file_size / 1024:.1f} KB")
    else:
        # Output to stdout
        json.dump(output, sys.stdout, indent=2)
        print()  # Newline after JSON


def display_summary(events: List[Dict]) -> None:
    """
    Display summary statistics to console.

    Args:
        events: List of events
    """
    registry = build_buffer_registry(events)
    timeline = compute_occupancy_timeline(events, registry)
    metadata = compute_metadata(events, registry, timeline)

    print("\n=== Buffer Statistics ===\n")
    print(f"Total events:       {metadata['total_events']}")
    print(f"Total buffers:      {metadata['total_buffers']}")
    print(f"Peak occupancy:     {format_size(metadata['peak_occupancy_bytes'])} ({metadata['peak_occupancy_mb']} MB)")
    print(f"Duration:           {metadata['duration_ms']:.2f} ms")

    print(f"\nBuffer usage breakdown:")
    for usage_name, count in metadata['usage_breakdown'].items():
        print(f"  {usage_name:<10}: {count} buffers")

    print(f"\nBuffers:")
    for buf in sorted(registry.values(), key=lambda b: b['size'], reverse=True)[:10]:
        lifetime = "active"
        if buf['dealloc_time_ms'] is not None:
            lifetime = f"{buf['dealloc_time_ms'] - buf['alloc_time_ms']:.2f}ms"

        print(f"  {buf['name']:<30} {format_size(buf['size']):>10} [{buf['usage_name']:<8}] ({lifetime})")

    if len(registry) > 10:
        print(f"  ... and {len(registry) - 10} more")


def main():
    parser = argparse.ArgumentParser(
        description='Parse buffer stats JSONL to timeline JSON'
    )
    parser.add_argument(
        'jsonl_file',
        nargs='?',
        type=Path,
        default=Path('/tmp/buffer_stats.jsonl'),
        help='Path to JSONL file (default: /tmp/buffer_stats.jsonl)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Output JSON file (default: stdout)'
    )
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Display summary statistics'
    )

    args = parser.parse_args()

    # Check input file exists
    if not args.jsonl_file.exists():
        print(f"Error: File not found: {args.jsonl_file}", file=sys.stderr)
        return 1

    # Parse JSONL
    print(f"Reading: {args.jsonl_file}")
    events = parse_jsonl(args.jsonl_file)

    if not events:
        print("No events found in file", file=sys.stderr)
        return 1

    print(f"✓ Parsed {len(events)} events")

    # Display summary or generate output
    if args.summary:
        display_summary(events)
    else:
        generate_output(events, args.output)

    return 0


if __name__ == '__main__':
    sys.exit(main())
