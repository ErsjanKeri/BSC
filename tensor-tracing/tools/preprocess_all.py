#!/usr/bin/env python3
"""
Preprocess All - Orchestrator for Tensor Trace Data Pipeline

This script coordinates all preprocessing steps to convert raw data
(graphs, traces, CSV) into JSON format ready for the WebUI.

Usage:
    # Basic usage (auto-detect tokens)
    python preprocess_all.py \
        --graphs /tmp/graphs \
        --trace /tmp/tensor_trace.bin \
        --csv /path/to/model_structure.csv \
        --output data

    # Process specific tokens only
    python preprocess_all.py \
        --graphs /tmp/graphs \
        --trace /tmp/tensor_trace.bin \
        --csv /path/to/model_structure.csv \
        --output data \
        --tokens 0,1,2,3

Directory structure created:
    data/
    ├── memory-map.json          (once, from CSV)
    ├── graphs/
    │   ├── token-00000.json
    │   ├── token-00001.json
    │   └── ...
    └── traces/
        ├── token-00000.json
        ├── token-00001.json
        └── ...
"""

import argparse
import os
import sys
import subprocess
import glob
import re
from pathlib import Path
from typing import List, Tuple


def find_script(script_name: str) -> Path:
    """
    Find a preprocessing script in the tools directory.

    Args:
        script_name: Name of script (e.g., "parse_csv.py")

    Returns:
        Path to script

    Raises:
        FileNotFoundError: If script not found
    """
    # Get directory where this script lives
    tools_dir = Path(__file__).parent

    script_path = tools_dir / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    return script_path


def detect_available_tokens(graphs_dir: Path) -> List[int]:
    """
    Auto-detect available token IDs from graph files.

    Args:
        graphs_dir: Directory containing .dot files

    Returns:
        Sorted list of token IDs
    """
    token_ids = []

    # Look for files like "token_00000.dot", "token_00001.dot", etc.
    pattern = graphs_dir / "token_*.dot"
    for dot_file in glob.glob(str(pattern)):
        filename = Path(dot_file).name
        match = re.search(r'token_(\d+)\.dot', filename)
        if match:
            token_id = int(match.group(1))
            token_ids.append(token_id)

    return sorted(token_ids)


def run_command(cmd: List[str], description: str) -> Tuple[bool, str]:
    """
    Run a command and return success status.

    Args:
        cmd: Command to run (list of args)
        description: Description for logging

    Returns:
        Tuple of (success, error_message)
    """
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        return True, ""
    except subprocess.CalledProcessError as e:
        error_msg = f"{description} failed:\n{e.stderr}"
        return False, error_msg
    except FileNotFoundError as e:
        error_msg = f"{description} failed: {e}"
        return False, error_msg


def preprocess_memory_map(csv_path: Path, output_dir: Path, verbose: bool = True) -> bool:
    """
    Generate memory-map.json from CSV.

    Args:
        csv_path: Path to model structure CSV
        output_dir: Output directory
        verbose: Print progress messages

    Returns:
        True if successful
    """
    if verbose:
        print("\n" + "="*60)
        print("STEP 1: Processing Memory Map")
        print("="*60)
        print(f"Input:  {csv_path}")

    output_file = output_dir / "memory-map.json"
    print(f"Output: {output_file}")

    # Find parse_csv.py script
    parse_csv = find_script("parse_csv.py")

    # Run parser
    cmd = [
        sys.executable,  # Use same Python as current process
        str(parse_csv),
        "--csv", str(csv_path),
        "--output", str(output_file),
        "--pretty"
    ]

    success, error = run_command(cmd, "parse_csv.py")

    if success:
        if verbose:
            size_kb = output_file.stat().st_size / 1024
            print(f"✓ Success! ({size_kb:.1f} KB)")
    else:
        print(f"✗ Error: {error}")

    return success


def preprocess_token(
    token_id: int,
    graphs_dir: Path,
    trace_file: Path,
    output_dir: Path,
    verbose: bool = True
) -> Tuple[bool, bool]:
    """
    Preprocess a single token (graph + trace).

    Args:
        token_id: Token ID to process
        graphs_dir: Directory containing .dot files
        trace_file: Path to binary trace file
        output_dir: Output directory
        verbose: Print progress messages

    Returns:
        Tuple of (graph_success, trace_success)
    """
    if verbose:
        print(f"\n{'─'*60}")
        print(f"Processing Token {token_id:05d}")
        print(f"{'─'*60}")

    graph_success = False
    trace_success = False

    # 1. Process graph
    dot_file = graphs_dir / f"token_{token_id:05d}.dot"
    graph_output = output_dir / "graphs" / f"token-{token_id:05d}.json"

    if dot_file.exists():
        if verbose:
            print(f"Graph:  {dot_file.name} → {graph_output.name}")

        parse_dot = find_script("parse_dot.py")
        cmd = [
            sys.executable,
            str(parse_dot),
            "--dot", str(dot_file),
            "--output", str(graph_output),
            "--pretty"
        ]

        graph_success, error = run_command(cmd, f"parse_dot.py (token {token_id})")
        if graph_success:
            size_kb = graph_output.stat().st_size / 1024
            print(f"  ✓ Graph JSON: {size_kb:.1f} KB")
        else:
            print(f"  ✗ Graph failed: {error}")
    else:
        print(f"  ⚠ Graph file not found: {dot_file}")

    # 2. Process trace
    trace_output = output_dir / "traces" / f"token-{token_id:05d}.json"

    if trace_file.exists():
        if verbose:
            print(f"Trace:  {trace_file.name} (token {token_id}) → {trace_output.name}")

        parse_trace = find_script("parse_trace.py")
        cmd = [
            sys.executable,
            str(parse_trace),
            str(trace_file),
            "--token", str(token_id),
            "--format", "json",
            "--output", str(trace_output)
        ]

        trace_success, error = run_command(cmd, f"parse_trace.py (token {token_id})")
        if trace_success and trace_output.exists():
            size_kb = trace_output.stat().st_size / 1024
            print(f"  ✓ Trace JSON: {size_kb:.1f} KB")
        elif trace_success:
            print(f"  ⚠ Trace completed but output file not found")
            trace_success = False
        else:
            print(f"  ✗ Trace failed: {error}")
    else:
        print(f"  ⚠ Trace file not found: {trace_file}")

    return graph_success, trace_success


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess all tensor trace data for WebUI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect all tokens
  python preprocess_all.py --graphs /tmp/graphs --trace /tmp/tensor_trace.bin \\
      --csv tinyllama_structure.csv --output data

  # Process specific tokens only
  python preprocess_all.py --graphs /tmp/graphs --trace /tmp/tensor_trace.bin \\
      --csv tinyllama_structure.csv --output data --tokens 0,1,2

  # Skip memory map (already generated)
  python preprocess_all.py --graphs /tmp/graphs --trace /tmp/tensor_trace.bin \\
      --csv tinyllama_structure.csv --output data --skip-memory-map
        """
    )

    parser.add_argument(
        '--graphs',
        required=True,
        type=Path,
        help='Directory containing .dot graph files (e.g., /tmp/graphs)'
    )
    parser.add_argument(
        '--trace',
        required=True,
        type=Path,
        help='Binary trace file (e.g., /tmp/tensor_trace.bin)'
    )
    parser.add_argument(
        '--csv',
        required=True,
        type=Path,
        help='Model structure CSV file (e.g., tinyllama_structure.csv)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data'),
        help='Output directory (default: data/)'
    )
    parser.add_argument(
        '--tokens',
        type=str,
        help='Comma-separated token IDs to process (e.g., "0,1,2"). Auto-detect if not specified.'
    )
    parser.add_argument(
        '--skip-memory-map',
        action='store_true',
        help='Skip memory map generation (if already exists)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress messages'
    )

    args = parser.parse_args()

    verbose = not args.quiet

    # Print header
    if verbose:
        print("\n" + "="*60)
        print("Tensor Trace Preprocessing Pipeline")
        print("="*60)

    # Validate inputs
    if not args.graphs.exists():
        print(f"Error: Graphs directory not found: {args.graphs}", file=sys.stderr)
        return 1

    if not args.trace.exists():
        print(f"Error: Trace file not found: {args.trace}", file=sys.stderr)
        return 1

    if not args.csv.exists():
        print(f"Error: CSV file not found: {args.csv}", file=sys.stderr)
        return 1

    # Create output directories
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "graphs").mkdir(exist_ok=True)
    (args.output / "traces").mkdir(exist_ok=True)

    if verbose:
        print(f"\nConfiguration:")
        print(f"  Graphs dir:  {args.graphs}")
        print(f"  Trace file:  {args.trace}")
        print(f"  CSV file:    {args.csv}")
        print(f"  Output dir:  {args.output}")

    # Determine tokens to process
    if args.tokens:
        # Parse comma-separated list
        token_ids = [int(t.strip()) for t in args.tokens.split(',')]
        if verbose:
            print(f"  Tokens:      {token_ids} (user-specified)")
    else:
        # Auto-detect
        token_ids = detect_available_tokens(args.graphs)
        if verbose:
            print(f"  Tokens:      {token_ids} (auto-detected)")

    if not token_ids:
        print("\nError: No tokens found. Check --graphs directory.", file=sys.stderr)
        return 1

    # Step 1: Process memory map (once)
    if not args.skip_memory_map:
        success = preprocess_memory_map(args.csv, args.output, verbose)
        if not success:
            print("\nWarning: Memory map generation failed. Continuing with tokens...")
    else:
        if verbose:
            print("\nSkipping memory map generation (--skip-memory-map)")

    # Step 2: Process each token
    if verbose:
        print("\n" + "="*60)
        print(f"STEP 2: Processing {len(token_ids)} Tokens")
        print("="*60)

    results = []
    for token_id in token_ids:
        graph_ok, trace_ok = preprocess_token(
            token_id,
            args.graphs,
            args.trace,
            args.output,
            verbose
        )
        results.append((token_id, graph_ok, trace_ok))

    # Summary
    if verbose:
        print("\n" + "="*60)
        print("Summary")
        print("="*60)

        graph_success_count = sum(1 for _, g, _ in results if g)
        trace_success_count = sum(1 for _, _, t in results if t)

        print(f"Tokens processed: {len(results)}")
        print(f"  Graphs: {graph_success_count}/{len(results)} successful")
        print(f"  Traces: {trace_success_count}/{len(results)} successful")

        # Show failures
        failures = [(tid, g, t) for tid, g, t in results if not (g and t)]
        if failures:
            print("\nFailures:")
            for tid, g, t in failures:
                status = []
                if not g:
                    status.append("graph")
                if not t:
                    status.append("trace")
                print(f"  Token {tid:05d}: {', '.join(status)} failed")

        print("\n✓ Preprocessing complete!")
        print(f"  Output directory: {args.output.absolute()}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
