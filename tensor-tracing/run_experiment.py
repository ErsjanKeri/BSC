#!/usr/bin/env python3
"""
Tensor Tracing Experiment Runner

Automates the complete pipeline for tensor tracing experiments:
1. Clean old trace files
2. Run llama-completion with tensor tracing enabled
3. Parse all generated data (GGUF, trace, graphs, buffer stats)
4. Move processed data to webui/public/data/ for visualization

Usage:
    python3 run_experiment.py

Configuration:
    Edit settings.json to configure experiment parameters
"""

import os
import sys
import json
import subprocess
import shutil
from pathlib import Path
from datetime import datetime


def log(message):
    """Print log message with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")


def error_exit(message):
    """Print error and exit."""
    print(f"\nERROR: {message}", file=sys.stderr)
    sys.exit(1)


def run_cmd(cmd, description, check=True, cwd=None):
    """
    Run shell command and handle errors.

    Args:
        cmd: Command string or list
        description: Description for logging
        check: Whether to exit on error
        cwd: Working directory

    Returns:
        CompletedProcess result
    """
    log(description)

    try:
        if isinstance(cmd, str):
            result = subprocess.run(
                cmd,
                shell=True,
                check=check,
                cwd=cwd,
                capture_output=True,
                text=True
            )
        else:
            result = subprocess.run(
                cmd,
                check=check,
                cwd=cwd,
                capture_output=True,
                text=True
            )

        if result.returncode != 0 and check:
            error_exit(f"{description} failed:\n{result.stderr}")

        return result

    except subprocess.CalledProcessError as e:
        error_exit(f"{description} failed:\n{e.stderr}")
    except FileNotFoundError:
        error_exit(f"Command not found: {cmd}")


def load_settings():
    """Load settings from settings.json."""
    settings_path = Path(__file__).parent / "settings.json"

    if not settings_path.exists():
        error_exit(f"Settings file not found: {settings_path}")

    with open(settings_path, 'r') as f:
        settings = json.load(f)

    log(f"Loaded settings from {settings_path}")
    return settings


def resolve_paths(settings):
    """
    Resolve all paths to absolute paths.

    Args:
        settings: Settings dictionary

    Returns:
        Dictionary of resolved paths
    """
    script_dir = Path(__file__).parent.absolute()

    # Resolve llama.cpp directory
    llama_cpp_dir = (script_dir / settings['paths']['llama_cpp_dir']).resolve()

    paths = {
        'script_dir': script_dir,
        'llama_cpp_dir': llama_cpp_dir,
        'llama_binary': llama_cpp_dir / settings['paths']['llama_binary'],
        'gguf_dump_binary': llama_cpp_dir / 'build/bin/llama-gguf-dump',
        'webui_data_dir': script_dir / settings['paths']['webui_data_dir'],
        'tools_dir': script_dir / 'tools',
        'trace_bin': Path(settings['temp_files']['trace_bin']),
        'buffer_stats': Path(settings['temp_files']['buffer_stats']),
        'graphs_dir': Path(settings['temp_files']['graphs_dir'])
    }

    # Resolve model path (relative to llama_cpp_dir)
    model_file = settings['experiment']['model_path']
    paths['model_file'] = (llama_cpp_dir / model_file).resolve()

    return paths


def verify_prerequisites(paths):
    """
    Verify all required files and directories exist.

    Args:
        paths: Dictionary of paths
    """
    log("Verifying prerequisites...")

    # Check llama-completion binary
    if not paths['llama_binary'].exists():
        error_exit(f"llama-completion not found: {paths['llama_binary']}")

    # Check gguf-dump binary
    if not paths['gguf_dump_binary'].exists():
        error_exit(f"llama-gguf-dump not found: {paths['gguf_dump_binary']}")

    # Check model file
    if not paths['model_file'].exists():
        error_exit(f"Model file not found: {paths['model_file']}")

    # Check parser scripts
    required_scripts = ['parse_trace.py', 'parse_dot.py', 'parse_buffer_stats.py', 'parse_csv.py']
    for script in required_scripts:
        script_path = paths['tools_dir'] / script
        if not script_path.exists():
            error_exit(f"Parser script not found: {script_path}")

    log("✓ All prerequisites verified")


def clean_temp_files(paths):
    """
    Clean old trace files and temporary data.

    Args:
        paths: Dictionary of paths
    """
    log("Cleaning old trace files...")

    files_to_remove = [
        paths['trace_bin'],
        paths['buffer_stats']
    ]

    for file_path in files_to_remove:
        if file_path.exists():
            file_path.unlink()
            log(f"  Removed: {file_path}")

    # Clean graphs directory
    if paths['graphs_dir'].exists():
        shutil.rmtree(paths['graphs_dir'])
        log(f"  Removed: {paths['graphs_dir']}")

    log("✓ Temp files cleaned")


def run_llama_inference(paths, settings):
    """
    Run llama-completion with tensor tracing enabled.

    Args:
        paths: Dictionary of paths
        settings: Settings dictionary

    Returns:
        Inference time in seconds
    """
    log("Running llama-completion with tensor tracing...")

    # Build command
    cmd = [
        str(paths['llama_binary']),
        '-m', str(paths['model_file']),
        '-p', settings['experiment']['prompt'],
        '-n', str(settings['experiment']['tokens_to_generate']),
        '-no-cnv'  # Disable conversation mode
    ]

    log(f"  Model: {paths['model_file'].name}")
    log(f"  Prompt: \"{settings['experiment']['prompt']}\"")
    log(f"  Tokens: {settings['experiment']['tokens_to_generate']}")

    # Run inference
    import time
    start_time = time.time()

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    elapsed_time = time.time() - start_time

    if result.returncode != 0:
        error_exit(f"llama-completion failed:\n{result.stderr}")

    log(f"✓ Inference complete ({elapsed_time:.2f}s)")

    # Verify trace files were generated
    if not paths['trace_bin'].exists():
        error_exit(f"Trace file not generated: {paths['trace_bin']}")

    if not paths['buffer_stats'].exists():
        error_exit(f"Buffer stats not generated: {paths['buffer_stats']}")

    if not paths['graphs_dir'].exists() or not list(paths['graphs_dir'].glob('*.dot')):
        error_exit(f"Graph files not generated in: {paths['graphs_dir']}")

    # Count generated files
    num_graphs = len(list(paths['graphs_dir'].glob('*.dot')))
    trace_size_mb = paths['trace_bin'].stat().st_size / (1024 * 1024)

    log(f"  Trace file: {trace_size_mb:.2f} MB")
    log(f"  Graph files: {num_graphs}")

    return elapsed_time


def parse_gguf_to_memory_map(paths):
    """
    Parse GGUF file to generate memory-map.json.

    Args:
        paths: Dictionary of paths
    """
    log("Parsing GGUF model...")

    # Step 1: Generate CSV using llama-gguf-dump
    csv_path = paths['script_dir'] / 'temp_model_structure.csv'

    result = subprocess.run(
        [str(paths['gguf_dump_binary']), str(paths['model_file'])],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        error_exit(f"llama-gguf-dump failed:\n{result.stderr}")

    # Write CSV output
    with open(csv_path, 'w') as f:
        f.write(result.stdout)

    log(f"  CSV generated: {csv_path}")

    # Step 2: Parse CSV to JSON
    output_path = paths['webui_data_dir'] / 'memory-map.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    parse_csv_script = paths['tools_dir'] / 'parse_csv.py'

    result = subprocess.run(
        [
            'python3',
            str(parse_csv_script),
            '--csv', str(csv_path),
            '--gguf-file', str(paths['model_file']),  # NEW: Pass GGUF file for offset calculation
            '--output', str(output_path)
        ],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        error_exit(f"parse_csv.py failed:\n{result.stderr}")

    # Clean up temp CSV
    csv_path.unlink()

    log(f"✓ Memory map generated: {output_path}")


def parse_trace_to_json(paths):
    """
    Parse binary trace file to JSON per token.

    Args:
        paths: Dictionary of paths
    """
    log("Parsing tensor trace...")

    output_dir = paths['webui_data_dir'] / 'traces'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clear existing trace JSONs
    for old_file in output_dir.glob('token-*.json'):
        old_file.unlink()

    parse_trace_script = paths['tools_dir'] / 'parse_trace.py'

    result = subprocess.run(
        [
            'python3',
            str(parse_trace_script),
            str(paths['trace_bin']),
            '--export-json', str(output_dir)
        ],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        error_exit(f"parse_trace.py failed:\n{result.stderr}")

    # Print output (shows per-token stats)
    print(result.stdout)

    num_files = len(list(output_dir.glob('token-*.json')))
    log(f"✓ Generated {num_files} trace JSON files")


def parse_graphs_to_json(paths):
    """
    Parse DOT graph files to JSON per token.

    Args:
        paths: Dictionary of paths
    """
    log("Parsing computation graphs...")

    output_dir = paths['webui_data_dir'] / 'graphs'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clear existing graph JSONs
    for old_file in output_dir.glob('token-*.json'):
        old_file.unlink()

    parse_dot_script = paths['tools_dir'] / 'parse_dot.py'

    # Process each DOT file
    dot_files = sorted(paths['graphs_dir'].glob('token_*.dot'))

    if not dot_files:
        error_exit(f"No DOT files found in {paths['graphs_dir']}")

    for dot_file in dot_files:
        # Extract token ID from filename (e.g., token_00001.dot -> 1)
        token_id_str = dot_file.stem.split('_')[1]  # "00001"
        output_file = output_dir / f"token-{token_id_str}.json"

        result = subprocess.run(
            [
                'python3',
                str(parse_dot_script),
                '--dot', str(dot_file),
                '--output', str(output_file)
            ],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            error_exit(f"parse_dot.py failed for {dot_file}:\n{result.stderr}")

    num_files = len(list(output_dir.glob('token-*.json')))
    log(f"✓ Generated {num_files} graph JSON files")


def parse_buffer_stats(paths):
    """
    Parse buffer stats JSONL to timeline JSON.

    Args:
        paths: Dictionary of paths
    """
    log("Parsing buffer statistics...")

    output_file = paths['webui_data_dir'] / 'buffer-timeline.json'
    output_file.parent.mkdir(parents=True, exist_ok=True)

    parse_buffer_script = paths['tools_dir'] / 'parse_buffer_stats.py'

    result = subprocess.run(
        [
            'python3',
            str(parse_buffer_script),
            str(paths['buffer_stats']),
            '--output', str(output_file)
        ],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        error_exit(f"parse_buffer_stats.py failed:\n{result.stderr}")

    log(f"✓ Buffer timeline generated: {output_file}")


def display_summary(paths, inference_time):
    """
    Display experiment summary.

    Args:
        paths: Dictionary of paths
        inference_time: Inference time in seconds
    """
    log("\n" + "=" * 70)
    log("EXPERIMENT COMPLETE!")
    log("=" * 70)

    # Count generated files
    data_dir = paths['webui_data_dir']

    num_trace_files = len(list((data_dir / 'traces').glob('token-*.json')))
    num_graph_files = len(list((data_dir / 'graphs').glob('token-*.json')))

    log(f"\nGenerated data:")
    log(f"  Memory map:      {data_dir / 'memory-map.json'}")
    log(f"  Buffer timeline: {data_dir / 'buffer-timeline.json'}")
    log(f"  Trace files:     {num_trace_files} tokens")
    log(f"  Graph files:     {num_graph_files} tokens")

    log(f"\nPerformance:")
    log(f"  Inference time:  {inference_time:.2f}s")

    log(f"\n✓ All data ready in: {data_dir}")
    log(f"  Start the webui dev server to visualize results")
    log("=" * 70 + "\n")


def main():
    """Main entry point."""

    print("\n" + "=" * 70)
    print("  Tensor Tracing Experiment Runner")
    print("=" * 70 + "\n")

    # Load settings
    settings = load_settings()

    # Resolve paths
    paths = resolve_paths(settings)

    # Verify prerequisites
    verify_prerequisites(paths)

    # Step 1: Clean old files
    clean_temp_files(paths)

    # Step 2: Run inference
    inference_time = run_llama_inference(paths, settings)

    # Step 3: Parse GGUF to memory map
    parse_gguf_to_memory_map(paths)

    # Step 4: Parse trace to JSON per token
    parse_trace_to_json(paths)

    # Step 5: Parse graphs to JSON per token
    parse_graphs_to_json(paths)

    # Step 6: Parse buffer stats
    parse_buffer_stats(paths)

    # Display summary
    display_summary(paths, inference_time)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
