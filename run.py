import subprocess
import os
import argparse
import sys

# constants: paths to the python scripts
# These paths point to the scripts within the 'src' directory.
SCRIPTS = {
    'multihead': 'src/multihead_attention.py',
    'plot_multihead': 'src/plot_multihead_attention.py',
    'kl': 'src/kl_divergence.py',
    'plot_kl': 'src/plot_kl_divergence.py',
    'score': 'src/score_distribution.py',
    'plot_score': 'src/plot_score_distribution.py',
    'lowrank': 'src/low_rank_attention.py',
    'plot_lowrank': 'src/plot_low_rank_attention.py',
    'kl_lowrank': 'src/kl_divergence_low_rank_attention.py',
    'plot_kl_lowrank': 'src/plot_kl_divergence_low_rank_attention.py',
    'multihead_relu': 'src/multihead_attention_relu.py',
    'plot_multihead_relu': 'src/plot_multihead_attention_relu.py',
    'kl_relu': 'src/kl_divergence_relu.py',
    'plot_kl_relu': 'src/plot_kl_divergence_relu.py'
}

def check_files_exist():
    # Checks if all required script files exist before running
    for name, path in SCRIPTS.items():
        if not os.path.exists(path):
            print(f"Error: Script '{path}' not found.", file=sys.stderr)
            sys.exit(1)

def create_dirs():
    # Creates the output directories for data and figures if they don't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    print("Prepared 'data' and 'figures' directories.")

def run_script(script_path):
    # Executes a given Python script and handles errors.
    print(f"\n--- Running: {script_path} ---")
    try:
        # Using check=True to raise an exception if the script returns a non-zero exit code.
        result = subprocess.run(
            ['python', script_path],
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8' # Specify encoding for cross-platform compatibility
        )
        print(result.stdout)
        if result.stderr:
            print("--- Stderr ---")
            print(result.stderr)
        print(f"--- Finished successfully: {script_path} ---")
    except subprocess.CalledProcessError as e:
        print(f"!!! Error: Failed to run {script_path}.", file=sys.stderr)
        print(f"Return code: {e.returncode}", file=sys.stderr)
        print("--- Stdout ---", file=sys.stderr)
        print(e.stdout, file=sys.stderr)
        print("--- Stderr ---", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        sys.exit(1) # Exit the main script if any sub-script fails.

def run_experiment_1():
    # Runs experiment 1: Multi-head attention and KL divergence.
    print("\n[Starting Experiment 1: Multi-Head Attention Output Distribution]")
    run_script(SCRIPTS['multihead'])
    run_script(SCRIPTS['plot_multihead'])

    print("\n[Starting Experiment 1b: KL Divergence Calculation]")
    run_script(SCRIPTS['kl'])
    run_script(SCRIPTS['plot_kl'])

def run_experiment_2():
    # Runs experiment 2: Score distribution.
    print("\n[Starting Experiment 2: Score Distribution]")
    run_script(SCRIPTS['score'])
    run_script(SCRIPTS['plot_score'])

def run_experiment_3():
    # Runs experiment 3: Low-rank attention.
    print("\n[Starting Experiment 3: Low-Rank Attention Output Distribution]")
    run_script(SCRIPTS['lowrank'])
    run_script(SCRIPTS['plot_lowrank'])

    print("\n[Starting Experiment 3b: KL Divergence Calculation for Low-Rank Attention]")
    run_script(SCRIPTS['kl_lowrank'])
    run_script(SCRIPTS['plot_kl_lowrank'])

def run_experiment_4():
    # Runs experiment 4: Multi-head attention with ReLU activation.
    print("\n[Starting Experiment 4: Multi-Head Attention with ReLU Activation Output Distribution]")
    run_script(SCRIPTS['multihead_relu'])
    run_script(SCRIPTS['plot_multihead_relu'])

    print("\n[Starting Experiment 4b: KL Divergence Calculation]")
    run_script(SCRIPTS['kl_relu'])
    run_script(SCRIPTS['plot_kl_relu'])

def main():
    parser = argparse.ArgumentParser(description="Run experiments on attention mechanisms.")
    parser.add_argument(
        'experiment',
        nargs='*',  # Allow zero or more arguments
        choices=['all', '1', '2', '3', '4'],
        help="Select experiment(s) to run: '1', '2', '3', '4', or 'all'. (default: all)"
    )

    args = parser.parse_args()

    # If no arguments provided, default to ['all']
    experiments_to_run = args.experiment if args.experiment else ['all']

    # preparation
    check_files_exist()
    create_dirs()

    # Run selected experiments
    if 'all' in experiments_to_run or '1' in experiments_to_run:
        run_experiment_1()

    if 'all' in experiments_to_run or '2' in experiments_to_run:
        run_experiment_2()

    if 'all' in experiments_to_run or '3' in experiments_to_run:
        run_experiment_3()

    if 'all' in experiments_to_run or '4' in experiments_to_run:
        run_experiment_4()

    print("\nAll specified tasks have been completed.")
    print("Generated data is stored in the 'data/' directory.")
    print("Generated plots are stored in the 'figures/' directory.")

if __name__ == '__main__':
    main()
