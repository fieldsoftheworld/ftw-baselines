#!/usr/bin/env python3
"""Runs the check_middle_sensitivity script in parallel."""

import argparse
import os
import subprocess
from multiprocessing import Process, Queue

import pandas as pd
import yaml

# list of GPU IDs that we want to use, one job will be started for every ID in the list
GPUS = [0,1,2,7]
DRY_RUN = False  # if False then print out the commands to be run, if True then run


def do_work(work: "Queue[list[str]]", gpu_idx: int) -> bool:
    """Process for each ID in GPUS."""
    while not work.empty():
        command = work.get()
        for i in range(len(command)):
            if command[i] == "GPU":
                command[i] = str(gpu_idx)
                break
        print(command)
        if not DRY_RUN:
            subprocess.call(command)
    return True


def main(args: argparse.Namespace):
    work: "Queue[list[str]]" = Queue()

    # Check if output directory exists, create if needed
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    existing_files = set()
    if args.skip_existing:
        # Check for existing consensus results files in the output directory
        for file in os.listdir(args.output_dir):
            if file.startswith("consensus_results_") and file.endswith(".csv"):
                # Extract the model name from the filename
                model_name = file[len("consensus_results_") : -len(".csv")]
                existing_files.add(model_name)
        print(
            f"Found {len(existing_files)} existing consensus results files in {args.output_dir}"
        )

    # Walk the user-specified root (defaults to 'logs/') for checkpoint folders
    search_root = args.search_root
    if not os.path.isdir(search_root):
        print(
            f"Warning: search_root '{search_root}' does not exist or is not a directory."
        )
        return

    checkpoints = []
    for root, dirs, files in os.walk(search_root):
        for file in files:
            if file.endswith("last.ckpt"):
                parent_dir = os.path.dirname(root)
                config_file_path = os.path.join(parent_dir, "config.yaml")
                checkpoint_path = os.path.join(root, file)

                if os.path.isfile(config_file_path):
                    # Generate model name for this checkpoint
                    if args.use_checkpoint_name:
                        model_name = os.path.splitext(
                            os.path.basename(checkpoint_path)
                        )[0]
                    else:
                        # Use directory structure as model name
                        rel_path = os.path.relpath(checkpoint_path, search_root)
                        model_name = rel_path.replace(os.sep, "_").replace(".ckpt", "")

                    # Check if we should skip this checkpoint
                    if args.skip_existing and model_name in existing_files:
                        print(
                            f"Skipping existing checkpoint {checkpoint_path} (output exists)"
                        )
                        continue

                    checkpoints.append((checkpoint_path, model_name))
                else:
                    print(f"Missing config for checkpoint {root}")

    print(f"Running consensus analysis on {len(checkpoints)} checkpoints:")
    for ckpt, model_name in checkpoints:
        print(f"  {ckpt} -> {model_name}")

    for checkpoint_path, model_name in checkpoints:
        # Construct the output filename
        output_file = os.path.join(
            args.output_dir, f"consensus_results_{model_name}.csv"
        )

        command = [
            "python",
            "configs/release/check_middle_sensitivity.py",
            "--model",
            checkpoint_path,
            "--gpu",
            "GPU",
            "--output_fn",
            output_file,  # This argument is required but not actually used by the script
            "--name",
            model_name,
        ]

        work.put(command)

    processes = []
    for gpu_idx in GPUS:
        p = Process(target=do_work, args=(work, gpu_idx))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()

    print(f"\nAll consensus analysis jobs completed!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run consensus analysis in parallel")
    parser.add_argument(
        "--search_root",
        type=str,
        default="logs/",
        help="Root directory to recursively search for checkpoint directories (default: logs/)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save consensus results files",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip checkpoints that already have consensus results files",
    )
    parser.add_argument(
        "--use_checkpoint_name",
        action="store_true",
        help="Use checkpoint filename as model name instead of directory structure",
    )
    args = parser.parse_args()
    main(args)
