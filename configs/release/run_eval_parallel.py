#!/usr/bin/env python3
"""Runs the eval script in parallel."""

import argparse
import os
import subprocess
from multiprocessing import Process, Queue

import pandas as pd
import yaml

# list of GPU IDs that we want to use, one job will be started for every ID in the list
GPUS = [0, 1, 2, 3, 4, 5, 6, 7]
DRY_RUN = False  # if False then print out the commands to be run, if True then run


COUNTRIES = [
    "austria",
    "belgium",
    "brazil",
    "cambodia",
    "corsica",
    "croatia",
    "denmark",
    "estonia",
    "finland",
    "france",
    "germany",
    "india",
    "kenya",
    "latvia",
    "lithuania",
    "luxembourg",
    "netherlands",
    "portugal",
    "rwanda",
    "slovakia",
    "slovenia",
    "south_africa",
    "spain",
    "sweden",
    "vietnam",
]

# This removes the countries with presence only data, and Portugal
FULL_DATA_COUNTRIES = [
    "austria",
    "belgium",
    "cambodia",
    "corsica",
    "croatia",
    "denmark",
    "estonia",
    "finland",
    "france",
    "germany",
    "latvia",
    "lithuania",
    "luxembourg",
    "netherlands",
    "slovakia",
    "slovenia",
    "south_africa",
    "spain",
    "sweden",
    "vietnam",
]


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

    existing_checkpoints = set()
    if os.path.exists(args.output_fn):
        df = pd.read_csv(args.output_fn)
        existing_checkpoints = set(df["train_checkpoint"].values)
    print(f"Found {len(existing_checkpoints)} existing checkpoints in {args.output_fn}")

    # Walk the user-specified root (defaults to 'logs/') for checkpoint folders
    search_root = args.search_root
    if not os.path.isdir(search_root):
        print(
            f"Warning: search_root '{search_root}' does not exist or is not a directory."
        )
    checkpoints = []
    for root, dirs, files in os.walk(search_root):
        for file in files:
            if file.endswith("last.ckpt"):
                parent_dir = os.path.dirname(root)
                config_file_path = os.path.join(parent_dir, "config.yaml")
                checkpoint_path = os.path.join(root, file)
                if os.path.isfile(config_file_path):
                    if checkpoint_path in existing_checkpoints:
                        print(f"Skipping existing checkpoint {checkpoint_path}")
                        continue

                    with open(config_file_path, "r") as conf_file:
                        config_data = yaml.safe_load(conf_file)

                    model_predicts_classes = (
                        config_data.get("model").get("init_args").get("num_classes", 3)
                    )
                    temporal_option = (
                        config_data.get("data")
                        .get("init_args")
                        .get("temporal_options", "stacked")
                    )
                    checkpoints.append(
                        (checkpoint_path, model_predicts_classes, temporal_option)
                    )
                else:
                    print(f"Missing config for checkpoint {root}")

    print(f"Running evaluation on {len(checkpoints)} checkpoints:")
    for ckpt, _, __ in checkpoints:
        print(f"  {ckpt}")

    for checkpoints_data in checkpoints:
        (checkpoint, model_predicts_classes, temporal_option) = checkpoints_data

        if args.country_eval:
            for country in COUNTRIES:
                command = [
                    "ftw",
                    "model",
                    "test",
                    "--gpu",
                    "GPU",
                    "--dir",
                    "data/ftw",
                    "--temporal_options",
                    temporal_option,
                    "--model",
                    checkpoint,
                    "--out",
                    args.output_fn,
                    "--countries",
                    country,
                ]
                if model_predicts_classes == 3:
                    command.append("--model_predicts_3_classes")
                if args.swap_order:
                    command.append("--swap_order")
                if args.split == "val":
                    command.append("--use_val_set")
                work.put(command)
        else:
            command = [
                "ftw",
                "model",
                "test",
                "--gpu",
                "GPU",
                "--dir",
                "data/ftw",
                "--temporal_options",
                temporal_option,
                "--model",
                checkpoint,
                "--out",
                args.output_fn,
            ]
            if model_predicts_classes == 3:
                command.append("--model_predicts_3_classes")
            if args.swap_order:
                command.append("--swap_order")
            if args.split == "val":
                command.append("--use_val_set")
            for country in FULL_DATA_COUNTRIES:
                command.append("--countries")
                command.append(country)
            work.put(command)

    processes = []
    for gpu_idx in GPUS:
        p = Process(target=do_work, args=(work, gpu_idx))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation")
    parser.add_argument(
        "--swap_order",
        action="store_true",
        help="Whether to swap the order of temporal images",
    )
    parser.add_argument(
        "--country_eval",
        action="store_true",
        help="Whether to evaluate on all countries separately",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val"],
        help="Which data split to use for evaluation",
    )
    parser.add_argument("--output_fn", required=True, type=str, help="Output filename")
    parser.add_argument(
        "--search_root",
        type=str,
        default="logs/",
        help="Root directory to recursively search for checkpoint directories (default: logs/)",
    )
    args = parser.parse_args()
    main(args)
