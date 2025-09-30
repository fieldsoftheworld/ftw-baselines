import argparse
import os
import subprocess

import pandas as pd
import yaml

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


def main(args):
    existing_checkpoints = set()
    if os.path.exists(args.output_fn):
        df = pd.read_csv(args.output_fn)
        existing_checkpoints = set(df["train_checkpoint"].values)
    print(f"Found {len(existing_checkpoints)} existing checkpoints in {args.output_fn}")

    checkpoints = []
    for root, dirs, files in os.walk("logs/"):
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

        # First test on the full test set
        command = [
            "ftw",
            "model",
            "test",
            "--gpu",
            str(args.gpu),
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
        subprocess.call(command)

        # Then test on each country individually
        if args.country_eval:
            for country in COUNTRIES:
                command = [
                    "ftw",
                    "model",
                    "test",
                    "--gpu",
                    str(args.gpu),
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
                if args.split == "val":
                    command.append("--use_val_set")
                if args.swap_order:
                    command.append("--swap_order")
                subprocess.call(command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation")
    parser.add_argument("--gpu", type=int, required=True, help="GPU ID to use")
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
    args = parser.parse_args()
    main(args)
