#!/usr/bin/env python3
"""Runs the train script with a grid of hyperparameters."""

import os
import itertools
import subprocess
from multiprocessing import Process, Queue

GPUS = [1,2,3,4,5]
DRY_RUN = False  # Set to False to actually run the experiments

model_options = ["unet", "deeplabv3+"]
backbone_options = ["resnet18", "resnet50", "resnext50_32x4d", "efficientnet-b3", "efficientnet-b4"]

def do_work(work: "Queue[str]", gpu_idx: int) -> bool:
    """Process for each ID in GPUS."""
    while not work.empty():
        experiment = work.get()
        experiment = experiment.replace("GPU", str(gpu_idx))
        print(experiment)
        if not DRY_RUN:
            subprocess.call(experiment.split(" "))
    return True

if __name__ == "__main__":
    work: "Queue[str]" = Queue()

    for model, backbone in itertools.product(
        model_options,
        backbone_options
    ):
        experiment_name = f"{model}_{backbone}"
        log_dir = os.path.join("logs", "FTW-25-Experiments-1-2", experiment_name)
        config_file = os.path.join("configs", "FTW-25-Experiments-1-2", "experiment-1-2.yaml")

        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        command = (
            "ftw model fit"
            + f" --config {config_file}"
            + f" --trainer.default_root_dir {log_dir}"
            + f" --model.init_args.model {model}"
            + f" --model.init_args.backbone {backbone}"
            + f" --trainer.devices [GPU]"
        )
        command = command.strip()
        work.put(command)

    if len(GPUS) > 1:
        # Use parallel processing if multiple GPUs are available
        processes = []
        for gpu_idx in GPUS:
            p = Process(target=do_work, args=(work, gpu_idx))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
    else:
        # Run sequentially if only one GPU is available
        do_work(work, 0)
