#!/usr/bin/env python3
"""Runs the train script with a grid of hyperparameters."""

import itertools
import os
import subprocess
from multiprocessing import Process, Queue
import torch
import glob
import random

GPUS = [1,2,3,4,5,6,7]
DRY_RUN = False  # if True then print out the commands to be run, if False then run

# Hyperparameter options
model = "unet"
backbone = "efficientnet-b3"
lr = 1e-3
loss = "ce"
seed = 7
batch_size = 32
patch_weights = False

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

    yml_files = glob.glob(os.path.join("configs/FTW-25-Experiments-1-3/", '*.yaml'))
    print(f"Running on {len(yml_files)} files")

    for experiment_config_file in yml_files:
        filename = os.path.splitext(experiment_config_file.split("/")[-1])[0]
        experiment_name = f"train_on_all_ftw_{model}_{backbone}_{lr}_{loss}_{seed}_{filename}"

        log_dir = os.path.join("logs", "train_on_all_ftw", experiment_name)
        config_file = experiment_config_file

        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        command = (
            "python main.py fit"
            + f" --config {config_file}"
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
