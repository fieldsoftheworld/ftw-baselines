#!/usr/bin/env python3
"""Runs the train script with a grid of hyperparameters."""

import subprocess
from multiprocessing import Process, Queue
import torch

GPUS = [0,1,2,3]
DRY_RUN = False  # Set to False to actually run the experiments

# for experiment-3-1 and experiment-3-2 we are reusing experiment-2-1-3 and experiment-2-2-3
experiment_configs = [
    "configs/FTW-Release/2_class/cc-by-ftw",
    "configs/FTW-Release/3_class/cc-by-ftw",
    "configs/FTW-Release/2_class/full-ftw",
    "configs/FTW-Release/3_class/full-ftw"
]

def run_experiments(work: "Queue[str]") -> None:
    """Run experiments from the queue."""
    print(f"Running {work.qsize()} experiments")
    print(f"work.empty(): {work.empty()}")
    while not work.empty():
        experiment = work.get()
        print(f"Running: {experiment}")
        if not DRY_RUN:
            subprocess.call(experiment.split(" "))

if __name__ == "__main__":
    work: "Queue[str]" = Queue()

    # Add the experiments to the queue with the GPU index
    for config in experiment_configs:
        command = (
            f"ftw model fit"
            + f" --config {config}.yaml"
        )
        work.put(command)

    # Run sequentially if only one GPU is available
    run_experiments(work)

