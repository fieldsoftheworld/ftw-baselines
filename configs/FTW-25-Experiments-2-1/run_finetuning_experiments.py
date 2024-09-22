#!/usr/bin/env python3
"""Runs the train script with a grid of hyperparameters."""
import subprocess
from multiprocessing import Process, Queue

GPUS = [0,2,3]
DRY_RUN = False  # Set to False to actually run the experiments

experiment_configs = [
    {
        "config": "configs/FTW-25-Experiments-2-1/finetuning/experiment-2-1.1_finetuning.yaml",
        "ckpt": "logs/FTW-25-Experiments-2-1/experiment-2-1.1/lightning_logs/version_0/checkpoints/last.ckpt"
    },
    {
        "config": "configs/FTW-25-Experiments-2-1/finetuning/experiment-2-1.2_finetuning.yaml",
        "ckpt": "logs/FTW-25-Experiments-2-1/experiment-2-1.2/lightning_logs/version_0/checkpoints/last.ckpt"
    },
    {
        "config": "configs/FTW-25-Experiments-2-1/finetuning/experiment-2-1.3_finetuning.yaml",
        "ckpt": "logs/FTW-25-Experiments-2-1/experiment-2-1.3/lightning_logs/version_0/checkpoints/last.ckpt"
    }
]

def run_experiments(work: "Queue[str]", gpu_idx: int) -> None:
    """Run experiments from the queue."""
    while not work.empty():
        experiment = work.get()
        experiment = experiment.replace("GPU", str(gpu_idx))
        if not DRY_RUN:
            subprocess.call(experiment.split(" "))

if __name__ == "__main__":
    work: "Queue[str]" = Queue()

    for experiment_config in experiment_configs:
        command = (
            f"ftw model fit"
            + f" --config {experiment_config['config']}"
            + f" --ckpt_path {experiment_config['ckpt']}"
            + f" --trainer.devices [GPU]"
        )
        work.put(command)

    if len(GPUS) > 1:
        # Spawn a process for each GPU separately and run the experiments in parallel
        processes = []
        for gpu_idx in GPUS:
            p = Process(target=run_experiments, args=(work, gpu_idx))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        # Run sequentially if only one GPU is available
        run_experiments(work, GPUS[0])
