#!/usr/bin/env python3
"""Runs the train script with a grid of hyperparameters."""

import os
import itertools
import subprocess
from multiprocessing import Process, Queue
import torch

DRY_RUN = False  # Set to False to actually run the experiments
num_samples_options = [1,10,50,100,500,1000,-1]
seeds= [7, 43, 24]

experiment_configs = [
    {
        "config": "configs/FTW-25-Experiments-3/finetuning/experiment-3-1_finetuning.yaml",
        "ckpt": "logs/FTW-25-Experiments-2-1/experiment-2-1.3/lightning_logs/version_17706488/checkpoints/last.ckpt", #TODO:Change checkpoints path if needed, use checkpoint from 2.1.3
        "exp_name" : "experiment-3.1-finetuning",
        "limit" : len(num_samples_options)
    },
     {
        "config": "configs/FTW-25-Experiments-3/finetuning/experiment-3-2_finetuning.yaml",
        "ckpt": "logs/FTW-25-Experiments-2-2/experiment-2-2.3/lightning_logs/version_17706488/checkpoints/last.ckpt", #TODO: Change checkpoints path if needed, use checkpoint from 2.2.3
        "exp_name" : "experiment-3.2-finetuning",
        "limit" : 6
    },
     {
        "config": "configs/FTW-25-Experiments-3/finetuning/experiment-3-3_finetuning.yaml",
        "ckpt": "logs/FTW-25-Experiments-3/experiment-3.3/lightning_logs/version_17706488/checkpoints/last.ckpt", #TODO: Change checkpoints path if needed
        "exp_name" : "experiment-3.3-finetuning",
        "limit" : len(num_samples_options)
    },
     {
        "config": "configs/FTW-25-Experiments-3/finetuning/experiment-3-4_finetuning.yaml",
        "ckpt": "logs/FTW-25-Experiments-3/experiment-3.4/lightning_logs/version_17706488/checkpoints/last.ckpt", #TODO: Change checkpoints path if needed
        "exp_name" : "experiment-3.4-finetuning",
        "limit" : len(num_samples_options)
    },
     {
        "config": "configs/FTW-25-Experiments-3/finetuning/experiment-3-5_finetuning.yaml",
        "ckpt": "logs/FTW-25-Experiments-3/experiment-3.5/lightning_logs/version_17706488/checkpoints/last.ckpt", #TODO: Change checkpoints path if needed
        "exp_name" : "experiment-3.5-finetuning",
        "limit" : 7
    },
]

def run_experiments(work: "Queue[str]"):
    """Run experiments from the queue."""
    print(f"Running {work.qsize()} experiments")
    print(f"work.empty(): {work.empty()}")
    while not work.empty():
        experiment = work.get()
        print(f"Running: {experiment}")
        if not DRY_RUN:
            subprocess.call(experiment.split(" "))

if __name__ == "__main__":
    # Check the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print('Available GPUs: ', num_gpus)
    GPUS = list(range(num_gpus))
    
    work: "Queue[str]" = Queue()
    gpu_index = 0
    for experiment_config in experiment_configs:
        num_samples_options_limit = experiment_config['limit']

        for idx in range(0, num_samples_options_limit):
            for seed in seeds:
                num_samples = num_samples_options[idx]
                experiment_name = f"{experiment_config['exp_name']}_{seed}_fts_{num_samples}"    # fts - fine tuning samples      
                log_dir = os.path.join("logs", "FTW-25-Experiments-3", experiment_name)

                command = (
                    f"python main.py fit "
                    f"--config {experiment_config['config']} "
                    f"--ckpt_path {experiment_config['ckpt']} "
                    f"--data.init_args.num_samples {int(num_samples)} "
                    f"--trainer.default_root_dir {log_dir} "
                    f"--seed_everything {seed}"
                )
                work.put(command)
                if gpu_index < num_gpus - 1:
                    gpu_index += 1
                else:
                    gpu_index = 0
                

    if num_gpus > 1:
        # Spawn a process for each GPU separately and run the experiments in parallel
        processes = []
        for gpu_idx in GPUS:
            p = Process(target=run_experiments, args=(work,))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        # Run sequentially if only one GPU is available
        run_experiments(work)
