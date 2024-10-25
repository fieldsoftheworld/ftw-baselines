import subprocess
import os
import yaml

if __name__ == "__main__":
    countries = [
        "Cambodia"
    ]

    checkpoints = []
    for root, dirs, files in os.walk("logs/FTW-25-Experiments-2-2"):
        for file in files:
            if file.endswith("last.ckpt"):
                checkpoints.append(os.path.join(root, file))

    for checkpoints_data in checkpoints:
        checkpoint = checkpoints_data
        command = [
            "ftw model test",
            "--gpu", "1",
            "--checkpoint", checkpoint,
            "--output", "results/experiments-2-2.csv",
            "--countries", "cambodia", "vietnam",
            "--model_predicts_3_classes", "True"
        ]
        subprocess.call(command)