import subprocess
import os
import yaml

if __name__ == "__main__":
    countries = [
        "india"
    ]

    checkpoints = []
    for root, dirs, files in os.walk("logs/FTW-25-Experiments-2-1"):
        for file in files:
            if file.endswith("last.ckpt"):
                checkpoints.append(os.path.join(root, file))

    for checkpoints_data in checkpoints:
        checkpoint = checkpoints_data
        for country in countries:
            command = [
                "ftw model test",
                "--gpu", "0",
                "--model", checkpoint,
                "--output", "results/experiments-2-1.csv",
                "--countries", country,
                "--model_predicts_3_classes", "True"
            ]
            subprocess.call(command)