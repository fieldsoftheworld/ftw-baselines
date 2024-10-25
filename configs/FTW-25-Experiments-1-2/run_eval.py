import subprocess
import os

if __name__ == "__main__":
    countries = [
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
        "vietnam"
    ]

    countries = [
        "slovenia",
        "south_africa",
        "france"
    ]

    checkpoints = []
    for root, dirs, files in os.walk("logs/FTW-25-Experiments-1-2"):
        for file in files:
            if file.endswith("last.ckpt"):
                checkpoints.append(os.path.join(root, file))

    for checkpoint in checkpoints:
        for country in countries:
            command = [
                "ftw model test",
                "--gpu", "7",
                "--model", checkpoint,
                "--out", "results/experiments-1-2.csv",
                "--countries", country,
                "--model_predicts_3_classes", "True"
            ]
            subprocess.call(command)