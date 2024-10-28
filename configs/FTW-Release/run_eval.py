import subprocess
import os
import yaml

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

    checkpoints = []
    for root, dirs, files in os.walk("logs/"):
        for file in files:
            if file.endswith("last.ckpt"):
                parent_dir = os.path.dirname(root)
                config_file_path = os.path.join(parent_dir, "config.yaml")
                if os.path.isfile(config_file_path):
                    with open(config_file_path, 'r') as conf_file:
                        config_data = yaml.safe_load(conf_file)

                    model_predicts_classes = config_data.get('model').get('init_args').get('num_classes', 3)
                    checkpoints.append((os.path.join(root, file), model_predicts_classes))
                else:
                    print(f'Missing config for checkpoint {root}')

    for checkpoints_data in checkpoints:
        (checkpoint, model_predicts_classes) = checkpoints_data
        for country in countries:
            if model_predicts_classes == 2:
                # Test on the same country
                command = [
                    "ftw", "model", "test",
                    "--gpu", "0",
                    "--dir", "data/ftw",
                    "--model", checkpoint,
                    "--out", "results/experiments-ftw_release-2_classes.csv",
                    "--countries", country
                ]
                subprocess.call(command)
            elif model_predicts_classes == 3:
                # Test on the same country
                command = [
                    "ftw", "model", "test",
                    "--gpu", "0",
                    "--dir", "data/ftw",
                    "--model", checkpoint,
                    "--out", "results/experiments-ftw_release-3_classes.csv",
                    "--countries", country,
                    "--model_predicts_3_classes", "True"
                ]
                subprocess.call(command)