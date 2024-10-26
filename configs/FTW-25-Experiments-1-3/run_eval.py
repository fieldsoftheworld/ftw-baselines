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
    for root, dirs, files in os.walk("logs/FTW-25-Experiments-1-3"):
        for file in files:
            if file.endswith("last.ckpt"):
                # checkpoints.append(os.path.join(root, file))

                parent_dir = os.path.dirname(root)
                config_file_path = os.path.join(parent_dir, "config.yaml")
                if os.path.isfile(config_file_path):
                    with open(config_file_path, 'r') as conf_file:
                        config_data = yaml.safe_load(conf_file)

                    temporal_option = config_data.get('data').get('init_args').get('temporal_options', "stacked")
                    checkpoints.append((os.path.join(root, file), temporal_option))
                else:
                    print(f'Missing config for checkpoint {root}')

    for checkpoints_data in checkpoints:
        (checkpoint, temporal_op) = checkpoints_data
        for country in countries:
            command = [
                "ftw model test",
                "--gpu", "0",
                "--model", checkpoint,
                "--out", "with_prp_all_results_train_all_contries_3_class.csv",
                "--countries", country,
                "--model_predicts_3_classes", "True",
                "--temporal_options", str(temporal_op),
            ]
            subprocess.call(command)
