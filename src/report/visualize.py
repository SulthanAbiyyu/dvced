import yaml
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def target_countplot(config):
    data = pd.read_csv(config['data_load']['raw_data'], delimiter=";")
    target = config['base']['target_column']

    sns.countplot(data[target])
    plt.savefig(f"{config['visualize']['output_dir']}/target_countplot.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    target_countplot(config)
    print('visualization done')
