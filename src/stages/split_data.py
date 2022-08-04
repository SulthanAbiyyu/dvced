import yaml
import argparse
import pandas as pd

from load_data import load_data
from sklearn.model_selection import train_test_split


def split_data(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    data: pd.DataFrame = load_data(config)

    train_data, test_data = train_test_split(
        data, test_size=config['data_split']['test_size'],
        random_state=config['base']['random_state']
    )

    train_data_path = config['data_split']['output_train_path']
    test_data_path = config['data_split']['output_test_path']

    train_data.to_csv(train_data_path, index=False)
    test_data.to_csv(test_data_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', required=True)
    args = parser.parse_args()

    split_data(args.config)
    print('data split done')
