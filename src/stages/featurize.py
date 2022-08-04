import yaml
import argparse
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelBinarizer


def gender_binarizer(train_data, test_data):

    lb = LabelBinarizer()
    train_data["gender"] = lb.fit_transform(train_data["gender"].ravel())
    test_data["gender"] = lb.transform(test_data["gender"].ravel())

    return train_data, test_data


def age_minmaxscaling(train_data, test_data):

    mms = MinMaxScaler()
    train_data["age"] = mms.fit_transform(
        train_data["age"].ravel().reshape(-1, 1))
    test_data["age"] = mms.transform(test_data["age"].ravel().reshape(-1, 1))

    return train_data, test_data


def featurize(config_path):

    with open(config_path) as f:
        config = yaml.safe_load(f)

    train_data_path = config["data_split"]["output_train_path"]
    test_data_path = config["data_split"]["output_test_path"]

    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)

    train_data, test_data = gender_binarizer(train_data, test_data)
    train_data, test_data = age_minmaxscaling(train_data, test_data)

    featurize_train_path = config["featurize"]["output_train_features_path"]
    featurize_test_path = config["featurize"]["output_test_features_path"]

    train_data.to_csv(featurize_train_path, index=False)
    test_data.to_csv(featurize_test_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', required=True)
    args = parser.parse_args()

    featurize(args.config)
    print('featurize done')
