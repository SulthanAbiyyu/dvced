import yaml
import argparse
import joblib
import pandas as pd
from xgboost import XGBClassifier

def get_estimator():
    return {
        "xgboost": XGBClassifier
    }


def train(config_path):

    with open(config_path) as f:
        config = yaml.safe_load(f)

    train_data_path = config["featurize"]["output_train_features_path"]
    estimator_name = config["train"]["estimator_name"]
    params = config["train"]["estimators"][estimator_name]["params"]

    train_data_feat = pd.read_csv(train_data_path)
    target_col = config["base"]["target_column"]
    X = train_data_feat.drop([target_col], axis=1)
    y = train_data_feat[target_col]

    if estimator_name not in get_estimator().keys():
        raise ValueError("Estimator name not found")

    estimator = get_estimator()[estimator_name](**params)
    estimator.fit(X, y)

    model_path = config["train"]["model_path"]
    joblib.dump(estimator, model_path)

    return estimator_name, params

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config", required=True)
    args = parser.parse_args()

    train(args.config)
    print("Training complete")
