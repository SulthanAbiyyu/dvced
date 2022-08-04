import yaml
import joblib
import json
import argparse
import pandas as pd

from src.report.confusion_matrix import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, f1_score


def evaluate(config_path):

    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_path = config["train"]["model_path"]
    test_data_path = config["featurize"]["output_test_features_path"]
    target_col = config["base"]["target_column"]
    estimator = joblib.load(model_path)
    test_data_feat = pd.read_csv(test_data_path)

    X = test_data_feat.drop([target_col], axis=1)
    y = test_data_feat[target_col]

    y_pred = estimator.predict(X)

    f1 = f1_score(y, y_pred, average="macro")
    cm = confusion_matrix(y, y_pred)

    report = {
        "f1": f1,
        "confusion_matrix": cm.tolist(),
        "actual": y.tolist(),
        "predicted": y_pred.tolist()
    }

    reports_folder = config["evaluate"]["reports_dir"]
    metrics_path = f"{reports_folder}/{config['evaluate']['metrics_file']}"

    json.dump(
        obj=report,
        fp=open(metrics_path, "w"),
    )

    plt = plot_confusion_matrix(cm, target_names=["0", "1"], normalize=False)
    plt.savefig(
        f"{reports_folder}/{config['evaluate']['confusion_matrix_file']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, dest="config")
    args = parser.parse_args()

    evaluate(args.config)
    print("Evaluation complete")
