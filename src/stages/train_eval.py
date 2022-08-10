import argparse
import mlflow

from src.stages.train import train
from src.stages.evaluate import evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, dest="config")
    args = parser.parse_args()

    with mlflow.start_run():
        estimator_name, params = train(args.config)
        f1 = evaluate(args.config)

        mlflow.log_param("estimator_name", estimator_name)
        for param, value in params.items():
            mlflow.log_param(param, value)
        
        mlflow.log_metric("f1", f1)
        mlflow.log_artifact("reports/confusion_matrix.png")
        print("Training and evaluation complete")