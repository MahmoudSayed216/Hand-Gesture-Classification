import mlflow 
import logging 
from pathlib import Path


def setup_mlflow_experiment(experiment_name: str, tracking_uri: str = "http://localhost:5000") -> str:
    mlflow.set_tracking_uri(tracking_uri)
    exp = mlflow.set_experiment(experiment_name)
    return exp.experiment_id

def log_model_with_mlflow(model, metrics, model_name: str, exp_id: str, output_dir: Path):
    with mlflow.start_run(experiment_id=exp_id, run_name=model_name) as run:
        logging.info(f"Logging {model_name} to MLflow...")

        mlflow.set_tag("model", model_name)

        mlflow.log_params(model.best_params_)

        mlflow.log_metrics({
            "Accuracy": metrics['accuracy'],
            "Precision": metrics['precision'],
            "Recall": metrics['recall'],
            "f1-score": metrics['f1'],
        })

        mlflow.log_artifact(str(output_dir / "ROC_curve.png"))
