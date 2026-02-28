import mlflow 
import logging 
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def setup_mlflow_experiment(experiment_name: str, tracking_uri: str = "file:./mlruns") -> str:
    mlflow.set_tracking_uri(tracking_uri)
    exp = mlflow.set_experiment(experiment_name)
    return exp.experiment_id


def log_model_with_mlflow(exp_id: str, model, metrics, model_name: str, output_dir: Path):

    with mlflow.start_run(experiment_id=exp_id, run_name=model_name):

        logging.info(f"Logging {model_name} to MLflow...")

        # -----------------------
        # Tags
        # -----------------------
        mlflow.set_tag("model", model_name)

        # -----------------------
        # Parameters
        # -----------------------
        mlflow.log_params(model.get_params())

        # -----------------------
        # Metrics (scalars only)
        # -----------------------
        mlflow.log_metrics({
            "Accuracy": metrics['Accuracy'],
            "Precision": metrics['Precision'],
            "Recall": metrics['Recall'],
            "f1-score": metrics['f1'],
        })

        # -----------------------
        # Confusion Matrix (%)
        # -----------------------
        cm = np.array(metrics["Confusion Matrix"])

        # If matrix is raw counts, normalize it row-wise
        if cm.max() > 1:
            cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        plt.figure()
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2%",
            cmap="Blues"
        )

        plt.title("Confusion Matrix (Row Normalized %)")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")

        mlflow.log_figure(plt.gcf(), "confusion_matrix_percentage.png")
        plt.close()

        # -----------------------
        # Model Saving (Proper MLflow way)
        # -----------------------
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model"  # <- simple folder name
    )