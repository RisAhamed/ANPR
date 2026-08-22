"""
YOLO training with MLflow + DagsHub — secrets via env.
"""
import os
import warnings
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from ultralytics import YOLO
import mlflow
import dagshub
import anpr_config as cfg

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")


def setup_mlflow_tracking() -> bool:
    token = cfg.DAGSHUB_TOKEN
    if not token:
        print("WARN: DAGSHUB_TOKEN not set — skipping MLflow/DagsHub tracking.")
        return False
    try:
        mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)
        dagshub.init(repo_owner=cfg.DAGSHUB_REPO_OWNER, repo_name=cfg.DAGSHUB_REPO_NAME, mlflow=True)
        os.environ["MLFLOW_TRACKING_USERNAME"] = token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
        mlflow.set_experiment("ANPR")
        print("MLflow and DagsHub tracking setup successful.")
        return True
    except Exception as e:
        print(f"Error setting up MLflow tracking: {e}")
        return False


def train_yolov8(model_size="yolov8n", data_path="LicensePlate-1/data.yaml", epochs=50):
    use_mlflow = setup_mlflow_tracking()

    # context manager only if mlflow available
    ctx = mlflow.start_run() if use_mlflow else _noop_context()
    with ctx:
        try:
            model = YOLO(f"{model_size}.pt")

            if use_mlflow:
                mlflow.log_params({
                    "model_size": model_size,
                    "data_path": data_path,
                    "epochs": epochs,
                    "image_size": 640,
                    "batch_size": 16,
                })

            results = model.train(
                data=data_path,
                epochs=epochs,
                imgsz=640,
                batch=16,
                project="runs/detect",
                name=f"{model_size}-finetuned",
                exist_ok=True,
            )

            model.val()

            if use_mlflow and hasattr(results, 'box'):
                mlflow.log_metrics({
                    "mAP50": float(getattr(results.box, 'map50', 0) or 0),
                    "mAP50-95": float(getattr(results.box, 'map', 0) or 0),
                })
                best = Path(f"runs/detect/{model_size}-finetuned/weights/best.pt")
                if best.exists():
                    mlflow.log_artifact(str(best))

        except Exception as e:
            if use_mlflow:
                try:
                    mlflow.log_param("training_error", str(e))
                except Exception:
                    pass
            print(f"Training error: {e}")
            raise


class _noop_context:
    def __enter__(self): return self
    def __exit__(self, *a): return False


if __name__ == "__main__":
    train_yolov8()
