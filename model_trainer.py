# Load model directly
# from transformers import AutoImageProcessor, AutoModelForObjectDetection

# processor = AutoImageProcessor.from_pretrained("nickmuchi/yolos-small-finetuned-license-plate-detection")
# model = AutoModelForObjectDetection.from_pretrained("nickmuchi/yolos-small-finetuned-license-plate-detection")
import os
import warnings
from ultralytics import YOLO, settings
import mlflow
import dagshub


warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")


CONFIG = {
"mlflow_tracking_uri": "https://dagshub.com/RisAhamed/ANPR.mlflow",
"dagshub_repo_owner": "RisAhamed",
"dagshub_repo_name": "ANPR",
"bucket_name": "ANPR",
"endpoint_url": "https://dagshub.com/api/v1/repo-buckets/s3/RisAhamed",
"public_key_id": "822dabf3de2f482e09baa3a4dd7259fafbc8bda8"
}


def setup_mlflow_tracking():
    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])

        # Initialize DagsHub
        dagshub.init(
            repo_owner=CONFIG["dagshub_repo_owner"], 
            repo_name=CONFIG["dagshub_repo_name"], 
            mlflow=True
        )
        
        # Set environment variables for authentication
        os.environ["MLFLOW_TRACKING_USERNAME"] = CONFIG["public_key_id"]
        os.environ["MLFLOW_TRACKING_PASSWORD"] = CONFIG["public_key_id"]
        
        # Set experiment name
        mlflow.set_experiment("ANPR")
    
        print("MLflow and DagsHub tracking setup successful.")
    except Exception as e:
     print(f"Error setting up MLflow tracking: {e}")

def train_yolov8(model_size="yolov8x", data_path="LicensePlate-1/data.yaml", epochs=50):


    setup_mlflow_tracking()


    with mlflow.start_run():
        try:
        # Load model (n = nano, s = small, m = medium, l = large, x = extra)
            model = YOLO(f"{model_size}.pt")

                # Log training parameters
            mlflow.log_params({
                    "model_size": model_size,
                    "data_path": data_path,
                    "epochs": epochs,
                    "image_size": 640,
                    "batch_size": 16
                })

                # Train the model
            results = model.train(
                    data=data_path,
                    epochs=epochs,
                    imgsz=640,
                    batch=16,
                    project="anpr-yolov8",
                    name=f"{model_size}-finetuned",
                    exist_ok=True
                )

                # Validate the model
            model.val()

                # Log metrics (assuming results contains performance metrics)
            if hasattr(results, 'box'):
                mlflow.log_metrics({
                    "precision": results.box.map,
                    "recall": results.box.mar,
                    "mAP50": results.box.map50,
                    "mAP50-95": results.box.map
                })

                # Log the trained model
            mlflow.log_artifact(f"runs/detect/anpr-yolov8/{model_size}-finetuned/weights/best.pt")

        except Exception as e:
            mlflow.log_param("training_error", str(e))
            print(f"Training error: {e}")
        
if __name__ =="__main__":
    train_yolov8()