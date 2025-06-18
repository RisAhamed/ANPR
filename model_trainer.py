# Load model directly
# from transformers import AutoImageProcessor, AutoModelForObjectDetection

# processor = AutoImageProcessor.from_pretrained("nickmuchi/yolos-small-finetuned-license-plate-detection")
# model = AutoModelForObjectDetection.from_pretrained("nickmuchi/yolos-small-finetuned-license-plate-detection")
from ultralytics import YOLO

def train_yolov8(model_size="yolov8x", data_path="LicensePlate-1/data.yaml", epochs=50):
    # Load model (n = nano, s = small, m = medium, l = large, x = extra)
    model = YOLO(f"{model_size}.pt")

    # Train the model
    model.train(
        data=data_path,
        epochs=epochs,
        imgsz=640,
        batch=16,
        project="anpr-yolov8",
        name=f"{model_size}-finetuned",
        exist_ok=True
    )


    model.val()

    # # Optional: Export model
    # model.export(format="onnx")

if __name__ == "__main__":
    train_yolov8()
