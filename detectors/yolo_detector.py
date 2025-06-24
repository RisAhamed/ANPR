from ultralytics import YOLO
import numpy as np

class YOLODetector:
    def __init__(self, weights_path):
        self.model = YOLO(weights_path)

    def detect_plates(self, image):
        results = self.model(image)
        boxes = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            boxes.append([x1, y1, x2, y2, conf])
        return boxes
