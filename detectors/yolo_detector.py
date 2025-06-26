from ultralytics import YOLO
import numpy as np

class YOLODetector:
    def __init__(self, weights_path):
        self.model = YOLO(weights_path)

    def detect_plates(self, image, conf_threshold=0.4):
        """
        Detects license plates in an image.

        Args:
            image: The input image (np.ndarray).
            conf_threshold: The confidence threshold for detection.

        Returns:
            A list of bounding boxes in the format [x1, y1, x2, y2, conf].
        """
        # Pass the confidence threshold directly to the model
        # This is more efficient as YOLO filters the results internally
        results = self.model(image, conf=conf_threshold, verbose=False)
        
        boxes = []
        # The results object now only contains boxes above the confidence threshold
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            boxes.append([x1, y1, x2, y2, conf])
            
        return boxes