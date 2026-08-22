"""
YOLO detector — supports both standard bbox and OBB models.
Production-grade: lazy load, device auto-select, clear errors.
"""
import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
except ImportError as e:
    raise ImportError("ultralytics is required. pip install -r requirements.txt") from e

Box = Tuple[int, int, int, int, float]  # x1,y1,x2,y2,conf


class YOLODetector:
    def __init__(self, weights_path: str | Path, device: str = "auto"):
        self.weights_path = Path(weights_path)
        if not self.weights_path.exists():
            raise FileNotFoundError(f"YOLO weights not found: {self.weights_path}")
        self.device = device
        logger.info(f"Loading YOLO model: {self.weights_path}")
        self.model = YOLO(str(self.weights_path))
        # device handling: ultralytics auto-selects; override if needed
        if device != "auto":
            try:
                self.model.to(device)
            except Exception:
                logger.warning(f"Could not move model to {device}, using default.")

    def detect_plates(self, image: np.ndarray, conf_threshold: float = 0.4) -> List[Box]:
        """Detect plates / vehicles. Handles OBB and regular boxes."""
        if image is None or image.size == 0:
            return []
        results = self.model(image, verbose=False)[0]
        boxes: List[Box] = []

        # OBB path (vehicle model is OBB)
        if hasattr(results, "obb") and results.obb is not None and len(results.obb) > 0:
            for obb in results.obb.data.tolist():
                # obb: [x_c, y_c, w, h, angle, conf, cls]
                if len(obb) < 7:
                    continue
                x_c, y_c, w, h, angle, conf, _ = obb
                if conf < conf_threshold:
                    continue
                rect = ((x_c, y_c), (w, h), angle * 180 / np.pi)
                pts = cv2.boxPoints(rect)
                x1, y1 = pts.min(axis=0).astype(int)
                x2, y2 = pts.max(axis=0).astype(int)
                # clip to image
                h_img, w_img = image.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_img, x2), min(h_img, y2)
                if x2 > x1 and y2 > y1:
                    boxes.append((int(x1), int(y1), int(x2), int(y2), float(conf)))
            return boxes

        # Standard bbox path
        if hasattr(results, "boxes") and results.boxes is not None:
            for box, conf in zip(results.boxes.xyxy, results.boxes.conf):
                if float(conf) < conf_threshold:
                    continue
                x1, y1, x2, y2 = map(int, box.tolist())
                boxes.append((x1, y1, x2, y2, float(conf)))
        return boxes

    def detect(self, image: np.ndarray, conf_threshold: float = 0.4) -> List[Box]:
        """Alias for compatibility."""
        return self.detect_plates(image, conf_threshold)
