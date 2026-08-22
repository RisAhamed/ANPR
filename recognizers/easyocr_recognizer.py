"""
EasyOCR recognizer — production wrapper with graceful GPU/CPU fallback.
"""
import logging
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    import easyocr
except ImportError as e:
    raise ImportError("easyocr is required. pip install -r requirements.txt") from e


class EasyOCRRecognizer:
    def __init__(self, langs=None, use_gpu: bool = True):
        langs = langs or ["en"]
        try:
            self.reader = easyocr.Reader(langs, gpu=use_gpu)
            logger.info(f"EasyOCR loaded (gpu={use_gpu})")
        except Exception as ex:
            logger.warning(f"EasyOCR GPU init failed ({ex}), falling back to CPU")
            self.reader = easyocr.Reader(langs, gpu=False)

    def recognize_plate(self, plate_img: np.ndarray) -> Tuple[str, float]:
        """Return (text, confidence). Empty string if nothing detected."""
        if plate_img is None or plate_img.size == 0:
            return "", 0.0
        # EasyOCR expects RGB
        if len(plate_img.shape) == 3:
            img_rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = cv2.cvtColor(plate_img, cv2.COLOR_GRAY2RGB)

        results = self.reader.readtext(
            img_rgb,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            detail=1,
        )
        if not results:
            return "", 0.0
        # Pick highest confidence result, join if multiple words
        # results: list of (bbox, text, conf)
        best = max(results, key=lambda x: x[2])
        _, text, conf = best
        # If multiple detections, concatenate close-by texts
        if len(results) > 1:
            # sort left-to-right by bbox x
            results_sorted = sorted(results, key=lambda x: x[0][0][0] if isinstance(x[0], list) else 0)
            text = "".join([t for _, t, _ in results_sorted])
            conf = float(np.mean([c for _, _, c in results]))
        return text.strip(), float(conf)
