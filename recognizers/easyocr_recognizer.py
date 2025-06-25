import easyocr
import numpy as np

class EasyOCRRecognizer:
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=False)

    def recognize_plate(self, image):
        # image: np.ndarray (grayscale or RGB)
        result = self.reader.readtext(image)
        best_text, best_conf = '', 0
        for detection in result:
            text, conf = detection[1], detection[2]
            if conf > best_conf and len(text.strip()) > 2:
                best_text, best_conf = text.strip(), conf
        return best_text, best_conf
