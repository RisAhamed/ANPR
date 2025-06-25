from hyperlpr import HyperLPR_plate_recognition
import cv2

class HyperLPRRecognizer:
    def __init__(self):
        pass  # No model loading needed for hyperlpr-pipeline

    def recognize_plate(self, image):
        # image: np.ndarray (BGR or RGB)
        # HyperLPR expects BGR
        if len(image.shape) == 3 and image.shape[2] == 3:
            img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = image
        results = HyperLPR_plate_recognition(img_bgr)
        best_text, best_conf = '', 0
        for res in results:
            text, conf = res[0], res[1]
            if conf > best_conf and len(text.strip()) > 2:
                best_text, best_conf = text.strip(), conf
        return best_text, best_conf
