from paddleocr import PaddleOCR
import numpy as np

class PaddleOCRRecognizer:
    def __init__(self):
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')

    def recognize_plate(self, image):
        # Perform OCR prediction
        result = self.ocr.predict(image)
        
        # Check if result is empty or lacks the expected structure
        if not result or not result[0]:
            return '', 0  # Return empty text and zero confidence if no result
        
        # Initialize variables to store the best result
        best_text, best_conf = '', 0
        
        # Process each line in the OCR result
        for line in result[0]:
            # Ensure line has enough elements and line[1] is a list/tuple with at least 2 elements
            if len(line) > 1 and len(line[1]) >= 2:
                text, conf = line[1][0], line[1][1]
                # Update best result if confidence is higher and text is meaningful
                if conf > best_conf and len(text.strip()) > 2:
                    best_text, best_conf = text.strip(), conf
            else:
                # Optionally log a warning for debugging
                print(f"Warning: Unexpected result structure in OCR output: {line}")
        
        # Return the best recognized text and its confidence score
        return best_text, best_conf
