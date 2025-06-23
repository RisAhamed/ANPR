import sys
import glob
import os
import numpy as np
import cv2
from PIL import Image
import pytesseract
import re

class NumberPlateDetector:
    def __init__(self, ):
        pass
        
        
    def preprocess_image(self, img):
        """Advanced image preprocessing"""
        # Noise reduction
        img = cv2.GaussianBlur(img, (5,5), 0)
        
        # Grayscale conversion
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Multiple preprocessing techniques
        preprocessed_images = [
            ('Original Gray', gray),
            ('Sobel Edge', cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)),
            ('Adaptive Threshold', cv2.adaptiveThreshold(
                gray, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 
                11, 2
            )),
            ('Equalized Histogram', cv2.equalizeHist(gray))
        ]
        
        return preprocessed_images

    def detect_plate(self, img):
        """Comprehensive plate detection"""
        preprocessed_images = self.preprocess_image(img)
        
        detected_plates = []
        
        for method_name, processed_img in preprocessed_images:
            # Morphological operations
            element = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(17, 3))
            morph_img = cv2.morphologyEx(processed_img, cv2.MORPH_CLOSE, element)
            
            # Find contours
            contours, _ = cv2.findContours(
                morph_img, 
                mode=cv2.RETR_EXTERNAL, 
                method=cv2.CHAIN_APPROX_SIMPLE
            )
            
            for cnt in contours:
                # Minimum area rectangle
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                
                # Check plate characteristics
                if self.validate_plate_region(rect, img):
                    x, y, w, h = cv2.boundingRect(cnt)
                    plate_img = img[y:y+h, x:x+w]
                    
                    # OCR
                    ocr_text = self.perform_ocr(plate_img)
                    
                    if ocr_text and len(ocr_text) > 2:  # Basic validation
                        detected_plates.append({
                            'text': ocr_text.strip(),
                            'method': method_name,
                            'region': (x, y, w, h)
                        })
        
        return detected_plates

    def validate_plate_region(self, rect, original_img):
        """Advanced plate region validation"""
        (x, y), (width, height), angle = rect
        
        # Aspect ratio check
        if width > height:
            ratio = width / height
        else:
            ratio = height / width
        
        # Area and ratio constraints
        area = width * height
        
        conditions = [
            1000 < area < 20000,  # Reasonable plate area
            3 < ratio < 6,         # Typical plate aspect ratio
            abs(angle) < 15        # Minimal rotation
        ]
        
        return all(conditions)

    def perform_ocr(self, plate_img):
        """Enhanced OCR with multiple configurations"""
        ocr_configs = [
            {'lang': 'eng', 'config': '--psm 7'},  # Treat image as single text line
            {'lang': 'eng', 'config': '--psm 6'},  # Assume single uniform block of text
        ]
        
        for config in ocr_configs:
            try:
                text = pytesseract.image_to_string(
                    plate_img, 
                    lang=config['lang'], 
                    config=config['config']
                ).strip()
                
                # Basic text validation
                if self.validate_plate_text(text):
                    return text
            except Exception as e:
                print(f"OCR Error: {e}")
        
        return None

    def validate_plate_text(self, text):
        """Plate number validation"""
        if not text:
            return False
        
        # Remove non-alphanumeric characters
        text = re.sub(r'[^A-Za-z0-9]', '', text)
        
        # Basic Indian license plate regex
        plate_pattern = r'^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$'
        
        return bool(re.match(plate_pattern, text))

# Usage
def main():
    detector = NumberPlateDetector()
    
    # Process single image
    img = cv2.imread('yolov8n-finetuned\yolov8n-finetuned\frame_0015_car_0.88.jpg')
    
    plates = detector.detect_plate(img)
    
    for plate in plates:
        print(f"Detected Plate: {plate['text']} (Method: {plate['method']})")
        
        # Optional: Draw rectangle on detected plate
        x, y, w, h = plate['region']
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imshow('Detected Plates', img)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()
