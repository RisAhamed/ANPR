# from openalpr import Alpr
# import cv2

# class OpenALPRDetector:
#     def __init__(self, country='us', config_path='/etc/openalpr/openalpr.conf', runtime_data='/usr/share/openalpr/runtime_data'):
#         self.alpr = Alpr(country, config_path, runtime_data)
#         if not self.alpr.is_loaded():
#             raise Exception('Error loading OpenALPR')

#     def detect_plates(self, image):
#         # image: np.ndarray (BGR)
#         results = self.alpr.recognize_ndarray(image)
#         boxes = []
#         for plate in results['results']:
#             x1 = plate['coordinates'][0]['x']
#             y1 = plate['coordinates'][0]['y']
#             x2 = plate['coordinates'][2]['x']
#             y2 = plate['coordinates'][2]['y']
#             conf = plate['confidence'] / 100.0
#             boxes.append([x1, y1, x2, y2, conf])
#         return boxes

import os
from openalpr import Alpr
import sys

# --- IMPORTANT ---
# Define the base path to your OpenALPR installation
# This is the folder that contains alpr.exe, openalpr.conf, and the runtime_data folder
OPENALPR_PATH = r'C:\Users\riswa\Downloads\openalpr-2.3.0-win-64bit\openalpr_64'
# -----------------

# (Keep your existing imports and the OPENALPR_PATH variable)
import os
import sys

# --- Your OPENALPR_PATH should be here ---
OPENALPR_PATH = r'C:\Users\riswa\Downloads\openalpr-2.3.0-win-64bit\openalpr_64'
# ------------------------------------------

class OpenALPRDetector:
    def __init__(self, country='us'):
        """
        Initializes the OpenALPR detector.
        """
        # --- START OF THE FIX ---
        # First, add the OpenALPR folder to the DLL search path.
        # This is the most reliable way to ensure the library and its dependencies are found.
        os.add_dll_directory(OPENALPR_PATH)
        # --- END OF THE FIX ---
        
        config_path = os.path.join(OPENALPR_PATH, 'openalpr.conf')
        runtime_data_path = os.path.join(OPENALPR_PATH, 'runtime_data')

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"OpenALPR config file not found at: {config_path}")
        if not os.path.exists(runtime_data_path):
             raise FileNotFoundError(f"OpenALPR runtime_data directory not found at: {runtime_data_path}")

        try:
            self.alpr = Alpr(country, config_path, runtime_data_path)
            if not self.alpr.is_loaded():
                raise Exception("Manually checked and Alpr is not loaded.")
        except Exception as e:
            print(f"Error initializing OpenALPR: {e}")
            print("\nPlease ensure that:")
            print(f"1. The OPENALPR_PATH is correct: '{OPENALPR_PATH}'")
            print("2. You have installed the Visual C++ Redistributables.")
            print("3. The 'openalpr' Python package is installed.")
            sys.exit(1)

    def detect_plates(self, image):
        """
        Detects license plates in a given image (numpy array).
        :param image: np.ndarray (BGR)
        :return: list of [x1, y1, x2, y2, conf, plate_text]
        """
        results = self.alpr.recognize_ndarray(image)
        boxes = []
        for plate in results['results']:
            best_candidate = plate['candidates'][0]
            conf = best_candidate['confidence'] / 100.0
            x_coords = [p['x'] for p in plate['coordinates']]
            y_coords = [p['y'] for p in plate['coordinates']]
            x1, y1 = min(x_coords), min(y_coords)
            x2, y2 = max(x_coords), max(y_coords)
            boxes.append([x1, y1, x2, y2, conf, best_candidate['plate']])
        return boxes

    def get_boxes(self, results):
        """
        Extracts bounding box coordinates and confidence from results.

        :param results: The raw results from OpenALPR.
        :return: A list of [x1, y1, x2, y2, confidence].
        """
        boxes = []
        for plate in results:
            # Get the best candidate for the plate number
            best_candidate = plate['candidates'][0]
            conf = best_candidate['confidence'] / 100.0

            # Get the bounding box coordinates
            x_coords = [p['x'] for p in plate['coordinates']]
            y_coords = [p['y'] for p in plate['coordinates']]
            x1, y1 = min(x_coords), min(y_coords)
            x2, y2 = max(x_coords), max(y_coords)

            boxes.append([x1, y1, x2, y2, conf, best_candidate['plate']])
        return boxes

    def __del__(self):
        """
        Destructor to ensure the Alpr object is properly released.
        """
        if hasattr(self, 'alpr'):
            self.alpr.unload()

### --- Example Usage ---
if __name__ == '__main__':
    # Ensure you have OpenCV installed: pip install opencv-python
    import cv2

    # Path to one of the sample images included with OpenALPR
    sample_image_path = os.path.join(OPENALPR_PATH, 'samples', 'us-1.jpg')

    if not os.path.exists(sample_image_path):
        print(f"Sample image not found at: {sample_image_path}")
    else:
        # Create an instance of the detector
        detector = OpenALPRDetector(country='us')

        # Detect plates from the image file
        plate_results = detector.detect_plates(sample_image_path)
        
        # Get bounding boxes and recognized text
        plate_boxes = detector.get_boxes(plate_results)

        print(f"Found {len(plate_boxes)} license plate(s).")
        
        if plate_boxes:
            # Load the image with OpenCV to draw the boxes
            image = cv2.imread(sample_image_path)

            for i, box in enumerate(plate_boxes):
                x1, y1, x2, y2, conf, plate_text = box
                print(f"  - Plate #{i+1}: {plate_text} (Confidence: {conf:.2f}%)")
                
                # Draw the bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Put the recognized text and confidence above the box
                label = f"{plate_text} ({conf:.2f}%)"
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Display the result
            cv2.imshow('OpenALPR Result', image)
            print("\nPress any key to close the image window.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()