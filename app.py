import streamlit as st
import cv2
import numpy as np
import tempfile
import pandas as pd
import os
from collections import defaultdict

# Import your project modules
from detectors.yolo_detector import YOLODetector
from recognizers.easyocr_recognizer import EasyOCRRecognizer
from tracking.deepsort_tracker import DeepSortTracker
from utils import deskew_and_clean_plate, clean_plate_text  # Our new utility functions

# --- Model Loading (Cached for Performance) ---
@st.cache_resource
def load_detector(model_path):
    return YOLODetector(weights_path=model_path)

@st.cache_resource
def load_recognizer():
    # We will focus on EasyOCR as it's robust and simpler to manage
    return EasyOCRRecognizer()

@st.cache_resource
def load_tracker():
    return DeepSortTracker()

# --- Main App UI ---
st.sidebar.title("Model and Parameters")
# For simplicity, we will fix the models to the best-performing ones first.
# You can re-add the selectbox later if needed.
detector_name = "YOLOv8n finetuned"
ocr_name = "EasyOCR"
tracker_name = "DeepSORT"
st.sidebar.info(f"Detector: **{detector_name}**\n\nOCR: **{ocr_name}**\n\nTracker: **{tracker_name}**")

# Confidence Threshold
conf_threshold = st.sidebar.slider("Detection Confidence Threshold", 0.0, 1.0, 0.4, 0.05)
frame_skip = st.sidebar.slider("Frame Processing Interval", 1, 10, 2)

# --- Load Models ---
# Correct path handling
model_path = os.path.join("yolov8n-finetuned", "weights", "best.pt")
if not os.path.exists(model_path):
    st.error(f"FATAL: The fine-tuned model 'best.pt' was not found at {model_path}. Please ensure the model file is in the correct directory.")
    st.stop()
    
detector = load_detector(model_path)
recognizer = load_recognizer()
tracker = load_tracker()

# --- Image Processing Function (with Accuracy Fixes) ---
def process_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Use the confidence threshold from the UI
    boxes = detector.detect_plates(image, conf_threshold=conf_threshold)
    st.write(f"Detected {len(boxes)} potential plates.")
    
    results = []
    
    # Create columns for displaying crops
    if boxes:
        cols = st.columns(len(boxes))
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2, conf = box
        plate_img = image[y1:y2, x1:x2]
        
        if plate_img.size == 0:
            st.write(f"Box {i} crop is empty, skipping.")
            continue

        # Show original crop
        with cols[i]:
            st.image(cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB), caption=f"Crop {i} (Original)", use_container_width=True)

        # De-skew and clean the plate for OCR (THE ACCURACY FIX)
        # ... inside the `for i, box in enumerate(boxes):` loop ...

# De-skew and clean the plate for OCR (THE ACCURACY FIX)
            preprocessed_plate_mono = deskew_and_clean_plate(plate_img) # mono-channel output
            st.image(preprocessed_plate_mono, caption=f"Crop {i} (Final for OCR)", use_container_width=True)

            # Convert back to 3-channel for EasyOCR
            preprocessed_plate_bgr = cv2.cvtColor(preprocessed_plate_mono, cv2.COLOR_GRAY2BGR)

            # Perform OCR
            raw_text, ocr_conf = recognizer.recognize_plate(preprocessed_plate_bgr) # Pass BGR image

        
        
        # Clean the text (Post-processing)
        plate_text = clean_plate_text(raw_text)

        st.write(f"Result for Crop {i}: **{plate_text}** (Raw: '{raw_text}', Conf: {ocr_conf:.2f})")

        if plate_text:
            results.append({
                'plate': plate_text,
                'confidence': ocr_conf,
                'box': (x1, y1, x2, y2)
            })
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(image, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
    return image, results

# --- Video Processing Function (with Performance Fixes) ---
def process_video(uploaded_file):
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_video.write(uploaded_file.read())
    temp_video.close()
    
    cap = cv2.VideoCapture(temp_video.name)
    frame_count = 0
    
    # This dictionary will store the OCR result for each tracked ID
    # to avoid re-processing. Structure: {track_id: {'plate': '...', 'confidence': 0.9}}
    track_history = defaultdict(lambda: {'plate': None, 'confidence': 0.0})
    
    # Setup Streamlit placeholders
    st_frame = st.empty()
    st_results_header = st.sidebar.header('Detected Plates')
    st_results_container = st.sidebar.empty()
    
    results_list = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process only every Nth frame
        if frame_count % frame_skip != 0:
            frame_count += 1
            continue

        # 1. Detection
        boxes = detector.detect_plates(frame, conf_threshold=conf_threshold)
        
        # 2. Tracking
        # Note: Ensure your tracker returns a list of [x1, y1, x2, y2, track_id]
        tracked_objects = tracker.track_plates(boxes, frame)
        
        # 3. Selective OCR (THE PERFORMANCE FIX)
        for t in tracked_objects:
            x1, y1, x2, y2, track_id = map(int, t)
            
            # If we have already processed this track_id with high confidence, skip
            if track_history[track_id]['confidence'] > 0.85: # High confidence threshold
                plate_text = track_history[track_id]['plate']
            else:
                # This is a new track or a low-confidence one, so we run OCR
                plate_img = frame[y1:y2, x1:x2]


            if plate_img.size > 0:
                # Pre-process for accuracy
                preprocessed_plate_mono = deskew_and_clean_plate(plate_img)
                preprocessed_plate_bgr = cv2.cvtColor(preprocessed_plate_mono, cv2.COLOR_GRAY2BGR)
                
                raw_text, ocr_conf = recognizer.recognize_plate(preprocessed_plate_bgr)
                plate_text = clean_plate_text(raw_text)

                # If the new OCR is better, update the history
                if plate_text and ocr_conf > track_history[track_id]['confidence']:
                    track_history[track_id] = {'plate': plate_text, 'confidence': ocr_conf}

            else:
                    plate_text = track_history[track_id]['plate'] # Use old text if crop fails

            # Draw bounding box and text on the frame
            if plate_text:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {track_id}: {plate_text}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Display the processed frame
        st_frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f'Processing Frame {frame_count}', use_container_width=True)
        
        # Update the results in the sidebar
        current_results = [
            {'track_id': tid, 'plate': data['plate'], 'confidence': data['confidence']} 
            for tid, data in track_history.items() if data['plate']
        ]
        
        # Create a DataFrame for cleaner display
        if current_results:
            df = pd.DataFrame(current_results).sort_values(by='track_id')
            st_results_container.dataframe(df)
            results_list = df.to_dict('records')

        frame_count += 1
        
    cap.release()
    os.unlink(temp_video.name)
    
    st.success("Video processing complete.")
    return results_list


def main():
    st.title('ANPR System - Real-World Edition')
    st.write('Upload an image or video. The system will detect, track, and read license plates.')
    
    uploaded_file = st.file_uploader('Upload Image or Video', type=['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov'])
    
    if uploaded_file:
        file_type = uploaded_file.type
        
        if 'image' in file_type:
            image, plates = process_image(uploaded_file)
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption='Annotated Image', use_container_width=True)
            results = plates
        elif 'video' in file_type:
            with st.spinner('Processing video... this may take a moment.'):
                results = process_video(uploaded_file)
        else:
            st.error('Unsupported file type.')
            results = []

        if results:
            st.sidebar.header('Final Results')
            df_final = pd.DataFrame(results)
            st.sidebar.dataframe(df_final)
            csv = df_final.to_csv(index=False).encode('utf-8')
            st.sidebar.download_button('Download Results as CSV', csv, 'final_plates.csv', 'text/csv')
        else:
            st.sidebar.write('No plates with sufficient confidence were found.')

if __name__ == '__main__':
    main()