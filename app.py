import streamlit as st
import cv2
import numpy as np
import tempfile
import pandas as pd
import os

from detectors.yolo_detector import YOLODetector
from recognizers.paddleocr_recognizer import PaddleOCRRecognizer
from tracking.deepsort_tracker import DeepSortTracker
from tracking.bytetrack_tracker import ByteTrackTracker

# Model selection UI
st.sidebar.title("Model Selection")

detector_name = st.sidebar.selectbox("Detection Model", ["YOLOv8n finetuned", "YOLOv8m", "YOLOv8l", "OpenALPR"])
ocr_name = st.sidebar.selectbox("OCR Model", ["PaddleOCR"])
tracker_name = st.sidebar.selectbox("Tracker", ["DeepSORT", "ByteTrack"])
# Frame skipping for real-time
frame_skip = st.sidebar.slider("Frame Skip Interval", 1, 10, 3)

# Model loader
if detector_name == "YOLOv8n finetuned":
    from pathlib import Path
    model_path = os.path.join("ANPR", "yolov8n-finetuned", "weights", "best.pt")
    detector = YOLODetector(weights_path=model_path)
elif detector_name == "YOLOv8m":
    detector = YOLODetector(weights_path="yolov8m.pt")
elif detector_name == "YOLOv8l":
    detector = YOLODetector(weights_path="yolov8l.pt")
elif detector_name == "OpenALPR":
    from detectors.openalpr_detector import OpenALPRDetector
    detector = OpenALPRDetector()
else:
    detector = YOLODetector(weights_path="yolov8n.pt")

if ocr_name == "PaddleOCR":
    recognizer = PaddleOCRRecognizer()
else:
    recognizer = PaddleOCRRecognizer()

if tracker_name == "DeepSORT":
    tracker = DeepSortTracker()
else:
    tracker = ByteTrackTracker()

def process_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    boxes = detector.detect_plates(image)
    results = []
    for box in boxes:
        x1, y1, x2, y2, conf = box
        plate_img = image[y1:y2, x1:x2]
        if plate_img.size == 0:
            continue
        plate_img_rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
        text, ocr_conf = recognizer.recognize_plate(plate_img_rgb)
        if text:
            results.append({
                'plate': text,
                'confidence': ocr_conf,
                'box': (x1, y1, x2, y2)
            })
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{text} ({ocr_conf:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return image, results

def process_video(uploaded_file):
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_video.write(uploaded_file.read())
    temp_video.close()
    cap = cv2.VideoCapture(temp_video.name)
    frame_count = 0
    all_results = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_skip != 0:
            frame_count += 1
            continue
        boxes = detector.detect_plates(frame)
        tracked = tracker.track_plates(boxes, frame)
        for t in tracked:
            x1, y1, x2, y2, track_id = t
            plate_img = frame[y1:y2, x1:x2]
            if plate_img.size == 0:
                continue
            plate_img_rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
            text, ocr_conf = recognizer.recognize_plate(plate_img_rgb)
            if text:
                all_results.append({
                    'track_id': track_id,
                    'plate': text,
                    'confidence': ocr_conf,
                    'frame': frame_count
                })
        frame_count += 1
    cap.release()
    os.unlink(temp_video.name)
    # Deduplicate by plate text and track_id
    unique_plates = {}
    for r in all_results:
        key = (r['track_id'], r['plate'])
        if key not in unique_plates or r['confidence'] > unique_plates[key]['confidence']:
            unique_plates[key] = r
    return list(unique_plates.values())

def main():
    st.title('Modular ANPR System')
    st.write('Upload an image or video. Detected number plates will be highlighted and listed.')
    uploaded_file = st.file_uploader('Upload Image or Video', type=['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov'])
    results = []
    if uploaded_file:
        file_type = uploaded_file.type
        if 'image' in file_type:
            image, plates = process_image(uploaded_file)
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption='Annotated Image', use_column_width=True)
            results = plates
        elif 'video' in file_type:
            st.video(uploaded_file)
            results = process_video(uploaded_file)
        else:
            st.error('Unsupported file type.')
        if results:
            st.sidebar.header('Detected Plates')
            for i, p in enumerate(results):
                if 'track_id' in p:
                    st.sidebar.write(f"{i+1}. {p['plate']} (Track: {p['track_id']}, Conf: {p['confidence']:.2f})")
                else:
                    st.sidebar.write(f"{i+1}. {p['plate']} (Conf: {p['confidence']:.2f})")
            df = pd.DataFrame(results)
            csv = df.to_csv(index=False).encode('utf-8')
            st.sidebar.download_button('Download Results as CSV', csv, 'plates.csv', 'text/csv')
        else:
            st.sidebar.write('No plates detected.')

if __name__ == '__main__':
    main()
