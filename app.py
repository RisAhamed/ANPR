import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import tempfile
import pandas as pd
import os

# Load YOLO model and EasyOCR reader once
@st.cache_resource
def load_models():
    # model = YOLO(os.path.join(os.path.dirname(__file__), '../yolov8n-finetuned/yolov8n-finetuned/weights/best.pt'))
    model = YOLO('yolo11n.pt')
    reader = easyocr.Reader(['en'], gpu=False)
    return model, reader

model, reader = load_models()

# def detect_and_read(image, model, reader, ocr_min_conf=0.4):
#     results = model(image)
#     detections = results[0].boxes
#     plates = []
#     h, w, _ = image.shape

#     for box in detections:
#         x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
#         x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

#         plate_img = image[y1:y2, x1:x2]
#         if plate_img.size == 0:
#             continue

#         # Preprocessing variations for OCR
#         processed_imgs = []
#         try:
#             gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
#             processed_imgs.extend([
#                 gray,
#                 cv2.equalizeHist(gray),
#                 cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                       cv2.THRESH_BINARY, 11, 2)
#             ])
#         except:
#             processed_imgs.append(plate_img)

#         # OCR on processed images, keep best result
#         best_text, best_conf = '', 0
#         for img in processed_imgs:
#             try:
#                 ocr_results = reader.readtext(img)
#                 for detection in ocr_results:
#                     text, conf = detection[1], detection[2]
#                     if conf > best_conf and len(text.strip()) > 2 and conf >= ocr_min_conf:
#                         best_text, best_conf = text.strip(), conf
#             except:
#                 continue

#         if best_text:
#             plates.append({
#                 'plate': best_text,
#                 'confidence': best_conf,
#                 'box': (x1, y1, x2, y2)
#             })
#             # Optional: Draw bounding box on image
#             cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(image, f"{best_text} ({best_conf:.2f})", (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#     return plates, image

def detect_and_read(image, model, reader, ocr_min_conf=0.3):
    results = model(image)
    detections = results[0].boxes
    plates = []
    h, w, _ = image.shape

    for box in detections:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

        plate_img = image[y1:y2, x1:x2]
        if plate_img.size == 0:
            continue

        # Aggressive Preprocessing Variations
        processed_imgs = []
        try:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            eq = cv2.equalizeHist(gray)
            blur = cv2.GaussianBlur(eq, (3, 3), 0)
            thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 15, 9)
            invert = cv2.bitwise_not(thresh)

            processed_imgs.extend([gray, eq, blur, thresh, invert])
        except:
            processed_imgs.append(plate_img)

        # OCR on all versions - keep best result
        best_text, best_conf = '', 0
        for img in processed_imgs:
            try:
                ocr_results = reader.readtext(img)
                for detection in ocr_results:
                    text, conf = detection[1], detection[2]
                    if conf > best_conf and len(text.strip()) > 2 and conf >= ocr_min_conf:
                        best_text, best_conf = text.strip(), conf
            except:
                continue

        if best_text:
            plates.append({
                'plate': best_text,
                'confidence': best_conf,
                'box': (x1, y1, x2, y2)
            })
            # Optional Visualization
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{best_text} ({best_conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return plates, image


def process_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    plates, annotated_image = detect_and_read(image, model, reader)
    return annotated_image, plates

# def process_video(uploaded_file):
#     temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
#     temp_video.write(uploaded_file.read())
#     temp_video.close()

#     cap = cv2.VideoCapture(temp_video.name)
#     all_plates = []
#     frame_count = 0

#     while True:
#         ret, frame = cap.read()
#         if not ret or frame_count > 100:  # Limit to first 100 frames
#             break

#         plates, _ = detect_and_read(frame, model, reader)
#         for p in plates:
#             p['frame'] = frame_count
#             all_plates.append(p)

#         frame_count += 1

#     cap.release()
#     os.unlink(temp_video.name)

#     # Deduplicate by plate text, keep highest confidence detection
#     unique_plates = {}
#     for p in all_plates:
#         if p['plate'] not in unique_plates or p['confidence'] > unique_plates[p['plate']]['confidence']:
#             unique_plates[p['plate']] = p

#     return all_plates, list(unique_plates.values())

from sort.sort import Sort
from collections import defaultdict, Counter

def process_video(uploaded_file):
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_video.write(uploaded_file.read())
    temp_video.close()

    cap = cv2.VideoCapture(temp_video.name)
    tracker = Sort()
    track_ocr_results = defaultdict(list)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame_count > 100:  # Limit for performance
            break

        results = model(frame)[0]
        h, w, _ = frame.shape
        detections = []

        for box in results.boxes:
            x1, y1, x2, y2, conf = box.xyxy[0].cpu().numpy().tolist() + [box.conf[0].cpu().numpy()]
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            detections.append([x1, y1, x2, y2, conf])

        # Ensure detections_np is always (N, 5) shape, even if N=0
        if len(detections) == 0:
            detections_np = np.empty((0, 5))
        else:
            detections_np = np.array(detections)
        tracks = tracker.update(detections_np)

        for track in tracks:
            x1, y1, x2, y2, track_id = map(int, track)
            plate_img = frame[y1:y2, x1:x2]

            if plate_img.size == 0:
                continue

            try:
                gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                ocr_results = reader.readtext(gray)
                for detection in ocr_results:
                    text, conf = detection[1], detection[2]
                    if conf > 0.4 and len(text.strip()) > 2:
                        track_ocr_results[track_id].append(text.strip())
            except:
                continue

        frame_count += 1

    cap.release()
    os.unlink(temp_video.name)

    # Final plate determination per tracked object
    final_plates = []
    for track_id, texts in track_ocr_results.items():
        if texts:
            most_common_text = Counter(texts).most_common(1)[0][0]
            final_plates.append({"track_id": int(track_id), "plate": most_common_text, "detections": len(texts)})

    return final_plates

def main():
    st.title('ANPR')
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
                if 'detections' in p:
                    st.sidebar.write(f"{i+1}. {p['plate']} (Detections: {p['detections']})")
                else:
                    st.sidebar.write(f"{i+1}. {p['plate']} (Conf: {p['confidence']:.2f})")
            df = pd.DataFrame(results)
            # Choose columns based on result type
            if 'detections' in df.columns:
                csv = df[['plate', 'detections']].to_csv(index=False).encode('utf-8')
            else:
                csv = df[['plate', 'confidence']].to_csv(index=False).encode('utf-8')
            st.sidebar.download_button('Download Results as CSV', csv, 'plates.csv', 'text/csv')
        else:
            st.sidebar.write('No plates detected.')

if __name__ == '__main__':
    main()
