import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import re
from sort.sort import Sort
import os
import csv

# --- 1. CONFIGURATION ---
VEHICLE_MODEL_PATH = 'models/vehicle_detection_model/best.pt'
PLATE_MODEL_PATH = 'models/number_plated/number_plates_model.pt'
VIDEO_SOURCE = 'test_video.mp4'

OUTPUT_DIR         = 'final_ocr'

# --- Confidence Thresholds ---
VEHICLE_CONFIDENCE_THRESHOLD = 0.2
PLATE_CONFIDENCE_THRESHOLD   = 0.4
OCR_CONFIDENCE_THRESHOLD     = 0.1  # Lowered for better capture

# --- Vehicle Class List ---
VEHICLE_CLASS_LIST = [
    '2-axle-trailer', '2-axle-truck', '3-axle-trailer', '3-axle-truck',
    '4-axle-trailer', '4-axle-truck', '5-axle-truck', '5+-axle-truck-trailer',
    'Ambulance', 'Auto', 'Autorickshaw', 'Bicycle', 'Bus-2-axle', 'Bus-3-axle',
    'Car', 'Firetruck', 'handcart', 'HCM/EME', 'LCV', 'Minivan',
    'Motorcycle', 'Tractor'
]

# --- Initialize EasyOCR Reader ---
try:
    reader = easyocr.Reader(['en'], gpu=True)
    print("EasyOCR is running on GPU.")
except Exception:
    reader = easyocr.Reader(['en'], gpu=False)
    print("EasyOCR is running on CPU.")

# --- Helper Functions ---
def preprocess_plate_image(plate_crop):
    # Convert to grayscale
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    smoothed = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(smoothed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    # Optional: Upscale for better OCR
    scale_factor = 2
    resized = cv2.resize(thresh, None, fx=scale_factor, fy=scale_factor, 
                         interpolation=cv2.INTER_CUBIC)
    
    return resized

def correct_rotation(image):
    # Detect edges
    edges = cv2.Canny(image, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    
    if lines is not None:
        angle = np.median([line[0][1] * 180 / np.pi for line in lines]) - 90
        (h, w) = image.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
    else:
        rotated = image
    return rotated

def clean_and_validate_plate(text):
    cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
    if len(cleaned) >= 4:  # Relaxed rule to capture more results
        return cleaned
    return None

def correct_errors(text):
    corrections = {'0': 'O', '1': 'I', '8': 'B', '5': 'S', '2': 'Z'}
    corrected = ''.join(corrections.get(c, c) if c in '01582' else c for c in text)
    return corrected

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Loading YOLO models...")
    vehicle_model = YOLO(VEHICLE_MODEL_PATH)
    plate_model   = YOLO(PLATE_MODEL_PATH)
    print("Models loaded successfully.")

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {VIDEO_SOURCE}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    out_video = cv2.VideoWriter(
        os.path.join(OUTPUT_DIR, 'annotated_video.mp4'),
        cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height)
    )

    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    all_detection_events = []
    latest_results        = {}
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # --- Vehicle Detection (OBB) ---
        v_res = vehicle_model(frame, verbose=False)[0]
        dets = []
        if v_res.obb is not None:
            for obb in v_res.obb.data.tolist():
                x_c, y_c, w, h, angle, conf, cls_id = obb
                if conf < VEHICLE_CONFIDENCE_THRESHOLD:
                    continue
                rect = ((x_c, y_c), (w, h), angle * 180 / np.pi)
                pts = cv2.boxPoints(rect)
                x1, y1 = pts.min(axis=0).astype(int)
                x2, y2 = pts.max(axis=0).astype(int)
                dets.append([x1, y1, x2, y2, float(conf), int(cls_id)])

        # --- Vehicle Tracking ---
        tracked = tracker.update(np.array(dets)[:, :5]) if dets else np.empty((0,5))

        # --- Process Each Tracked Vehicle ---
        for tr in tracked:
            x1, y1, x2, y2, tid = map(int, tr)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            vehicle_crop = frame[y1:y2, x1:x2]
            if vehicle_crop.size == 0 or vehicle_crop.shape[0] == 0 or vehicle_crop.shape[1] == 0:
                continue

            # Vehicle-specific folder
            veh_folder = os.path.join(OUTPUT_DIR, f"vehicle_{tid}")
            plate_folder = os.path.join(veh_folder, 'plates')
            os.makedirs(plate_folder, exist_ok=True)

            # Save vehicle crop
            base_name = f"v{tid}_f{frame_idx}"
            veh_path  = os.path.join(veh_folder, f"{base_name}_vehicle.jpg")
            cv2.imwrite(veh_path, vehicle_crop)

            # Determine class
            cx, cy = (x1+x2)//2, (y1+y2)//2
            v_class = 'Unknown'
            for d in dets:
                if d[0] < cx < d[2] and d[1] < cy < d[3]:
                    v_class = VEHICLE_CLASS_LIST[d[5]]
                    break

            # --- Plate Detection & OCR ---
            plate_text = "Not Detected"
            plate_path = "N/A"
            p_res = plate_model(vehicle_crop, verbose=False)[0]
            if hasattr(p_res, 'boxes') and p_res.boxes is not None:
                for pb, pconf in zip(p_res.boxes.xyxy, p_res.boxes.conf):
                    if pconf < PLATE_CONFIDENCE_THRESHOLD:
                        continue
                    px1, py1, px2, py2 = map(int, pb.tolist())
                    if px2 <= px1 or py2 <= py1:
                        continue
                    plate_crop = vehicle_crop[py1:py2, px1:px2]
                    if plate_crop.size == 0 or plate_crop.shape[0] == 0 or plate_crop.shape[1] == 0:
                        continue

                    # Save plate crop for debugging
                    debug_plate_path = os.path.join(plate_folder, f"{base_name}_plate_debug.jpg")
                    cv2.imwrite(debug_plate_path, plate_crop)

                    # Map plate coordinates to original frame
                    abs_px1 = x1 + px1
                    abs_py1 = y1 + py1
                    abs_px2 = x1 + px2
                    abs_py2 = y1 + py2

                    # Draw plate bounding box (blue)
                    cv2.rectangle(frame, (abs_px1, abs_py1), (abs_px2, abs_py2), (255, 0, 0), 2)

                    # Preprocess and rotate
                    preprocessed = preprocess_plate_image(plate_crop)
                    rotated = correct_rotation(preprocessed)

                    # Perform OCR with restricted character set
                    ocr_res = reader.readtext(
                        rotated,
                        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                        text_threshold=OCR_CONFIDENCE_THRESHOLD
                    )
                    print(f"Frame {frame_idx}, ID {tid}: Raw OCR results: {ocr_res}")

                    for _, text, conf in ocr_res:
                        if conf < OCR_CONFIDENCE_THRESHOLD:
                            continue
                        cleaned = clean_and_validate_plate(text)
                        if cleaned:
                            corrected = correct_errors(cleaned)
                            plate_text = corrected
                            plate_path = os.path.join(plate_folder, f"{base_name}_plate.jpg")
                            cv2.imwrite(plate_path, plate_crop)
                            # Draw plate text above plate box
                            cv2.putText(frame, plate_text, (abs_px1, abs_py1 - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                            print(f"Frame {frame_idx}, ID {tid}: Detected plate {plate_text}")
                            break
                    if plate_text != "Not Detected":
                        break

            # --- Log Event ---
            all_detection_events.append({
                'frame_index':   frame_idx,
                'track_id':      tid,
                'vehicle_class': v_class,
                'license_plate': plate_text,
                'vehicle_image': veh_path,
                'plate_image':   plate_path
            })
            latest_results[tid] = (v_class, plate_text)

        # --- Annotate and Write Frame ---
        for tr in tracked:
            x1, y1, x2, y2, tid = map(int, tr)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)
            label_color = (0, 0, 255)  # Red by default
            label = f"ID:{tid}"
            if tid in latest_results and latest_results[tid][1] != "Not Detected":
                label_color = (0, 255, 0)  # Green if plate detected
                vc, lp = latest_results[tid]
                label = f"{vc} ID:{tid} | {lp}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), label_color, 2)
            tw, th = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1-th-6), (x1+tw, y1), label_color, -1)
            cv2.putText(frame, label, (x1, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        out_video.write(frame)

    cap.release()
    out_video.release()

    # --- Save CSV ---
    csv_path = os.path.join(OUTPUT_DIR, 'summary_all_events.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'frame_index','track_id','vehicle_class','license_plate','vehicle_image','plate_image'
        ])
        writer.writeheader()
        writer.writerows(all_detection_events)

    print(f"Finished. Frames: {frame_idx}, Events: {len(all_detection_events)}")
    print(f"Results in '{OUTPUT_DIR}'")