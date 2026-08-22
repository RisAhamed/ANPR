"""
ANPR Streamlit App — production-grade
- Central config via anpr_config.py / .env
- Health checks, logging, graceful fallbacks
- Supports image + video with tracking + selective OCR
"""
import logging
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

import anpr_config as cfg
from detectors.yolo_detector import YOLODetector
from recognizers.easyocr_recognizer import EasyOCRRecognizer
from tracking.deepsort_tracker import DeepSortTracker
from utils import clean_plate_text, deskew_and_clean_plate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Page config ───────────────────────────────────────
st.set_page_config(
    page_title="ANPR System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Cached model loaders ──────────────────────────────
@st.cache_resource(show_spinner="Loading detection model…")
def load_detector(model_path: Path):
    return YOLODetector(weights_path=model_path)

@st.cache_resource(show_spinner="Loading OCR engine…")
def load_recognizer():
    return EasyOCRRecognizer()

@st.cache_resource(show_spinner="Loading tracker…")
def load_tracker():
    return DeepSortTracker()

# ── Resolve model path ────────────────────────────────
# Prefer plate model for single-stage demo; fallback to vehicle model, then base yolo
candidate_paths = [
    cfg.PLATE_MODEL_PATH,
    Path("models/number_plated/number_plates_model.pt"),
    Path("models/vehicle_detection_model/best.pt"),
    Path("yolov8n.pt"),
    Path("yolo11n.pt"),
]
model_path = next((p for p in candidate_paths if Path(p).exists()), None)

# ── Sidebar ───────────────────────────────────────────
st.sidebar.title("⚙️ Settings")
st.sidebar.caption(f"OCR: **EasyOCR** | Tracker: **DeepSORT/SORT**")
conf_threshold = st.sidebar.slider("Detection confidence", 0.0, 1.0, float(cfg.PLATE_CONF), 0.05)
frame_skip = st.sidebar.slider("Frame interval (video)", 1, 10, 2)
st.sidebar.divider()
st.sidebar.caption(f"Model: `{model_path}`" if model_path else "⚠️ No model found")
if model_path:
    st.sidebar.caption(f"Device: `{cfg.DEVICE}`")

# ── Validate model ────────────────────────────────────
if model_path is None:
    st.error(
        "No YOLO weights found. Expected one of:\n"
        + "\n".join(f"- `{p}`" for p in candidate_paths)
        + "\n\nAdd model weights to `models/` or set `ANPR_PLATE_MODEL` in `.env`."
    )
    st.stop()

try:
    detector = load_detector(model_path)
    recognizer = load_recognizer()
    tracker = load_tracker()
except Exception as e:
    logger.exception("Model load failed")
    st.error(f"Failed to load models: {e}")
    st.stop()

# ── Helpers ───────────────────────────────────────────
def process_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        st.error("Could not decode image.")
        return None, []

    boxes = detector.detect_plates(image, conf_threshold=conf_threshold)
    st.write(f"Detected **{len(boxes)}** potential plates.")

    results = []
    if boxes:
        cols = st.columns(min(len(boxes), 4))

    for i, box in enumerate(boxes):
        x1, y1, x2, y2, conf = box
        h, w = image.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        plate_img = image[y1:y2, x1:x2]
        if plate_img.size == 0:
            st.write(f"Box {i} crop empty — skipping.")
            continue

        if boxes and i < len(cols):
            with cols[i]:
                st.image(cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB), caption=f"Crop {i} (original)", use_container_width=True)

        preprocessed_mono = deskew_and_clean_plate(plate_img)
        if boxes and i < len(cols):
            with cols[i]:
                st.image(preprocessed_mono, caption=f"Crop {i} (preprocessed)", use_container_width=True)

        preprocessed_bgr = cv2.cvtColor(preprocessed_mono, cv2.COLOR_GRAY2BGR)
        raw_text, ocr_conf = recognizer.recognize_plate(preprocessed_bgr)
        plate_text = clean_plate_text(raw_text)

        st.write(f"Result {i}: **{plate_text or '—'}** (raw: `{raw_text}` conf: {ocr_conf:.2f})")

        if plate_text:
            results.append({"plate": plate_text, "confidence": round(float(ocr_conf), 3), "box": (x1, y1, x2, y2)})
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, plate_text, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return image, results


def process_video(uploaded_file):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(uploaded_file.read())
    tmp.close()

    cap = cv2.VideoCapture(tmp.name)
    if not cap.isOpened():
        st.error("Could not open video.")
        return []

    track_history: dict = defaultdict(lambda: {"plate": None, "confidence": 0.0})
    st_frame = st.empty()
    st_results_container = st.sidebar.empty()
    frame_count = 0
    results_list = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip != 0:
                frame_count += 1
                continue

            boxes = detector.detect_plates(frame, conf_threshold=conf_threshold)
            tracked = tracker.track_plates(boxes, frame)

            for x1, y1, x2, y2, track_id in tracked:
                track_id = int(track_id)
                # skip OCR if already high-confidence
                if track_history[track_id]["confidence"] > 0.85:
                    plate_text = track_history[track_id]["plate"]
                else:
                    plate_img = frame[y1:y2, x1:x2]
                    if plate_img.size == 0:
                        plate_text = track_history[track_id]["plate"]
                    else:
                        mono = deskew_and_clean_plate(plate_img)
                        bgr = cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)
                        raw_text, ocr_conf = recognizer.recognize_plate(bgr)
                        plate_text = clean_plate_text(raw_text)
                        if plate_text and ocr_conf > track_history[track_id]["confidence"]:
                            track_history[track_id] = {"plate": plate_text, "confidence": float(ocr_conf)}
                        else:
                            plate_text = track_history[track_id]["plate"] or plate_text

                if plate_text:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID {track_id}: {plate_text}", (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            st_frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"Frame {frame_count}", use_container_width=True)

            current = [{"track_id": tid, "plate": d["plate"], "confidence": round(d["confidence"], 3)}
                       for tid, d in track_history.items() if d["plate"]]
            if current:
                df = pd.DataFrame(current).sort_values("track_id")
                st_results_container.dataframe(df, use_container_width=True)
                results_list = df.to_dict("records")

            frame_count += 1
    finally:
        cap.release()
        try:
            os.unlink(tmp.name)
        except Exception:
            pass

    st.success(f"Done — {frame_count} frames, {len(results_list)} unique plates.")
    return results_list


# ── Main ──────────────────────────────────────────────
def main():
    st.title("🚗 ANPR — Automatic Number Plate Recognition")
    st.caption("Detect → Track → OCR  •  YOLO + EasyOCR + DeepSORT/SORT  •  Production build")

    with st.expander("ℹ️ How it works", expanded=False):
        st.markdown("""
        1. **Detection** — YOLO (OBB-capable) finds vehicles/plates
        2. **Tracking** — DeepSORT (fallback: SORT/IOU) keeps IDs across frames
        3. **OCR** — Deskew + CLAHE + Otsu + EasyOCR (allowlist A-Z0-9)
        4. Adjust **confidence** and **frame interval** in the sidebar.
        """)

    uploaded = st.file_uploader("Upload image or video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])
    if not uploaded:
        st.info("👆 Upload a file to start. Try `test_video.mp4` or any plate image.")
        return

    ctype = uploaded.type or ""
    if "image" in ctype:
        image, plates = process_image(uploaded)
        if image is not None:
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Annotated", use_container_width=True)
        results = plates
    elif "video" in ctype:
        with st.spinner("Processing video…"):
            results = process_video(uploaded)
    else:
        st.error("Unsupported file type.")
        results = []

    if results:
        st.sidebar.divider()
        st.sidebar.header("📋 Final results")
        df_final = pd.DataFrame(results)
        st.sidebar.dataframe(df_final, use_container_width=True)
        csv = df_final.to_csv(index=False).encode("utf-8")
        st.sidebar.download_button("⬇️ Download CSV", csv, "plates.csv", "text/csv")
    else:
        if uploaded:
            st.sidebar.write("No confident plates found — try lowering the threshold.")

if __name__ == "__main__":
    main()
