# ANPR — Automatic Number Plate Recognition (Production-Grade)

[![Python 3.9+](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue)](https://www.python.org)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)](https://streamlit.io)
[![YOLOv8/v11](https://img.shields.io/badge/detection-YOLOv8%20%2F%20YOLO11-00D1B2)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-ready-blue)](Dockerfile)

Production-grade ANPR system: **detect → track → OCR** Indian and generic license plates from **images & videos**.  
Live model weights ship in the repo via **Git LFS** — clone and run, no separate download step.

> **Stack:** YOLO (OBB-aware) · EasyOCR / Keras-OCR / Nanonets-OCR-s · DeepSORT (fallback: SORT/IOU) · Streamlit · OpenCV · Docker

---

## Table of contents
- [Features](#features)
- [Architecture](#architecture)
- [Repo structure](#repo-structure)
- [Quick start](#quick-start)
- [Docker (recommended for prod)](#docker-recommended-for-prod)
- [Configuration (.env)](#configuration-env)
- [Models & Git LFS](#models--git-lfs)
- [Usage](#usage)
- [Training & dataset](#training--dataset)
- [API & scripts](#api--scripts)
- [Production checklist](#production-checklist)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Unified pipeline** — single YOLO → deskew/CLAHE/Otsu → OCR chain for both images and video.
- **Video tracking** — DeepSORT (mobilenet embedder, Kalman) with pure-IOU SORT fallback so the app never crashes if `deep-sort-realtime` is missing.
- **Accuracy fixes** — `utils.deskew_and_clean_plate()` does perspective warp → bilateral filter → CLAHE → Otsu → morphology → resize + inversion, matching the research path: `Crop → Denoise → Grayscale → Enhance → Threshold → Morphology → Resize → OCR`.
- **Multiple OCR backends** — `easy_ocr.py` (EasyOCR), `final_ocr_keras.py` (Keras-OCR), `Nanonets.py` (Nanonets-OCR-s transformer). App defaults to EasyOCR for robustness.
- **Streamlit UI** — upload image/video, tune confidence & frame-skip, see crops + annotated output, download CSV.
- **Production ready** — Dockerfile + healthcheck, `docker-compose.yml`, `.env` for secrets, pinned `requirements.txt`, `pyproject.toml`, `.gitattributes` LFS, structured logging, `Makefile`.

---

## Architecture

```
Upload (jpg/png/mp4)
      │
      ▼
  YOLODetector (detectors/yolo_detector.py)
  ├─ OBB path: (x_c,y_c,w,h,angle,conf,cls) → cv2.boxPoints → x1y1x2y2
  └─ BBox path: xyxy + conf
      │
      ▼
  DeepSortTracker (tracking/deepsort_tracker.py)
  └─ deep-sort-realtime (or fallback IOU tracker)
      │
      ▼
  Preprocess (utils.py)
  deskew → gray → CLAHE → Otsu → morph → resize → invert
      │
      ▼
  EasyOCRRecognizer (recognizers/easyocr_recognizer.py)
  └─ allowlist A-Z0-9, best-confidence selection
      │
      ▼
  clean_plate_text() → CSV / overlay
```

---

## Repo structure

```
ANPR/
├── app.py                      # Streamlit app (production entrypoint)
├── anpr_config.py              # Central config — reads .env, single source of truth
├── utils.py                    # deskew_and_clean_plate(), clean_plate_text(), order_points()
├── detectors/
│   ├── __init__.py
│   └── yolo_detector.py        # YOLO wrapper (OBB + bbox)
├── recognizers/
│   ├── __init__.py
│   └── easyocr_recognizer.py
├── tracking/
│   ├── __init__.py
│   └── deepsort_tracker.py     # DeepSORT with SORT fallback
├── sort/
│   ├── __init__.py
│   └── sort.py                 # Lightweight SORT (filterpy Kalman or pure IOU)
├── models/
│   ├── vehicle_detection_model/best.pt   # YOLO OBB — vehicle detector (LFS)
│   └── number_plated/number_plates_model.pt # YOLO — plate detector (LFS)
├── yolo11n.pt                  # Base weights (LFS)
├── yolov8n.pt                  # Base weights (LFS)
├── easy_ocr.py                 # Standalone: vehicle→plate→EasyOCR→CSV/video
├── final_ocr_keras.py          # Standalone: vehicle→plate→Keras-OCR
├── Nanonets.py                 # Standalone: vehicle→plate→Nanonets-OCR-s
├── data_downloader.py          # Roboflow download (env-based auth)
├── model_trainer.py            # YOLO train + MLflow/DagsHub (env-based)
├── roboflow_app.py             # Roboflow hosted inference demo
├── requirements.txt            # Pinned production deps
├── pyproject.toml              # PEP 517 build + ruff config
├── setup.py
├── Dockerfile                  # python:3.10-slim + ffmpeg + healthcheck
├── docker-compose.yml
├── Makefile
├── .env.example                # Copy to .env — never commit real keys
├── .gitignore                  # Production ignore (keeps *.pt via LFS)
├── .gitattributes              # *.pt / *.pth / *.onnx → LFS
├── .dockerignore
└── test_video.mp4              # Sample video
```

---

## Quick start

### 1. Clone (with LFS)

```bash
git lfs install
git clone https://github.com/RisAhamed/ANPR.git
cd ANPR
```

> If you already cloned without LFS, run `git lfs pull` to fetch `*.pt` files. Verify with `ls -lh models/**.pt`.

### 2. Python env

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Env

```bash
cp .env.example .env
# edit .env — at minimum set ROBOFLOW_API_KEY / DAGSHUB_TOKEN if you use those scripts
```

### 4. Run

```bash
streamlit run app.py --server.port 8501 --server.headless true
# or
make run
```

Open http://localhost:8501 → upload an image or `test_video.mp4`.

---

## Docker (recommended for prod)

```bash
cp .env.example .env   # fill secrets
docker compose up --build
# app → http://localhost:8501  (healthcheck: /_stcore/health)
```

Manual build:

```bash
docker build -t anpr:latest .
docker run --env-file .env -p 8501:8501 -v ./outputs:/app/outputs anpr:latest
```

---

## Configuration (.env)

All secrets and tunables come from env — **no hard-coded keys** (fixed from original repo).

| Var | Default | Description |
|-----|---------|-------------|
| `ROBOFLOW_API_KEY` | — | Roboflow API key (data_downloader / roboflow_app) |
| `ROBOFLOW_WORKSPACE` | `objectdetection-twsk1` | Workspace |
| `ROBOFLOW_PROJECT` | `licenseplate-mswpd-lbgrc` | Project |
| `ROBOFLOW_VERSION` | `1` | Version |
| `DAGSHUB_TOKEN` | — | DagsHub/MLflow token |
| `MLFLOW_TRACKING_URI` | `https://dagshub.com/RisAhamed/ANPR.mlflow` | MLflow URI |
| `ANPR_VEHICLE_MODEL` | `models/vehicle_detection_model/best.pt` | Vehicle YOLO weights |
| `ANPR_PLATE_MODEL` | `models/number_plated/number_plates_model.pt` | Plate YOLO weights |
| `ANPR_VEHICLE_CONF` | `0.2` | Vehicle det. threshold |
| `ANPR_PLATE_CONF` | `0.4` | Plate det. threshold |
| `ANPR_DEVICE` | `auto` | `auto|cpu|cuda|mps` |

See [.env.example](.env.example).

---

## Models & Git LFS

Live weights **are committed** via Git LFS:

- `models/vehicle_detection_model/best.pt` (6.5 MB — OBB vehicle detector)
- `models/number_plated/number_plates_model.pt` (16 MB — plate detector)
- `yolo11n.pt` / `yolov8n.pt` (base checkpoints)

```bash
# verify LFS
git lfs ls-files
cat .gitattributes   # *.pt filter=lfs diff=lfs merge=lfs -text
```

> `.gitignore` **does not** ignore `*.pt` — weights are versioned intentionally. Large datasets (`LicensePlate-1/`, `final_ocr/`, `runs/`, `outputs/`) are ignored.

---

## Usage

### Streamlit (app.py)

- **Image:** upload → see detected crops + preprocessed view → annotated image + CSV download.
- **Video:** upload → every Nth frame is detected+tracked → selective OCR (skips already-confident IDs) → live annotated preview + sidebar table → final CSV.

Tune in sidebar:
- `Detection confidence` (default 0.4)
- `Frame interval` (process every Nth frame — 2 is a good speed/accuracy tradeoff)

### Standalone scripts (batch / headless)

```bash
# EasyOCR pipeline (vehicle→plate→OCR) — writes final_ocr/ + summary CSV
python easy_ocr.py

# Keras-OCR variant
python final_ocr_keras.py

# Nanonets transformer variant (heavy — needs transformers + GPU)
python Nanonets.py
```

Each writes `annotated_video.mp4` + per-vehicle crops + `summary_all_events.csv`.

---

## Training & dataset

```bash
# 1) Download dataset from Roboflow (needs .env)
python data_downloader.py   # → LicensePlate-1/data.yaml

# 2) Train (logs to DagsHub MLflow if DAGSHUB_TOKEN set, else local only)
python model_trainer.py          # default: yolov8x 50 epochs
# or
python -c "from model_trainer import train_yolov8; train_yolov8('yolov8n', epochs=30)"
```

Artifacts: `runs/detect/<size>-finetuned/weights/best.pt`

---

## API & scripts

| Script | Purpose | Key deps |
|--------|---------|----------|
| `app.py` | Streamlit prod UI | streamlit, ultralytics, easyocr, deep-sort-realtime |
| `utils.py` | `deskew_and_clean_plate`, `clean_plate_text`, `order_points` | opencv, scipy |
| `detectors/yolo_detector.py` | `YOLODetector.detect_plates()` | ultralytics |
| `recognizers/easyocr_recognizer.py` | `EasyOCRRecognizer.recognize_plate()` | easyocr |
| `tracking/deepsort_tracker.py` | `DeepSortTracker.track_plates()` | deep-sort-realtime / filterpy |
| `sort/sort.py` | `Sort.update()` (classic SORT) | filterpy |
| `data_downloader.py` | Roboflow → `yolov8` dataset | roboflow |
| `model_trainer.py` | YOLO train + MLflow | ultralytics, mlflow, dagshub |
| `roboflow_app.py` | Hosted inference demo | inference, supervision |

---

## Production checklist

- [x] Secrets in `.env` (no hard-coded `pyEqWe…` / `822dab…` keys)
- [x] Model weights via Git LFS (live in repo)
- [x] `.gitignore` cleans `__pycache__`, `runs/`, `final_ocr/`, `LicensePlate-1/`, `.idea/`, `.vscode/`
- [x] `.idea/` and `.vscode/` removed from git history (untracked)
- [x] Pinned `requirements.txt` + `pyproject.toml` + `setup.py`
- [x] `Dockerfile` + `docker-compose.yml` + `HEALTHCHECK`
- [x] `Makefile` (`install`, `run`, `docker`, `lint`, `test`, `clean`)
- [x] Central `anpr_config.py` + modular `detectors` / `recognizers` / `tracking`
- [x] Graceful fallbacks (DeepSORT → SORT → IOU, EasyOCR GPU → CPU)
- [x] Logging + input validation + path handling via `pathlib`

---

## Troubleshooting

**`*.pt` files are 0 bytes / pointer text** → you didn't pull LFS: `git lfs install && git lfs pull`.

**`ImportError: ultralytics / easyocr`** → `pip install -r requirements.txt` in a fresh venv (Python 3.9–3.11). For GPU, install `torch` with CUDA first.

**`ROBOFLOW_API_KEY not set`** → `cp .env.example .env` and fill the key.

**`Cannot open video`** → check codec; Dockerfile installs `ffmpeg`. Locally: `pip install opencv-python` (not headless) for GUI.

**OCR poor on tilted plates** → `utils.deskew_and_clean_plate` already does perspective warp; try lowering `ANPR_PLATE_CONF` or using the Nanonets variant for harder cases.

---

## Contributing

PRs welcome. Run `make lint && make test` before submitting. Keep `*.pt` adds via `git lfs track "*.pt"`.

---

## License

MIT — see [LICENSE](LICENSE) (add one if missing; template below).

```
MIT License — Copyright (c) 2025 Riswan Ahamed
```

---

## Citation

If you use this project, please cite:

```
@software{anpr_2025,
  title = {ANPR — Production-Grade Automatic Number Plate Recognition},
  author = {Riswan Ahamed},
  year = {2025},
  url = {https://github.com/RisAhamed/ANPR}
}
```
