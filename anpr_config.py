"""
Central configuration — single source of truth for all scripts.
All secrets come from environment (.env). No hard-coded keys.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

# ── Models ────────────────────────────────────────────
VEHICLE_MODEL_PATH: Path = Path(os.getenv("ANPR_VEHICLE_MODEL", str(BASE_DIR / "models/vehicle_detection_model/best.pt")))
PLATE_MODEL_PATH: Path = Path(os.getenv("ANPR_PLATE_MODEL", str(BASE_DIR / "models/number_plated/number_plates_model.pt")))

VEHICLE_CONF: float = float(os.getenv("ANPR_VEHICLE_CONF", "0.2"))
PLATE_CONF: float = float(os.getenv("ANPR_PLATE_CONF", "0.4"))
OCR_ENGINE: str = os.getenv("ANPR_OCR_ENGINE", "easyocr")  # easyocr | keras_ocr | nanonets

# ── Roboflow ──────────────────────────────────────────
ROBOFLOW_API_KEY: str | None = os.getenv("ROBOFLOW_API_KEY")
ROBOFLOW_WORKSPACE: str = os.getenv("ROBOFLOW_WORKSPACE", "objectdetection-twsk1")
ROBOFLOW_PROJECT: str = os.getenv("ROBOFLOW_PROJECT", "licenseplate-mswpd-lbgrc")
ROBOFLOW_VERSION: int = int(os.getenv("ROBOFLOW_VERSION", "1"))

# ── DagsHub / MLflow ─────────────────────────────────
DAGSHUB_REPO_OWNER: str = os.getenv("DAGSHUB_REPO_OWNER", "RisAhamed")
DAGSHUB_REPO_NAME: str = os.getenv("DAGSHUB_REPO_NAME", "ANPR")
MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "https://dagshub.com/RisAhamed/ANPR.mlflow")
DAGSHUB_TOKEN: str | None = os.getenv("DAGSHUB_TOKEN")

# ── Runtime ───────────────────────────────────────────
DEVICE: str = os.getenv("ANPR_DEVICE", "auto")

VEHICLE_CLASSES: list[str] = [
    '2-axle-trailer', '2-axle-truck', '3-axle-trailer', '3-axle-truck',
    '4-axle-trailer', '4-axle-truck', '5-axle-truck', '5+-axle-truck-trailer',
    'Ambulance', 'Auto', 'Autorickshaw', 'Bicycle', 'Bus-2-axle', 'Bus-3-axle',
    'Car', 'Firetruck', 'handcart', 'HCM/EME', 'LCV', 'Minivan',
    'Motorcycle', 'Tractor'
]
