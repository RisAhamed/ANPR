import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from roboflow import Roboflow
import anpr_config as cfg

api_key = cfg.ROBOFLOW_API_KEY or os.getenv("ROBOFLOW_API_KEY")
if not api_key:
    print("ERROR: ROBOFLOW_API_KEY not set. Copy .env.example -> .env and fill it.", file=sys.stderr)
    sys.exit(1)

rf = Roboflow(api_key=api_key)
project = rf.workspace(cfg.ROBOFLOW_WORKSPACE).project(cfg.ROBOFLOW_PROJECT)
version = project.version(cfg.ROBOFLOW_VERSION)

dataset = version.download("yolov8")
 