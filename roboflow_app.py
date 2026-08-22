from inference import get_model
import supervision as sv
import cv2
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
import anpr_config as cfg

# Roboflow API key from env — never hard-code
if cfg.ROBOFLOW_API_KEY:
    os.environ["ROBOFLOW_API_KEY"] = cfg.ROBOFLOW_API_KEY

# define the image file to use for inference
image_file = Path(__file__).parent / "download (1).jpeg"
image = cv2.imread(str(image_file))
if image is None:
    raise FileNotFoundError(f"Image not found: {image_file}")

# load a pre-trained yolov8n model
model = get_model(model_id="licenseplate-mswpd-lbgrc/1")

# run inference on our chosen image, image can be a url, a numpy array, a PIL image, etc.
results = model.infer(image)[0]

# load the results into the supervision Detections api
detections = sv.Detections.from_inference(results)

# create supervision annotators
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# annotate the image with our inference results
annotated_image = bounding_box_annotator.annotate(
    scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections)

# display the image
sv.plot_image(annotated_image)