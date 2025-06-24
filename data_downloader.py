import os
from roboflow import Roboflow
rf = Roboflow(api_key="pyEqWe4YjqAAcdOOz8l0")
project = rf.workspace("objectdetection-twsk1").project("licenseplate-mswpd-lbgrc")
version = project.version(1)

dataset = version.download("yolov8")
 