from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="anpr",
    version="1.0.0",
    author="Riswan Ahamed",
    description="Automatic Number Plate Recognition — YOLO + EasyOCR + DeepSORT/SORT (production-grade)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RisAhamed/ANPR",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "streamlit>=1.35",
        "numpy<2",
        "opencv-python-headless>=4.9",
        "ultralytics>=8.2",
        "easyocr>=1.7",
        "deep-sort-realtime>=1.3",
        "scipy>=1.11",
        "pandas>=2.0",
        "python-dotenv>=1.0",
        "filterpy>=1.4",
    ],
    extras_require={
        "train": ["roboflow>=1.1", "mlflow>=2.15", "dagshub>=0.3"],
        "ocr-extra": ["keras-ocr>=0.9", "tensorflow>=2.16", "pytesseract>=0.3"],
        "dev": ["ruff>=0.6", "pytest>=8"],
    },
    include_package_data=True,
)
