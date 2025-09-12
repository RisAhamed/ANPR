# License Plate Recognition Application
 
This project is an Automatic Number Plate Recognition (ANPR) application that can detect and extract license plate numbers from images or videos. It supports multiple OCR (Optical Character Recognition) techniques, including Keras OCR, OpenALPR, and EasyOCR.

### Main Function Points
- Multiple OCR techniques: Keras OCR, OpenALPR, and EasyOCR
- Detect and extract license plate numbers from images or videos
- User interface to select preferred OCR technique and upload images

### Technology Stack
- Python 3.8+
- Flask (web application framework)
- Keras, OpenALPR, and EasyOCR (OCR libraries)
- YOLO (object detection model)


## Prerequisites

- Python 3.8+
- pip

## Installation

1. Clone the repository:
```bash
git clone https://repository-url.git
cd project-directory
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

```bash
python app.py
```

### Features

- Multiple OCR Techniques:
  - Keras OCR
  - OpenALPR
  - EasyOCR

### How to Use

1. Launch the application
2. Select your preferred OCR technique from the UI
3. Upload an image
4. Let the processor run
5. View tracked objects and corresponding license plate numbers

## Supported OCR Techniques

- [x] Keras OCR
- [x] OpenALPR
- [x] EasyOCR

## Requirements

See `requirements.txt` for a full list of Python dependencies.

