.PHONY: install run docker lint clean

install:
	pip install -r requirements.txt

run:
	streamlit run app.py --server.port 8501 --server.headless true

docker:
	docker compose up --build

docker-run:
	docker build -t anpr:latest . && docker run --env-file .env -p 8501:8501 anpr:latest

lint:
	ruff check . || true
	python -m py_compile app.py utils.py anpr_config.py detectors/yolo_detector.py recognizers/easyocr_recognizer.py tracking/deepsort_tracker.py

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf outputs final_ocr final_keras_ocr runs

test:
	python -c "from detectors.yolo_detector import YOLODetector; print('detector OK')"
	python -c "from recognizers.easyocr_recognizer import EasyOCRRecognizer; print('ocr OK')"
	python -c "from tracking.deepsort_tracker import DeepSortTracker; print('tracker OK')"
	python -c "import utils; print('utils OK')"
