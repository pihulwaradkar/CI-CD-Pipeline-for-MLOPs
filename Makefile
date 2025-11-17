.PHONY: install setup train serve test clean

install:
	pip install --upgrade pip setuptools wheel
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pip install -e .

setup:
	mkdir -p data/{raw,processed,feature_store} mlflow
	chmod +x scripts/*.sh

download-data:
	bash scripts/download_data.sh

train:
	python src/models/train.py

serve:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

mlflow-ui:
	mlflow ui --host 0.0.0.0 --port 5000

test:
	pytest tests/ -v --cov=src

format:
	black src/ tests/
	isort src/ tests/

lint:
	flake8 src/ tests/
	pylint src/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info

docker-build:
	docker build -f docker/Dockerfile -t house-price-api:latest .

docker-run:
	docker run -p 8000:8000 house-price-api:latest
