# 🏠 MLflow House Price Prediction Pipeline

A production-ready machine learning pipeline for house price prediction with full MLOps lifecycle management using MLflow, FastAPI, and Docker.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![MLflow](https://img.shields.io/badge/MLflow-2.8.1-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green)
![Docker](https://img.shields.io/badge/Docker-ready-blue)

---

## 🎯 Features

- **End-to-End ML Pipeline**: Data ingestion, preprocessing, training & deployment
- **Experiment Tracking**: MLflow-based experiment management
- **REST API**: FastAPI prediction service
- **Drift Detection**: Data & model drift monitoring via Evidently
- **CI/CD**: GitHub Actions workflows for continuous testing & retraining
- **Containerized**: Docker-ready deployment
- **Monitoring Dashboard**: Real-time model performance tracking

---

## 📁 Project Structure

```bash
house-price-mlflow-pipeline/
├── .github/workflows/       # CI/CD pipelines
├── data/                    # Raw & processed data
├── src/                     # Source code
│   ├── data/                # Data ingestion & preprocessing
│   ├── models/              # Model training & evaluation
│   ├── monitoring/          # Drift detection
│   └── api/                 # FastAPI application
├── config/                  # YAML configs
├── docker/                  # Docker deployment files
├── tests/                   # Unit tests
└── notebooks/               # Jupyter notebooks
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
git clone <your-repo-url>
cd house-price-mlflow-pipeline

pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
```

### 2. Download Data

```bash
make download-data
# or
bash scripts/download_data.sh
```

### 3. Train Model

```bash
make train
# or
python src/models/train.py
```

### 4. Start MLflow UI

```bash
make mlflow-ui
# Access at http://localhost:5000
```

### 5. Start API Server

```bash
make serve
# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

## 📊 Usage Examples

### Python API

```python
from src.data.ingestion import DataIngestion
from src.data.preprocessing import DataPreprocessor
from src.models.train import ModelTrainer

# Load and preprocess data
ingestion = DataIngestion()
df = ingestion.ingest()

preprocessor = DataPreprocessor()
X_train, X_test, y_train, y_test = preprocessor.preprocess(df)

# Train model
trainer = ModelTrainer()
model = trainer.train(X_train, X_test, y_train, y_test)
```

### REST API

```bash
# Health check
curl http://localhost:8000/health

# Make prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "longitude": -122.23,
    "latitude": 37.88,
    "housing_median_age": 41.0,
    "total_rooms": 880.0,
    "total_bedrooms": 129.0,
    "population": 322.0,
    "households": 126.0,
    "median_income": 8.3252,
    "ocean_proximity": "NEAR BAY"
  }'
```

### Python Requests

```python
import requests

url = "http://localhost:8000/predict"
data = {
    "longitude": -122.23,
    "latitude": 37.88,
    "housing_median_age": 41.0,
    "total_rooms": 880.0,
    "total_bedrooms": 129.0,
    "population": 322.0,
    "households": 126.0,
    "median_income": 8.3252,
    "ocean_proximity": "NEAR BAY"
}

response = requests.post(url, json=data)
print(response.json())
```

## 🐳 Docker Deployment

### Build and Run

```bash
# Build image
make docker-build

# Run container
make docker-run

# Or use docker-compose
cd docker
docker-compose up -d
```

### Access Services

- API: http://localhost:8000
- MLflow UI: http://localhost:5000

## 🔧 Configuration

### Model Configuration (`config/model_config.yaml`)

```yaml
random_forest:
  n_estimators: 100
  max_depth: 20
  min_samples_split: 5
```

### Main Configuration (`config/config.yaml`)

```yaml
model:
  type: "random_forest"
  target_column: "median_house_value"

mlflow:
  tracking_uri: "./mlflow"
  experiment_name: "house-price-prediction"
```

## 📈 Monitoring & Drift Detection

```python
from src.monitoring.drift_detection import DriftDetector

# Initialize detector
detector = DriftDetector()

# Detect drift
drift_summary = detector.detect_data_drift(
    current_data=new_data,
    reference_data=training_data
)

print(f"Drift Detected: {drift_summary['drift_detected']}")
```

## 🧪 Testing

```bash
# Run all tests
make test

# Run specific test
pytest tests/test_api.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

## 🔄 CI/CD Pipeline

### Continuous Integration (`.github/workflows/ci.yml`)
- Runs on every push/PR
- Linting, testing, security checks
- Multi-version Python testing

### Retraining & Deployment (`.github/workflows/retrain_deploy.yml`)
- Scheduled weekly retraining
- Automated model deployment
- Docker image building and pushing

## 📝 Available Commands

```bash
make install          # Install dependencies
make setup           # Create directories
make download-data   # Download dataset
make train           # Train model
make serve           # Start API server
make mlflow-ui       # Start MLflow UI
make test            # Run tests
make format          # Format code
make lint            # Lint code
make clean           # Clean artifacts
make docker-build    # Build Docker image
make docker-run      # Run Docker container
```

## 🛠️ Development

### Add New Model

1. Add configuration in `config/model_config.yaml`
2. Update `src/models/train.py` to support new model
3. Test and train

### Add New Features

1. Modify `src/data/preprocessing.py`
2. Update configuration
3. Retrain model

## 📊 MLflow Tracking

All experiments are tracked in MLflow:
- Parameters (hyperparameters)
- Metrics (MAE, RMSE, R², MAPE)
- Artifacts (models, plots, preprocessors)
- Models (versioned and registered)

Access MLflow UI at `http://localhost:5000`

## 🔐 Environment Variables

```bash
MLFLOW_TRACKING_URI=./mlflow
PYTHONUNBUFFERED=1
```

## 📚 Tech Stack

- **ML Framework**: scikit-learn, TensorFlow
- **Experiment Tracking**: MLflow
- **API Framework**: FastAPI
- **Data Processing**: Pandas, NumPy
- **Monitoring**: Evidently AI
- **Testing**: pytest
- **CI/CD**: GitHub Actions
- **Containerization**: Docker
- **Code Quality**: black, flake8, pylint

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 👤 Author

[Pihul Waradkar](https://github.com/pihulwaradkar)

## 🙏 Acknowledgments

- California Housing Dataset from scikit-learn
- MLflow for experiment tracking
- FastAPI for the amazing web framework
- Evidently AI for drift detection

## 📮 Contact

For questions or feedback, please open an issue on GitHub.

---

**Happy Predicting! 🏠📈**
