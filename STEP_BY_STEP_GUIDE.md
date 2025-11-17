#!/bin/bash

# MLflow House Price Prediction Pipeline - Installation Script
# For GitHub Codespaces

set -e  # Exit on error

echo "🚀 Starting MLflow Pipeline Setup..."

# Step 1: Upgrade pip and install build tools
echo "📦 Step 1: Installing build tools..."
pip install --upgrade pip setuptools wheel

# Step 2: Create directory structure
echo "📁 Step 2: Creating project structure..."
mkdir -p .github/workflows \
  data/raw data/processed data/feature_store \
  src/data src/models src/monitoring src/api \
  notebooks tests mlflow docker config scripts

# Create __init__.py files
touch src/__init__.py \
  src/data/__init__.py \
  src/models/__init__.py \
  src/monitoring/__init__.py \
  src/api/__init__.py \
  tests/__init__.py

# Create placeholder files
touch mlflow/.gitkeep

echo "✅ Directory structure created!"

# Step 3: Create requirements.txt
echo "📝 Step 3: Creating requirements.txt..."
cat > requirements.txt << 'EOF'
# Build Tools
setuptools>=65.0.0
wheel>=0.38.0

# Core ML Libraries
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
tensorflow==2.13.0

# MLflow & Tracking
mlflow==2.8.1
databricks-cli==0.18.0

# API & Deployment
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
python-multipart==0.0.6

# Data Processing
pyarrow==13.0.0
openpyxl==3.1.2

# Monitoring & Drift Detection
evidently==0.4.10
scipy==1.11.3

# Feature Store (Optional)
feast==0.35.0

# Utilities
python-dotenv==1.0.0
pyyaml==6.0.1
requests==2.31.0
boto3==1.28.85

# Visualization
matplotlib==3.7.3
seaborn==0.12.2
plotly==5.17.0
EOF

echo "✅ requirements.txt created!"

# Step 4: Create requirements-dev.txt
echo "📝 Step 4: Creating requirements-dev.txt..."
cat > requirements-dev.txt << 'EOF'
# Testing
pytest==7.4.3
pytest-cov==4.1.0
pytest-mock==3.12.0
httpx==0.25.1

# Code Quality
black==23.11.0
flake8==6.1.0
isort==5.12.0
mypy==1.7.1
pylint==3.0.2

# Jupyter
jupyter==1.0.0
ipykernel==6.26.0
notebook==7.0.6

# Documentation
mkdocs==1.5.3
mkdocs-material==9.4.14
EOF

echo "✅ requirements-dev.txt created!"

# Step 5: Create setup.py
echo "📝 Step 5: Creating setup.py..."
cat > setup.py << 'EOF'
from setuptools import setup, find_packages

setup(
    name="house_price_mlflow_pipeline",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        line.strip()
        for line in open("requirements.txt").readlines()
        if line.strip() and not line.startswith("#")
    ],
    python_requires=">=3.9",
    author="Your Name",
    description="MLflow-based House Price Prediction Pipeline",
    long_description=open("README.md").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
)
EOF

echo "✅ setup.py created!"

# Step 6: Create .gitignore
echo "📝 Step 6: Creating .gitignore..."
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# MLflow
mlflow/
mlruns/
artifacts/

# Data
data/raw/*.csv
data/processed/*.parquet
*.db

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# IDE
.vscode/
.idea/
*.swp
*.swo

# Docker
*.log

# Environment
.env
.env.local

# OS
.DS_Store
Thumbs.db

# Models
models/*.pkl
models/*.h5
models/*.joblib
EOF

echo "✅ .gitignore created!"

# Step 7: Create Makefile
echo "📝 Step 7: Creating Makefile..."
cat > Makefile << 'EOF'
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
EOF

echo "✅ Makefile created!"

# Step 8: Install main dependencies
echo "📦 Step 8: Installing main dependencies (this may take a few minutes)..."
pip install -r requirements.txt

echo "✅ Main dependencies installed!"

# Step 9: Install development dependencies
echo "📦 Step 9: Installing development dependencies..."
pip install -r requirements-dev.txt

echo "✅ Development dependencies installed!"

# Step 10: Install package in editable mode
echo "📦 Step 10: Installing package in editable mode..."
pip install -e .

echo "✅ Package installed!"

# Step 11: Set up MLflow tracking
echo "🔧 Step 11: Configuring MLflow..."
export MLFLOW_TRACKING_URI=./mlflow
echo 'export MLFLOW_TRACKING_URI=./mlflow' >> ~/.bashrc

echo "✅ MLflow configured!"

# Final message
echo ""
echo "🎉 Setup Complete!"
echo ""
echo "📋 Next Steps:"
echo "  1. Run 'source ~/.bashrc' to load MLflow environment variables"
echo "  2. Run 'make mlflow-ui' to start MLflow UI (http://localhost:5000)"
echo "  3. Start building your pipeline components!"
echo ""
echo "💡 Useful Commands:"
echo "  - make train       # Train the model"
echo "  - make serve       # Start FastAPI server"
echo "  - make test        # Run tests"
echo "  - make format      # Format code"
echo ""