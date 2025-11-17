"""
Basic tests to ensure modules can be imported
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_imports():
    """Test that all modules can be imported"""
    try:
        import src.data.ingestion
        import src.data.preprocessing
        import src.models.train
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


def test_config_files_exist():
    """Test that config files exist"""
    assert os.path.exists("config/config.yaml")
    assert os.path.exists("config/model_config.yaml")


def test_directory_structure():
    """Test that required directories exist"""
    required_dirs = [
        "src/data",
        "src/models",
        "src/api",
        "src/monitoring",
        "config",
        "tests"
    ]
    
    for dir_path in required_dirs:
        assert os.path.exists(dir_path), f"Directory {dir_path} does not exist"


def test_requirements_file():
    """Test that requirements.txt exists and is readable"""
    assert os.path.exists("requirements.txt")
    
    with open("requirements.txt", 'r') as f:
        content = f.read()
        assert "mlflow" in content
        assert "fastapi" in content
        assert "scikit-learn" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])