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
