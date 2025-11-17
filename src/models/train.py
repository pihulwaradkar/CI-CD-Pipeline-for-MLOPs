"""
Model Training Module with MLflow Integration
"""

import logging
import yaml
import joblib
import os
from datetime import datetime
import pandas as pd
import numpy as np

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import sys
sys.path.append('.')
from src.data.ingestion import DataIngestion
from src.data.preprocessing import DataPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Class to handle model training with MLflow tracking"""
    
    def __init__(self, config_path: str = "config/config.yaml",
                 model_config_path: str = "config/model_config.yaml"):
        """
        Initialize ModelTrainer
        
        Args:
            config_path: Path to main configuration file
            model_config_path: Path to model configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        with open(model_config_path, 'r') as f:
            self.model_config = yaml.safe_load(f)
        
        self.model_type = self.config['model']['type']
        self.mlflow_tracking_uri = self.config['mlflow']['tracking_uri']
        self.experiment_name = self.config['mlflow']['experiment_name']
        
        # Set up MLflow
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        
        self.model = None
        
    def get_model(self):
        """
        Get model instance based on configuration
        
        Returns:
            Model instance
        """
        logger.info(f"Initializing {self.model_type} model...")
        
        if self.model_type == 'random_forest':
            params = self.model_config['random_forest']
            return RandomForestRegressor(**params)
        
        elif self.model_type == 'gradient_boosting':
            params = self.model_config['gradient_boosting']
            return GradientBoostingRegressor(**params)
        
        elif self.model_type == 'linear_regression':
            params = self.model_config['linear_regression']
            return LinearRegression(**params)
        
        elif self.model_type == 'ridge_regression':
            params = self.model_config['ridge_regression']
            return Ridge(**params)
        
        elif self.model_type == 'lasso_regression':
            params = self.model_config['lasso_regression']
            return Lasso(**params)
        
        else:
            logger.warning(f"Unknown model type: {self.model_type}. Using RandomForest.")
            return RandomForestRegressor(**self.model_config['random_forest'])
    
    def calculate_metrics(self, y_true, y_pred) -> dict:
        """
        Calculate regression metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        }
        
        return metrics
    
    def train(self, X_train, X_test, y_train, y_test):
        """
        Train model with MLflow tracking
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training target
            y_test: Test target
            
        Returns:
            Trained model
        """
        with mlflow.start_run(run_name=f"{self.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            
            # Log parameters
            model_params = self.model_config.get(self.model_type, {})
            mlflow.log_params(model_params)
            mlflow.log_param("model_type", self.model_type)
            
            # Initialize and train model
            logger.info("Training model...")
            self.model = self.get_model()
            self.model.fit(X_train, y_train)
            logger.info("Training complete!")
            
            # Make predictions
            y_train_pred = self.model.predict(X_train)
            y_test_pred = self.model.predict(X_test)
            
            # Calculate metrics
            train_metrics = self.calculate_metrics(y_train, y_train_pred)
            test_metrics = self.calculate_metrics(y_test, y_test_pred)
            
            # Log metrics
            for metric_name, metric_value in train_metrics.items():
                mlflow.log_metric(f"train_{metric_name}", metric_value)
                logger.info(f"Train {metric_name}: {metric_value:.4f}")
            
            for metric_name, metric_value in test_metrics.items():
                mlflow.log_metric(f"test_{metric_name}", metric_value)
                logger.info(f"Test {metric_name}: {metric_value:.4f}")
            
            # Log feature importance if available
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                # Save feature importance plot
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 6))
                plt.barh(feature_importance['feature'][:10], 
                        feature_importance['importance'][:10])
                plt.xlabel('Importance')
                plt.title('Top 10 Feature Importances')
                plt.tight_layout()
                
                os.makedirs('artifacts', exist_ok=True)
                plt.savefig('artifacts/feature_importance.png')
                mlflow.log_artifact('artifacts/feature_importance.png')
                plt.close()
                
                logger.info("Feature importance logged")
            
            # Log model
            mlflow.sklearn.log_model(self.model, "model")
            
            # Save model locally
            os.makedirs('models', exist_ok=True)
            model_path = "models/model.pkl"
            joblib.dump(self.model, model_path)
            logger.info(f"Model saved to {model_path}")
            
            # Log run info
            run_id = mlflow.active_run().info.run_id
            logger.info(f"MLflow Run ID: {run_id}")
            
            return self.model
    
    def load_model(self, path: str):
        """
        Load a trained model
        
        Args:
            path: Path to saved model
        """
        self.model = joblib.load(path)
        logger.info(f"Model loaded from {path}")
        return self.model


def main():
    """Main training pipeline"""
    logger.info("="*60)
    logger.info("STARTING TRAINING PIPELINE")
    logger.info("="*60)
    
    # 1. Data Ingestion
    logger.info("\n[1/3] Data Ingestion")
    ingestion = DataIngestion()
    df = ingestion.ingest()
    
    # 2. Data Preprocessing
    logger.info("\n[2/3] Data Preprocessing")
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.preprocess(df)
    preprocessor.save_preprocessor()
    
    # 3. Model Training
    logger.info("\n[3/3] Model Training")
    trainer = ModelTrainer()
    model = trainer.train(X_train, X_test, y_train, y_test)
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING PIPELINE COMPLETE!")
    logger.info("="*60)
    logger.info(f"\nModel Type: {trainer.model_type}")
    logger.info(f"MLflow Tracking URI: {trainer.mlflow_tracking_uri}")
    logger.info("\nNext steps:")
    logger.info("  - Run 'make mlflow-ui' to view experiments")
    logger.info("  - Run 'make serve' to start the API")


if __name__ == "__main__":
    main()
