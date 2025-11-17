"""
Data Preprocessing Module
Handles feature engineering, scaling, and data transformation
"""

import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
import yaml
import joblib
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Class to handle data preprocessing and feature engineering"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize DataPreprocessor
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.target_column = self.config['model']['target_column']
        self.numerical_features = self.config['model']['features']['numerical']
        self.categorical_features = self.config['model']['features']['categorical']
        self.test_size = self.config['data']['test_size']
        self.random_state = self.config['data']['random_state']
        self.scaling_method = self.config['model']['scaling']['method']
        
        self.scaler = None
        self.imputer = None
        
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with handled missing values
        """
        logger.info("Handling missing values...")
        df_copy = df.copy()
        
        # Fill numerical missing values with median
        for col in self.numerical_features:
            if col in df_copy.columns and df_copy[col].isnull().sum() > 0:
                median_value = df_copy[col].median()
                df_copy[col].fillna(median_value, inplace=True)
                logger.info(f"Filled {col} missing values with median: {median_value:.2f}")
        
        # Fill categorical missing values with mode
        for col in self.categorical_features:
            if col in df_copy.columns and df_copy[col].isnull().sum() > 0:
                mode_value = df_copy[col].mode()[0]
                df_copy[col].fillna(mode_value, inplace=True)
                logger.info(f"Filled {col} missing values with mode: {mode_value}")
        
        return df_copy
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features from existing ones
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Engineering new features...")
        df_copy = df.copy()
        
        # Create new features if columns exist
        if 'total_rooms' in df_copy.columns and 'households' in df_copy.columns:
            df_copy['rooms_per_household'] = df_copy['total_rooms'] / df_copy['households']
        
        if 'total_bedrooms' in df_copy.columns and 'total_rooms' in df_copy.columns:
            df_copy['bedrooms_per_room'] = df_copy['total_bedrooms'] / df_copy['total_rooms']
        
        if 'population' in df_copy.columns and 'households' in df_copy.columns:
            df_copy['population_per_household'] = df_copy['population'] / df_copy['households']
        
        # Handle any infinite values created by division
        df_copy.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Fill any new NaN values with 0
        df_copy.fillna(0, inplace=True)
        
        logger.info(f"Created {len([c for c in df_copy.columns if c not in df.columns])} new features")
        
        return df_copy
    
    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical variables
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with encoded categorical variables
        """
        logger.info("Encoding categorical variables...")
        df_copy = df.copy()
        
        for col in self.categorical_features:
            if col in df_copy.columns:
                # Use one-hot encoding
                dummies = pd.get_dummies(df_copy[col], prefix=col, drop_first=True)
                df_copy = pd.concat([df_copy, dummies], axis=1)
                df_copy.drop(col, axis=1, inplace=True)
                logger.info(f"Encoded {col} with {len(dummies.columns)} dummy variables")
        
        return df_copy
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame = None) -> tuple:
        """
        Scale numerical features
        
        Args:
            X_train: Training features
            X_test: Test features (optional)
            
        Returns:
            Tuple of scaled training and test features
        """
        logger.info(f"Scaling features using {self.scaling_method} scaler...")
        
        # Select scaler
        if self.scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif self.scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.scaling_method == 'robust':
            self.scaler = RobustScaler()
        else:
            logger.warning(f"Unknown scaling method: {self.scaling_method}. Using StandardScaler.")
            self.scaler = StandardScaler()
        
        # Fit and transform training data
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        # Transform test data if provided
        if X_test is not None:
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled, None
    
    def split_data(self, df: pd.DataFrame) -> tuple:
        """
        Split data into train and test sets
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Splitting data into train and test sets...")
        
        # Separate features and target
        X = df.drop(self.target_column, axis=1)
        y = df[self.target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state
        )
        
        logger.info(f"Train set size: {X_train.shape[0]}")
        logger.info(f"Test set size: {X_test.shape[0]}")
        
        return X_train, X_test, y_train, y_test
    
    def preprocess(self, df: pd.DataFrame) -> tuple:
        """
        Complete preprocessing pipeline
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (X_train_scaled, X_test_scaled, y_train, y_test)
        """
        logger.info("Starting preprocessing pipeline...")
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Encode categorical variables
        df = self.encode_categorical(df)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(df)
        
        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        logger.info("Preprocessing complete!")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def save_preprocessor(self, path: str = "models/preprocessor.pkl"):
        """
        Save the preprocessor (scaler) for later use
        
        Args:
            path: Path to save the preprocessor
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.scaler, path)
        logger.info(f"Preprocessor saved to {path}")
    
    def load_preprocessor(self, path: str = "models/preprocessor.pkl"):
        """
        Load a saved preprocessor
        
        Args:
            path: Path to the saved preprocessor
        """
        self.scaler = joblib.load(path)
        logger.info(f"Preprocessor loaded from {path}")


def main():
    """Main function to test preprocessing"""
    from ingestion import DataIngestion
    
    # Load data
    ingestion = DataIngestion()
    df = ingestion.ingest()
    
    # Preprocess data
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.preprocess(df)
    
    print("\n" + "="*50)
    print("PREPROCESSING COMPLETE")
    print("="*50)
    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"\nFeature names: {X_train.columns.tolist()}")
    
    # Save preprocessor
    preprocessor.save_preprocessor()


if __name__ == "__main__":
    main()