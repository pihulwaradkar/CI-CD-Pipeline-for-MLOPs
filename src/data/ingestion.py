"""
Data Ingestion Module
Handles downloading and loading data from various sources
"""

import os
import logging
import pandas as pd
import requests
from pathlib import Path
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataIngestion:
    """Class to handle data ingestion from various sources"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize DataIngestion
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.raw_data_path = self.config['data']['raw_data_path']
        self.download_url = self.config['data']['download_url']
        
    def download_data(self, url: str = None, save_path: str = None) -> str:
        """
        Download data from URL
        
        Args:
            url: URL to download data from
            save_path: Path to save downloaded data
            
        Returns:
            Path to saved file
        """
        url = url or self.download_url
        save_path = save_path or self.raw_data_path
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Check if file already exists
        if os.path.exists(save_path):
            logger.info(f"File already exists at {save_path}")
            return save_path
        
        try:
            logger.info(f"Downloading data from {url}")
            response = requests.get(url)
            response.raise_for_status()
            
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Data downloaded successfully to {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"Error downloading data: {str(e)}")
            raise
    
    def load_csv(self, file_path: str = None) -> pd.DataFrame:
        """
        Load data from CSV file
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Pandas DataFrame
        """
        file_path = file_path or self.raw_data_path
        
        try:
            logger.info(f"Loading data from {file_path}")
            df = pd.read_csv(file_path)
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def get_data_info(self, df: pd.DataFrame) -> dict:
        """
        Get basic information about the dataset
        
        Args:
            df: Pandas DataFrame
            
        Returns:
            Dictionary with dataset information
        """
        info = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
            'duplicates': df.duplicated().sum()
        }
        
        return info
    
    def ingest(self) -> pd.DataFrame:
        """
        Complete data ingestion pipeline
        
        Returns:
            Pandas DataFrame with loaded data
        """
        # Download data if not exists
        if not os.path.exists(self.raw_data_path):
            self.download_data()
        
        # Load data
        df = self.load_csv()
        
        # Get and log data info
        info = self.get_data_info(df)
        logger.info(f"Dataset Info: {info}")
        
        return df


def main():
    """Main function to test data ingestion"""
    ingestion = DataIngestion()
    df = ingestion.ingest()
    
    print("\n" + "="*50)
    print("DATA INGESTION COMPLETE")
    print("="*50)
    print(f"\nDataset Shape: {df.shape}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nData Types:")
    print(df.dtypes)
    print(f"\nMissing Values:")
    print(df.isnull().sum())


if __name__ == "__main__":
    main()