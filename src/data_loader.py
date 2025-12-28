"""
Data Loader Module
Handles loading and initial processing of aviation safety datasets
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR


class DataLoader:
    """
    Loads and performs initial processing of aviation accident and safety data.
    
    Attributes:
        raw_data_dir (Path): Directory containing raw data files
        processed_data_dir (Path): Directory for saving processed data
    """
    
    def __init__(self, raw_data_dir: Path = RAW_DATA_DIR, 
                 processed_data_dir: Path = PROCESSED_DATA_DIR):
        """
        Initialize DataLoader with data directories.
        
        Args:
            raw_data_dir: Path to raw data directory
            processed_data_dir: Path to processed data directory
        """
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.accidents_df = None
        self.safety_df = None
        
    def load_aviation_data(self, filename: str = 'AviationData.csv') -> pd.DataFrame:
        """
        Load aviation accident data from CSV file.
        
        Args:
            filename: Name of the aviation data file
            
        Returns:
            DataFrame containing aviation accident data
            
        Raises:
            FileNotFoundError: If data file doesn't exist
        """
        filepath = self.raw_data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Aviation data file not found: {filepath}")
        
        print(f"📂 Loading aviation accident data from {filename}...")
        df = pd.read_csv(filepath, encoding='latin-1', low_memory=False)
        print(f"   ✅ Loaded {len(df):,} accident records")
        print(f"   ✅ Columns: {len(df.columns)} features")
        
        self.accidents_df = df
        return df
    
    def load_airline_safety_data(self, filename: str = 'airline-safety.csv') -> pd.DataFrame:
        """
        Load airline safety rankings data from CSV file.
        
        Args:
            filename: Name of the airline safety file
            
        Returns:
            DataFrame containing airline safety data
            
        Raises:
            FileNotFoundError: If data file doesn't exist
        """
        filepath = self.raw_data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Airline safety file not found: {filepath}")
        
        print(f"📂 Loading airline safety data from {filename}...")
        df = pd.read_csv(filepath)
        print(f"   ✅ Loaded {len(df):,} airline records")
        print(f"   ✅ Columns: {len(df.columns)} features")
        
        self.safety_df = df
        return df
    
    def load_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load both aviation accident and airline safety data.
        
        Returns:
            Tuple of (accidents_df, safety_df)
        """
        print("=" * 80)
        print("📊 LOADING ALL DATASETS")
        print("=" * 80)
        
        accidents_df = self.load_aviation_data()
        safety_df = self.load_airline_safety_data()
        
        print("\n✅ All datasets loaded successfully!")
        return accidents_df, safety_df
    
    def get_data_summary(self) -> Dict:
        """
        Get summary statistics for loaded datasets.
        
        Returns:
            Dictionary containing summary information
        """
        summary = {}
        
        if self.accidents_df is not None:
            summary['accidents'] = {
                'n_records': len(self.accidents_df),
                'n_features': len(self.accidents_df.columns),
                'missing_pct': (self.accidents_df.isnull().sum().sum() / 
                               (len(self.accidents_df) * len(self.accidents_df.columns)) * 100),
                'memory_mb': self.accidents_df.memory_usage(deep=True).sum() / 1024**2
            }
        
        if self.safety_df is not None:
            summary['safety'] = {
                'n_records': len(self.safety_df),
                'n_features': len(self.safety_df.columns),
                'missing_pct': (self.safety_df.isnull().sum().sum() / 
                               (len(self.safety_df) * len(self.safety_df.columns)) * 100),
                'memory_mb': self.safety_df.memory_usage(deep=True).sum() / 1024**2
            }
        
        return summary
    
    def print_data_info(self):
        """Print detailed information about loaded datasets."""
        print("\n" + "=" * 80)
        print("📋 DATASET INFORMATION")
        print("=" * 80)
        
        summary = self.get_data_summary()
        
        for dataset_name, stats in summary.items():
            print(f"\n{dataset_name.upper()} DATASET:")
            print(f"   Records: {stats['n_records']:,}")
            print(f"   Features: {stats['n_features']}")
            print(f"   Missing data: {stats['missing_pct']:.2f}%")
            print(f"   Memory: {stats['memory_mb']:.2f} MB")
    
    def save_processed_data(self, df: pd.DataFrame, filename: str):
        """
        Save processed data to CSV file.
        
        Args:
            df: DataFrame to save
            filename: Output filename
        """
        filepath = self.processed_data_dir / filename
        df.to_csv(filepath, index=False)
        print(f"💾 Saved processed data to {filename}")


def load_processed_dataset(filename: str = 'final_dataset.csv') -> pd.DataFrame:
    """
    Load pre-processed dataset from processed data directory.
    
    Args:
        filename: Name of the processed data file
        
    Returns:
        DataFrame containing processed data
        
    Raises:
        FileNotFoundError: If processed file doesn't exist
    """
    filepath = PROCESSED_DATA_DIR / filename
    
    if not filepath.exists():
        raise FileNotFoundError(
            f"Processed data file not found: {filepath}\n"
            f"Please run data preprocessing first."
        )
    
    print(f"📂 Loading processed dataset from {filename}...")
    df = pd.read_csv(filepath)
    print(f"   ✅ Loaded {len(df):,} records with {len(df.columns)} features")
    
    return df


def load_train_test_split(split_file: str = 'clean_train_test_split.pkl'):
    """
    Load pre-split train/test data from pickle file.
    
    Args:
        split_file: Name of the pickle file containing split data
        
    Returns:
        Dictionary containing X_train, X_test, y_train, y_test
        
    Raises:
        FileNotFoundError: If split file doesn't exist
    """
    import joblib
    from config import MODELS_DIR
    
    filepath = MODELS_DIR / split_file
    
    if not filepath.exists():
        raise FileNotFoundError(
            f"Train/test split file not found: {filepath}\n"
            f"Please run data preprocessing and splitting first."
        )
    
    print(f"📂 Loading train/test split from {split_file}...")
    split_data = joblib.load(filepath)
    
    print(f"   ✅ Training set: {split_data['X_train'].shape}")
    print(f"   ✅ Test set: {split_data['X_test'].shape}")
    print(f"   ✅ Features: {list(split_data['X_train'].columns)[:5]}...")
    
    return split_data


if __name__ == "__main__":
    # Example usage
    loader = DataLoader()
    
    # Load all data
    accidents_df, safety_df = loader.load_all_data()
    
    # Print information
    loader.print_data_info()
    
    # Display first few rows
    print("\n" + "=" * 80)
    print("SAMPLE DATA")
    print("=" * 80)
    print("\nAccidents Data (first 3 rows):")
    print(accidents_df.head(3))
    
    print("\nSafety Data (first 3 rows):")
    print(safety_df.head(3))
