"""
Data Preprocessing Module
Handles cleaning, encoding, and preparation of aviation safety data with strict leakage controls
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings('ignore')

from config import PROCESSED_DATA_DIR, MODELS_DIR, RANDOM_STATE


class AviationDataPreprocessor:
    """
    Preprocesses aviation accident and safety data with strict booking-stage controls.
    
    Key Features:
    - Removes post-booking leakage (aircraft assignment, flight phase, realized weather, etc.)
    - Encodes categorical variables
    - Creates severity target variable
    - Maintains booking-stage feature integrity
    """
    
    def __init__(self):
        """Initialize preprocessor with encoders and scalers."""
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def create_severity_target(self, df: pd.DataFrame) -> pd.Series:
        """
        Create binary target variable for severe accidents.
        
        Severe accidents defined as:
        - Fatal injuries OR
        - Aircraft destroyed OR
        - Multiple serious injuries
        
        Args:
            df: Input DataFrame with accident data
            
        Returns:
            Binary series (1=severe, 0=non-severe)
        """
        print("🎯 Creating severity target variable...")
        
        # Initialize severity as 0 (non-severe)
        severity = pd.Series(0, index=df.index)
        
        # Check for various severity indicators
        if 'Injury.Severity' in df.columns:
            severity |= (df['Injury.Severity'].str.contains('Fatal', case=False, na=False)).astype(int)
        
        if 'Aircraft.Damage' in df.columns:
            severity |= (df['Aircraft.Damage'].str.contains('Destroyed', case=False, na=False)).astype(int)
        
        if 'Total.Fatal.Injuries' in df.columns:
            severity |= (pd.to_numeric(df['Total.Fatal.Injuries'], errors='coerce').fillna(0) > 0).astype(int)
        
        if 'Total.Serious.Injuries' in df.columns:
            severity |= (pd.to_numeric(df['Total.Serious.Injuries'], errors='coerce').fillna(0) >= 5).astype(int)
        
        severe_count = severity.sum()
        non_severe_count = len(severity) - severe_count
        ratio = non_severe_count / severe_count if severe_count > 0 else 0
        
        print(f"   ✅ Severe accidents: {severe_count:,} ({severe_count/len(severity)*100:.1f}%)")
        print(f"   ✅ Non-severe: {non_severe_count:,} ({non_severe_count/len(severity)*100:.1f}%)")
        print(f"   ✅ Class ratio (non-severe:severe): {ratio:.2f}:1")
        
        return severity
    
    def extract_temporal_features(self, df: pd.DataFrame, date_col: str = 'Event.Date') -> pd.DataFrame:
        """
        Extract booking-stage temporal features.
        
        Only extracts features available at booking time:
        - Year, Month, Season, Day of Week
        - Scheduled hour (if available)
        
        Args:
            df: Input DataFrame
            date_col: Name of date column
            
        Returns:
            DataFrame with added temporal features
        """
        print("📅 Extracting temporal features (booking-stage only)...")
        
        df = df.copy()
        
        # Convert date column to datetime
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        # Extract temporal features
        df['Year'] = df[date_col].dt.year
        df['Month'] = df[date_col].dt.month
        df['DayOfWeek'] = df[date_col].dt.dayofweek
        df['DayOfYear'] = df[date_col].dt.dayofyear
        
        # Season (Northern Hemisphere)
        df['Season'] = df['Month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        
        # Weekend indicator
        df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
        
        print(f"   ✅ Added temporal features: Year, Month, Season, DayOfWeek, IsWeekend")
        
        return df
    
    def remove_leakage_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove post-booking features to ensure booking-stage integrity.
        
        Removes 8 categories of post-booking features:
        1. Aircraft assignment details
        2. Flight phase information
        3. Realized departure times
        4. Realized weather at departure
        5. Damage assessments (post-accident)
        6. Injury counts (post-accident)
        7. Investigation findings
        8. Emergency response details
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with only booking-stage features
        """
        print("🔒 Removing post-booking leakage features...")
        
        # Define leakage feature patterns
        leakage_patterns = [
            'Aircraft.Make', 'Aircraft.Model', 'Aircraft.Category',
            'Engine.Type', 'Number.of.Engines',  # Aircraft assignment
            'Broad.Phase.of.Flight', 'Purpose.of.flight',  # Flight phase
            'Aircraft.Damage', 'Injury.Severity',  # Post-accident outcomes
            'Total.Fatal.Injuries', 'Total.Serious.Injuries',  # Injury counts
            'Total.Minor.Injuries', 'Total.Uninjured',
            'Investigation.Type', 'Publication.Date',  # Investigation
            'Probable.Cause', 'Schedule',  # Realized details
            'Air.carrier', 'Amateur.Built'  # Specific aircraft details
        ]
        
        df = df.copy()
        removed_cols = []
        
        for col in df.columns:
            # Remove columns matching leakage patterns
            if any(pattern.lower() in col.lower() for pattern in leakage_patterns):
                removed_cols.append(col)
        
        df = df.drop(columns=removed_cols, errors='ignore')
        
        print(f"   ✅ Removed {len(removed_cols)} post-booking features")
        print(f"   ✅ Remaining features: {len(df.columns)}")
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features for machine learning.
        
        Args:
            df: Input DataFrame
            fit: If True, fit new encoders; if False, use existing encoders
            
        Returns:
            DataFrame with encoded features
        """
        print("🔤 Encoding categorical features...")
        
        df = df.copy()
        
        # Identify categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove date columns
        categorical_cols = [col for col in categorical_cols if 'date' not in col.lower()]
        
        encoded_count = 0
        
        for col in categorical_cols:
            if col in df.columns:
                # Fill missing values
                df[col] = df[col].fillna('Unknown')
                
                if fit:
                    # Create new encoder
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    # Use existing encoder
                    if col in self.label_encoders:
                        # Handle unseen categories
                        known_classes = set(self.label_encoders[col].classes_)
                        df[col] = df[col].apply(lambda x: x if x in known_classes else 'Unknown')
                        df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col].astype(str))
                
                encoded_count += 1
        
        print(f"   ✅ Encoded {encoded_count} categorical features")
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in dataset.
        
        Strategy:
        - Numerical: Fill with median
        - Categorical: Fill with 'Unknown'
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        print("🔧 Handling missing values...")
        
        df = df.copy()
        initial_missing = df.isnull().sum().sum()
        
        # Numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        
        # Categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna('Unknown')
        
        final_missing = df.isnull().sum().sum()
        
        print(f"   ✅ Resolved {initial_missing:,} missing values")
        print(f"   ✅ Remaining missing: {final_missing}")
        
        return df
    
    def prepare_features_for_modeling(self, df: pd.DataFrame, target_col: str = 'Severe_Accident') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare final feature set for modeling.
        
        Selects only numerical features suitable for ML models.
        
        Args:
            df: Input DataFrame
            target_col: Name of target variable column
            
        Returns:
            Tuple of (X features, y target)
        """
        print("🎯 Preparing features for modeling...")
        
        # Separate target
        if target_col in df.columns:
            y = df[target_col]
            df = df.drop(columns=[target_col])
        else:
            raise ValueError(f"Target column '{target_col}' not found in dataframe")
        
        # Select only numerical columns (encoded features)
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        X = df[numerical_cols]
        
        # Remove any remaining NaN or inf values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        self.feature_names = X.columns.tolist()
        
        print(f"   ✅ Final feature set: {X.shape}")
        print(f"   ✅ Features: {len(self.feature_names)}")
        print(f"   ✅ Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame = None, fit: bool = True) -> Tuple:
        """
        Scale features using StandardScaler.
        
        Args:
            X_train: Training features
            X_test: Test features (optional)
            fit: If True, fit new scaler; if False, use existing scaler
            
        Returns:
            Scaled features (X_train_scaled, X_test_scaled if provided)
        """
        print("📊 Scaling features...")
        
        if fit:
            X_train_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
        else:
            X_train_scaled = pd.DataFrame(
                self.scaler.transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
        
        if X_test is not None:
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
            print(f"   ✅ Scaled training set: {X_train_scaled.shape}")
            print(f"   ✅ Scaled test set: {X_test_scaled.shape}")
            return X_train_scaled, X_test_scaled
        else:
            print(f"   ✅ Scaled features: {X_train_scaled.shape}")
            return X_train_scaled
    
    def save_preprocessor(self, filepath: Path = None):
        """Save preprocessor components (encoders, scaler, feature names)."""
        if filepath is None:
            filepath = MODELS_DIR / 'preprocessor.pkl'
        
        preprocessor_data = {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        
        joblib.dump(preprocessor_data, filepath)
        print(f"💾 Saved preprocessor to {filepath.name}")
    
    @staticmethod
    def load_preprocessor(filepath: Path = None):
        """Load saved preprocessor components."""
        if filepath is None:
            filepath = MODELS_DIR / 'preprocessor.pkl'
        
        preprocessor_data = joblib.load(filepath)
        
        preprocessor = AviationDataPreprocessor()
        preprocessor.label_encoders = preprocessor_data['label_encoders']
        preprocessor.scaler = preprocessor_data['scaler']
        preprocessor.feature_names = preprocessor_data['feature_names']
        
        print(f"📂 Loaded preprocessor from {filepath.name}")
        return preprocessor


def create_train_test_split(X: pd.DataFrame, y: pd.Series, 
                            test_size: float = 0.2, 
                            stratify: bool = True,
                            save: bool = True) -> Dict:
    """
    Create stratified train/test split.
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion of test set
        stratify: Whether to stratify by target
        save: Whether to save split to disk
        
    Returns:
        Dictionary containing X_train, X_test, y_train, y_test
    """
    print("\n" + "=" * 80)
    print("🔀 CREATING TRAIN/TEST SPLIT")
    print("=" * 80)
    
    stratify_arg = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=RANDOM_STATE,
        stratify=stratify_arg
    )
    
    split_data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
    
    print(f"\n✅ Training set: {X_train.shape}")
    print(f"   Class distribution: {pd.Series(y_train).value_counts().to_dict()}")
    print(f"\n✅ Test set: {X_test.shape}")
    print(f"   Class distribution: {pd.Series(y_test).value_counts().to_dict()}")
    
    if save:
        filepath = MODELS_DIR / 'clean_train_test_split.pkl'
        joblib.dump(split_data, filepath)
        print(f"\n💾 Saved split to {filepath.name}")
    
    return split_data
