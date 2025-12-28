# # src/feature_engineering.py
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler, PolynomialFeatures
# from sklearn.feature_selection import SelectKBest, f_classif
# import warnings
# warnings.filterwarnings('ignore')

# class FeatureEngineer:
#     def __init__(self):
#         self.scalers = {}
#         self.feature_selector = None
        
#     def create_temporal_features(self, df):
#         """Create advanced temporal features"""
#         print("⏰ Creating temporal features...")
        
#         if 'Event.Date' in df.columns:
#             df['Event.Date'] = pd.to_datetime(df['Event.Date'])
            
#             # Seasonal features
#             df['Season'] = (df['Event.Date'].dt.month % 12 + 3) // 3
#             df['Is_Weekend'] = df['Event.Date'].dt.dayofweek >= 5
#             df['Is_Holiday_Season'] = df['Event.Date'].dt.month.isin([11, 12])
            
#             # Time-based features
#             df['Year_Sin'] = np.sin(2 * np.pi * df['Event.Date'].dt.dayofyear/365)
#             df['Year_Cos'] = np.cos(2 * np.pi * df['Event.Date'].dt.dayofyear/365)
        
#         return df
    
#     def create_risk_composite_features(self, df):
#         """Create composite risk features"""
#         print("🎯 Creating risk composite features...")
        
#         # Weather risk score
#         weather_risk_map = {
#             'VMC': 1, 'Unknown': 2, 'IMC': 4, 
#             'Rain': 3, 'Snow': 5, 'Fog': 3
#         }
#         df['Weather_Risk_Score'] = df['Weather.Condition'].map(
#             lambda x: weather_risk_map.get(x, 2)
#         )
        
#         # Phase of flight risk
#         flight_phase_risk = {
#             'Takeoff': 4, 'Landing': 4, 'Climb': 3, 
#             'Descent': 3, 'Cruise': 2, 'Taxi': 1
#         }
#         df['Flight_Phase_Risk'] = df['Broad.Phase.of.Flight'].map(
#             lambda x: flight_phase_risk.get(x, 2)
#         )
        
#         # Aircraft category risk
#         aircraft_risk = {
#             'Airplane': 2, 'Helicopter': 3, 'Glider': 4,
#             'Balloon': 4, 'Unknown': 2
#         }
#         df['Aircraft_Risk'] = df['Aircraft.Category'].map(
#             lambda x: aircraft_risk.get(x, 2)
#         )
        
#         # Composite risk score
#         df['Composite_Risk_Score'] = (
#             df['Weather_Risk_Score'] * 0.3 +
#             df['Flight_Phase_Risk'] * 0.4 +
#             df['Aircraft_Risk'] * 0.3
#         )
        
#         return df
    
#     def create_interaction_features(self, df):
#         """Create interaction features"""
#         print("🔄 Creating interaction features...")
        
#         # Weather + Phase interaction
#         df['Weather_Phase_Risk'] = df['Weather_Risk_Score'] * df['Flight_Phase_Risk']
        
#         # Safety score interactions
#         if 'Safety_Score_Normalized' in df.columns:
#             df['Safety_Weather_Risk'] = df['Safety_Score_Normalized'] * df['Weather_Risk_Score']
#             df['Safety_Phase_Risk'] = df['Safety_Score_Normalized'] * df['Flight_Phase_Risk']
        
#         return df
    
#     def scale_features(self, X_train, X_test):
#         """Scale features using StandardScaler"""
#         print("⚖️ Scaling features...")
        
#         self.scalers['standard'] = StandardScaler()
#         X_train_scaled = self.scalers['standard'].fit_transform(X_train)
#         X_test_scaled = self.scalers['standard'].transform(X_test)
        
#         return X_train_scaled, X_test_scaled
    
#     def select_features(self, X, y, k=20):
#         """Select top k features using ANOVA F-test"""
#         print(f"🔍 Selecting top {k} features...")
        
#         self.feature_selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
#         X_selected = self.feature_selector.fit_transform(X, y)
        
#         feature_scores = pd.DataFrame({
#             'feature': range(X.shape[1]),
#             'score': self.feature_selector.scores_,
#             'p_value': self.feature_selector.pvalues_
#         })
        
#         print("📊 Top 10 features by ANOVA F-score:")
#         print(feature_scores.nlargest(10, 'score')[['feature', 'score', 'p_value']])
        
#         return X_selected
    
#     def engineer_all_features(self, df, X_train, X_test, y_train, feature_names):
#         """Complete feature engineering pipeline"""
#         print("🚀 Starting complete feature engineering...")
        
#         # Create new features
#         df_enhanced = self.create_temporal_features(df)
#         df_enhanced = self.create_risk_composite_features(df_enhanced)
#         df_enhanced = self.create_interaction_features(df_enhanced)
        
#         # Update feature set with new features
#         new_features = [
#             'Season', 'Is_Weekend', 'Is_Holiday_Season',
#             'Year_Sin', 'Year_Cos', 'Weather_Risk_Score',
#             'Flight_Phase_Risk', 'Aircraft_Risk', 'Composite_Risk_Score',
#             'Weather_Phase_Risk', 'Safety_Weather_Risk', 'Safety_Phase_Risk'
#         ]
        
#         # Add new features that exist in dataframe
#         existing_new_features = [f for f in new_features if f in df_enhanced.columns]
        
#         # Combine original and new features
#         X_train_enhanced = X_train.copy()
#         X_test_enhanced = X_test.copy()
        
#         for feature in existing_new_features:
#             # Get feature values for train and test indices
#             train_feature = df_enhanced.iloc[X_train.index][feature].values
#             test_feature = df_enhanced.iloc[X_test.index][feature].values
            
#             X_train_enhanced[feature] = train_feature
#             X_test_enhanced[feature] = test_feature
        
#         # Update feature names
#         enhanced_feature_names = list(feature_names) + existing_new_features
        
#         print(f"✅ Feature engineering completed!")
#         print(f"   Original features: {len(feature_names)}")
#         print(f"   New features: {len(existing_new_features)}")
#         print(f"   Total features: {len(enhanced_feature_names)}")
        
#         return X_train_enhanced, X_test_enhanced, enhanced_feature_names

# def main():
#     """Main feature engineering pipeline"""
#     from preprocessing import main as preprocess_main
    
#     # Get preprocessed data
#     X_train, X_test, y_train, y_test, feature_names = preprocess_main()
    
#     # Load full dataset for feature engineering
#     df = pd.read_csv('../data/processed/final_dataset.csv')
    
#     # Perform feature engineering
#     engineer = FeatureEngineer()
#     X_train_enhanced, X_test_enhanced, enhanced_features = engineer.engineer_all_features(
#         df, X_train, X_test, y_train, feature_names
#     )
    
#     # Scale features
#     X_train_scaled, X_test_scaled = engineer.scale_features(X_train_enhanced, X_test_enhanced)
    
#     # Feature selection
#     X_train_selected = engineer.select_features(X_train_scaled, y_train)
#     X_test_selected = engineer.feature_selector.transform(X_test_scaled)
    
#     print(f"🎉 Final feature set: {X_train_selected.shape[1]} features")
    
#     return X_train_selected, X_test_selected, y_train, y_test, enhanced_features

# if __name__ == "__main__":
#     X_train, X_test, y_train, y_test, features = main()


# src/feature_engineering.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer:
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_selector = None
        
    def create_advanced_temporal_features(self, df):
        """Create sophisticated temporal patterns"""
        print("⏰ Creating advanced temporal features...")
        
        if 'Event.Date' in df.columns:
            df['Event.Date'] = pd.to_datetime(df['Event.Date'])
            
            # Extract hour if available
            if 'Event.Time' in df.columns:
                df['Hour'] = pd.to_datetime(df['Event.Time'], errors='coerce').dt.hour
                df['Hour'] = df['Hour'].fillna(12)  # Default to noon if missing

            # Ensure Hour column exists for downstream logic
            if 'Hour' not in df.columns:
                df['Hour'] = 12
            
            # Advanced temporal features
            df['DayOfYear'] = df['Event.Date'].dt.dayofyear
            df['WeekOfYear'] = df['Event.Date'].dt.isocalendar().week
            df['Quarter'] = df['Event.Date'].dt.quarter
            
            # Cyclical features for seasonality
            df['Year_Sin'] = np.sin(2 * np.pi * df['DayOfYear']/365)
            df['Year_Cos'] = np.cos(2 * np.pi * df['DayOfYear']/365)
            df['Month_Sin'] = np.sin(2 * np.pi * df['Month']/12)
            df['Month_Cos'] = np.cos(2 * np.pi * df['Month']/12)
            
            # Operational features
            df['Is_Rush_Hour'] = ((df['Hour'] >= 7) & (df['Hour'] <= 9)) | ((df['Hour'] >= 16) & (df['Hour'] <= 19))
            df['Is_Night_Flight'] = (df['Hour'] >= 22) | (df['Hour'] <= 6)
            df['Is_Holiday_Season'] = df['Month'].isin([11, 12]).astype(int)
            df['Is_Summer'] = df['Month'].isin([6, 7, 8]).astype(int)
            
            # Seasonal risk multipliers
            df['Seasonal_Risk_Multiplier'] = np.where(df['Month'].isin([12, 1, 2]), 1.2, 
                                                     np.where(df['Month'].isin([6, 7, 8]), 1.1, 1.0))
        
        return df
    
    def create_aircraft_age_features(self, df):
        """Create aircraft age-related features"""
        print("🛩️ Creating aircraft age features...")
        
        # Create aircraft age proxy
        if 'Year' in df.columns:
            current_year = 2024
            df['Aircraft_Age_Proxy'] = current_year - df['Year']
            
            # Age buckets
            df['Aircraft_Age_Group'] = pd.cut(df['Aircraft_Age_Proxy'], 
                                             bins=[0, 5, 10, 15, 20, 30, 100],
                                             labels=['New', 'Young', 'Mid', 'Mature', 'Old', 'Very_Old'])
            
            # Age risk score (older aircraft = higher risk)
            age_risk_map = {
                'New': 1.0, 'Young': 1.1, 'Mid': 1.3, 
                'Mature': 1.6, 'Old': 2.0, 'Very_Old': 2.5
            }
            df['Aircraft_Age_Risk'] = df['Aircraft_Age_Group'].map(age_risk_map)
        
        return df
    
    def create_risk_composite_features(self, df):
        """Create enhanced composite risk scores"""
        print("🎯 Creating enhanced risk composite features...")
        
        # Weather risk mapping (higher = more risky)
        weather_risk_map = {
            'VMC': 1, 'Unknown': 2, 'UNK': 2,
            'IMC': 4, 'Rain': 3, 'Snow': 5, 
            'Fog': 3, 'Storm': 6, 'Thunderstorm': 6,
            'Crosswind': 4, 'Turbulence': 4
        }
        df['Weather_Risk_Score'] = df['Weather.Condition'].map(
            lambda x: weather_risk_map.get(x, 2)
        ).fillna(2)
        
        # Flight phase risk mapping
        flight_phase_risk = {
            'Takeoff': 4, 'Landing': 4, 'Climb': 3, 
            'Descent': 3, 'Cruise': 2, 'Taxi': 1,
            'Approach': 4, 'Standing': 1, 'Maneuvering': 3
        }
        df['Flight_Phase_Risk'] = df['Broad.Phase.of.Flight'].map(
            lambda x: flight_phase_risk.get(x, 2)
        ).fillna(2)
        
        # Aircraft category risk
        aircraft_risk = {
            'Airplane': 2, 'Helicopter': 3, 'Glider': 4,
            'Balloon': 4, 'Unknown': 2, 'Amphibian': 3,
            'Gyroplane': 3, 'Blimp': 2
        }
        df['Aircraft_Risk'] = df['Aircraft.Category'].map(
            lambda x: aircraft_risk.get(x, 2)
        ).fillna(2)
        
        # Enhanced composite risk score with interactions
        df['Composite_Risk_Score'] = (
            df['Weather_Risk_Score'] * 0.25 +
            df['Flight_Phase_Risk'] * 0.30 +
            df['Aircraft_Risk'] * 0.20 +
            # Ensure numeric multiplication even if previous steps produced categorical dtypes
            pd.to_numeric(df.get('Aircraft_Age_Risk', 1.0), errors='coerce').fillna(1.0) * 0.15 +
            pd.to_numeric(df.get('Seasonal_Risk_Multiplier', 1.0), errors='coerce').fillna(1.0) * 0.10
        )
        
        return df
    
    def create_operational_features(self, df):
        """Create advanced operational risk features"""
        print("🏢 Creating advanced operational features...")
        
        # Total people involved
        injury_cols = ['Total.Fatal.Injuries', 'Total.Serious.Injuries', 
                      'Total.Minor.Injuries', 'Total.Uninjured']
        
        for col in injury_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        if all(col in df.columns for col in injury_cols):
            df['Total_People_Involved'] = df[injury_cols].sum(axis=1)
            df['Fatality_Rate'] = df['Total.Fatal.Injuries'] / df['Total_People_Involved']
            df['Fatality_Rate'] = df['Fatality_Rate'].replace([np.inf, -np.inf], 0).fillna(0)
            
            # Severity index
            df['Severity_Index'] = (
                df['Total.Fatal.Injuries'] * 3 +
                df['Total.Serious.Injuries'] * 2 +
                df['Total.Minor.Injuries'] * 1
            ) / (df['Total_People_Involved'] + 1)  # +1 to avoid division by zero
        
        return df
    
    def create_interaction_features(self, df):
        """Create enhanced interaction features"""
        print("🔄 Creating enhanced interaction features...")
        
        # Risk score interactions
        df['Weather_Phase_Interaction'] = df['Weather_Risk_Score'] * df['Flight_Phase_Risk']
        df['Weather_Aircraft_Interaction'] = df['Weather_Risk_Score'] * df['Aircraft_Risk']
        df['Phase_Age_Interaction'] = df['Flight_Phase_Risk'] * pd.to_numeric(
            df.get('Aircraft_Age_Risk', 1.0), errors='coerce').fillna(1.0)
        
        # Temporal risk interactions
        if 'Month' in df.columns:
            df['Month_Risk_Interaction'] = df['Month'] * df['Composite_Risk_Score']
        
        # Safety score interactions (if available)
        if 'Safety_Score_Normalized' in df.columns:
            df['Safety_Weather_Interaction'] = df['Safety_Score_Normalized'] * df['Weather_Risk_Score']
            df['Safety_Phase_Interaction'] = df['Safety_Score_Normalized'] * df['Flight_Phase_Risk']
            df['Safety_Age_Interaction'] = df['Safety_Score_Normalized'] * pd.to_numeric(
                df.get('Aircraft_Age_Risk', 1.0), errors='coerce').fillna(1.0)
        
        # Operational context interactions
        if 'Is_Rush_Hour' in df.columns:
            df['Rush_Hour_Risk'] = df['Is_Rush_Hour'] * df['Composite_Risk_Score']
        if 'Is_Night_Flight' in df.columns:
            df['Night_Flight_Risk'] = df['Is_Night_Flight'] * df['Composite_Risk_Score']
        
        return df
    
    def encode_categorical_features(self, df, categorical_columns):
        """Encode categorical variables with advanced handling"""
        print("🔤 Encoding categorical features...")
        
        for col in categorical_columns:
            if col in df.columns:
                # Handle missing values
                df[col] = df[col].fillna('Unknown')
                
                # Create frequency encoding as additional feature
                freq_encoding = df[col].value_counts().to_dict()
                df[f'{col}_freq'] = df[col].map(freq_encoding)
                
                # Create label encoding
                self.encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.encoders[col].fit_transform(df[col])
        
        return df
    
    def engineer_all_features(self, df, X_train, X_test, y_train, feature_names):
        """Complete enhanced feature engineering pipeline"""
        print("🚀 Starting enhanced feature engineering...")
        
        # Create new features
        df_enhanced = self.create_advanced_temporal_features(df)
        df_enhanced = self.create_aircraft_age_features(df_enhanced)
        df_enhanced = self.create_risk_composite_features(df_enhanced)
        df_enhanced = self.create_operational_features(df_enhanced)
        
        # Encode categorical features
        categorical_columns = [
            'Injury.Severity', 'Aircraft.Damage', 'Aircraft.Category',
            'Engine.Type', 'Weather.Condition', 'Broad.Phase.of.Flight'
        ]
        df_enhanced = self.encode_categorical_features(df_enhanced, categorical_columns)
        
        # Interaction features
        df_enhanced = self.create_interaction_features(df_enhanced)
        
        # Update feature set with new features
        new_features = [
            # Temporal features
            'DayOfYear', 'WeekOfYear', 'Quarter', 'Hour',
            'Year_Sin', 'Year_Cos', 'Month_Sin', 'Month_Cos',
            'Is_Rush_Hour', 'Is_Night_Flight', 'Is_Holiday_Season', 'Is_Summer',
            'Seasonal_Risk_Multiplier',
            
            # Aircraft age features
            'Aircraft_Age_Proxy', 'Aircraft_Age_Risk',
            
            # Risk scores
            'Weather_Risk_Score', 'Flight_Phase_Risk', 'Aircraft_Risk',
            'Composite_Risk_Score',
            
            # Operational features
            'Total_People_Involved', 'Fatality_Rate', 'Severity_Index',
            
            # Interaction features
            'Weather_Phase_Interaction', 'Weather_Aircraft_Interaction',
            'Phase_Age_Interaction', 'Month_Risk_Interaction',
            'Safety_Weather_Interaction', 'Safety_Phase_Interaction', 
            'Safety_Age_Interaction', 'Rush_Hour_Risk', 'Night_Flight_Risk'
        ]
        
        # Add frequency encoded features
        for col in categorical_columns:
            new_features.append(f'{col}_freq')
        
        # Only use columns that exist in dataframe
        existing_new_features = [f for f in new_features if f in df_enhanced.columns]
        
        # Combine original and new features
        X_train_enhanced = X_train.copy()
        X_test_enhanced = X_test.copy()
        
        for feature in existing_new_features:
            # Get feature values for train and test indices
            train_feature = df_enhanced.iloc[X_train.index][feature].values
            test_feature = df_enhanced.iloc[X_test.index][feature].values
            
            X_train_enhanced[feature] = train_feature
            X_test_enhanced[feature] = test_feature
        
        # Update feature names
        enhanced_feature_names = list(feature_names) + existing_new_features
        
        print(f"✅ Enhanced feature engineering completed!")
        print(f"   Original features: {len(feature_names)}")
        print(f"   New features: {len(existing_new_features)}")
        print(f"   Total features: {len(enhanced_feature_names)}")
        
        return X_train_enhanced, X_test_enhanced, enhanced_feature_names

    def scale_features(self, X_train, X_test):
        """Scale features using StandardScaler"""
        print("⚖️ Scaling features...")
        
        self.scalers['standard'] = StandardScaler()
        X_train_scaled = self.scalers['standard'].fit_transform(X_train)
        X_test_scaled = self.scalers['standard'].transform(X_test)
        
        return X_train_scaled, X_test_scaled
    
    def select_features(self, X, y, k=25):
        """Select top k features using ANOVA F-test"""
        print(f"🔍 Selecting top {k} features...")
        
        self.feature_selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
        X_selected = self.feature_selector.fit_transform(X, y)
        
        feature_scores = pd.DataFrame({
            'feature': range(X.shape[1]),
            'score': self.feature_selector.scores_,
            'p_value': self.feature_selector.pvalues_
        })
        
        print("📊 Top 10 features by ANOVA F-score:")
        print(feature_scores.nlargest(10, 'score')[['feature', 'score', 'p_value']])
        
        return X_selected


def main():
    """Main feature engineering entrypoint expected by model_training.

    Returns:
        X_train_selected, X_test_selected, y_train, y_test, selected_feature_names
    """
    import os, sys

    # Ensure project root is on path so sibling modules can be imported
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.append(project_root)

    # Import preprocessing pipeline
    from preprocessing import main as preprocess_main
    from utils import ResearchConfig

    # Run preprocessing to get the base train/test splits and feature names
    X_train, X_test, y_train, y_test, feature_names = preprocess_main()

    # Load the merged full dataset (ensures engineered features can be extracted)
    final_path = os.path.join(project_root, ResearchConfig.PROCESSED_DATA_PATH, 'final_dataset.csv')
    if not os.path.exists(final_path):
        # fallback to relative path
        final_path = os.path.join(project_root, 'data', 'processed', 'final_dataset.csv')

    df = pd.read_csv(final_path)

    # Run the advanced feature engineering
    engineer = AdvancedFeatureEngineer()
    X_train_enhanced, X_test_enhanced, enhanced_features = engineer.engineer_all_features(
        df, X_train, X_test, y_train, feature_names
    )

    # Scale features
    X_train_scaled, X_test_scaled = engineer.scale_features(X_train_enhanced, X_test_enhanced)

    # Feature selection (keep k best)
    X_train_selected = engineer.select_features(X_train_scaled, y_train, k=25)
    X_test_selected = engineer.feature_selector.transform(X_test_scaled)

    # Determine selected feature names
    selected_idx = engineer.feature_selector.get_support(indices=True)
    try:
        selected_feature_names = [enhanced_features[i] for i in selected_idx]
    except Exception:
        # Fallback to enhanced_features if mapping fails
        selected_feature_names = enhanced_features

    print("🎉 Feature engineering pipeline completed.")

    return X_train_selected, X_test_selected, y_train, y_test, selected_feature_names


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, features = main()
