# config.py
"""
Configuration file for Aviation Safety Flight Risk Prediction System
Centralizes all constants, paths, and hyperparameters
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()

# Data directories
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
EXTERNAL_DATA_DIR = DATA_DIR / 'external'

# Model directories
MODELS_DIR = PROJECT_ROOT / 'models'

# Output directories
PAPERS_DIR = PROJECT_ROOT / 'papers'
FIGURES_DIR = PAPERS_DIR / 'figures'

# Notebook directory
NOTEBOOKS_DIR = PROJECT_ROOT / 'notebooks'

# Source code directory
SRC_DIR = PROJECT_ROOT / 'src'

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR, 
                  MODELS_DIR, FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================================
# DATA FILES
# ============================================================================

# Input files
NTSB_ACCIDENTS_FILE = RAW_DATA_DIR / 'ntsb_accidents.csv'
AIRLINE_SAFETY_FILE = RAW_DATA_DIR / 'airline_safety_rankings.csv'

# Processed files
ACCIDENTS_ENHANCED_FILE = PROCESSED_DATA_DIR / 'accidents_enhanced.csv'
SAFETY_ENHANCED_FILE = PROCESSED_DATA_DIR / 'safety_enhanced.csv'
FINAL_FEATURES_FILE = PROCESSED_DATA_DIR / 'final_features.csv'
FINAL_FEATURES_CLEAN_FILE = PROCESSED_DATA_DIR / 'final_features_clean.csv'

# Model files
BEST_MODEL_FILE = MODELS_DIR / 'flight_risk_model.pkl'
BASELINE_CLEAN_MODEL_FILE = MODELS_DIR / 'baseline_clean_model.pkl'
FEATURE_SCALER_FILE = MODELS_DIR / 'feature_scaler.pkl'
CLEAN_FEATURE_SCALER_FILE = MODELS_DIR / 'clean_feature_scaler.pkl'
FEATURE_NAMES_FILE = MODELS_DIR / 'feature_names.pkl'
CLEAN_FEATURE_NAMES_FILE = MODELS_DIR / 'clean_feature_names.pkl'


# ============================================================================
# MODEL HYPERPARAMETERS
# ============================================================================

# Random seed for reproducibility
RANDOM_STATE = 42

# Train-test split
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# Cross-validation
CV_FOLDS = 5
CV_STRATEGY = 'stratified'  # 'stratified', 'time-series', 'group'

# Model configurations
MODEL_CONFIGS = {
    'RandomForest': {
        'n_estimators': 200,
        'max_depth': 20,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'class_weight': 'balanced',
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    },
    
    'XGBoost': {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': RANDOM_STATE,
        'eval_metric': 'logloss',
        'use_label_encoder': False
    },
    
    'GradientBoosting': {
        'n_estimators': 200,
        'max_depth': 5,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'random_state': RANDOM_STATE
    },
    
    'LogisticRegression': {
        'max_iter': 1000,
        'class_weight': 'balanced',
        'random_state': RANDOM_STATE,
        'solver': 'lbfgs'
    },
    
    'SVM': {
        'kernel': 'rbf',
        'C': 1.0,
        'class_weight': 'balanced',
        'probability': True,
        'random_state': RANDOM_STATE
    }
}

# Hyperparameter tuning grids
PARAM_GRIDS = {
    'RandomForest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    
    'XGBoost': {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0]
    },
    
    'GradientBoosting': {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2]
    }
}


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

# Features that cause DATA LEAKAGE (post-accident information)
LEAKED_FEATURES = [
    'Total.Fatal.Injuries',
    'Total.Serious.Injuries',
    'Total.Minor.Injuries',
    'Total.Uninjured',
    'Injury.Severity',
    'Injury.Severity_encoded',
    'Aircraft.Damage',
    'Aircraft.Damage_encoded',
    'Aircraft.Damage_freq',
    'Fatality_Rate',
    'Total_People_Involved',
    'Injury_Risk_Score',
]

# Valid pre-flight observable features (organized by category)
VALID_FEATURE_CATEGORIES = {
    'temporal': [
        'Year', 'Month', 'DayOfWeek', 'DayOfYear',
        'Month_Sin', 'Month_Cos', 'Quarter',
        'Is_Weekend', 'Is_Holiday_Season', 'Season'
    ],
    
    'weather': [
        'Weather.Condition',
        'Weather.Condition_encoded',
        'Weather_Risk_Score'
    ],
    
    'flight_planning': [
        'Broad.Phase.of.Flight',
        'Broad.Phase.of.Flight_encoded',
        'Flight_Phase_Risk',
        'Purpose.of.Flight',
        'Purpose.of.Flight_encoded'
    ],
    
    'aircraft': [
        'Aircraft.Category',
        'Aircraft.Category_encoded',
        'Aircraft_Risk',
        'Make',
        'Model',
        'Number.of.Engines',
        'Engine.Type',
        'Engine.Type_encoded',
        'Amateur.Built',
        'Is_Amateur_Built',
        'Aircraft_Age_Proxy',
        'Engine_Count_Category'
    ],
    
    'operator': [
        'Airline_Safety_Rank',
        'Safety_Score',
        'Safety_Score_Normalized',
        'Incident_Rate',
        'Total_Incidents'
    ],
    
    'location': [
        'Location',
        'State',
        'State_Risk_Score'
    ],
    
    'interactions': [
        'Weather_Phase_Interaction',
        'Weather_Aircraft_Interaction',
        'Safety_Weather_Interaction',
        'Age_Category_Interaction',
        'Engine_Phase_Interaction'
    ]
}

# Risk scoring mappings
WEATHER_RISK_MAP = {
    'VMC': 1,      # Visual Meteorological Conditions
    'IMC': 4,      # Instrument Meteorological Conditions
    'Snow': 5,
    'Fog': 5,
    'Rain': 3,
    'Thunderstorm': 6,
    'Unknown': 2
}

FLIGHT_PHASE_RISK_MAP = {
    'Taxi': 1,
    'Takeoff': 4,
    'Initial Climb': 4,
    'Climb': 3,
    'Cruise': 2,
    'Descent': 3,
    'Approach': 4,
    'Landing': 4,
    'Maneuvering': 5,
    'Standing': 1,
    'Unknown': 2
}

AIRCRAFT_CATEGORY_RISK_MAP = {
    'Airplane': 2,
    'Helicopter': 3,
    'Glider': 4,
    'Balloon': 4,
    'Weight-Shift': 4,
    'Powered Parachute': 4,
    'Gyrocraft': 4,
    'Unknown': 2
}


# ============================================================================
# RISK THRESHOLDS
# ============================================================================

# Classification thresholds
LOW_RISK_THRESHOLD = 0.3       # Below this = low risk
MEDIUM_RISK_THRESHOLD = 0.7    # Above this = high risk
# Between LOW and MEDIUM = medium risk

# Optimal decision threshold (to be tuned based on business requirements)
OPTIMAL_THRESHOLD = 0.5


# ============================================================================
# BUSINESS METRICS
# ============================================================================

# Cost parameters (in USD)
COST_FALSE_NEGATIVE = 10_000_000  # Cost of missing a severe accident
COST_FALSE_POSITIVE = 50_000       # Cost of unnecessary prevention measures
BENEFIT_TRUE_POSITIVE = 2_000_000  # Benefit of preventing severe accident
COST_INSPECTION = 5_000            # Cost of enhanced pre-flight check

# Additional business benefits
BENEFIT_REPUTATION = 500_000       # Brand protection value
BENEFIT_INSURANCE_SAVINGS = 300_000  # Reduced insurance premiums


# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================

# Plot style
PLOT_STYLE = 'default'
COLOR_PALETTE = 'husl'

# Figure sizes
FIGURE_SIZE_SMALL = (8, 6)
FIGURE_SIZE_MEDIUM = (12, 8)
FIGURE_SIZE_LARGE = (15, 10)
FIGURE_SIZE_WIDE = (20, 6)

# DPI for saving figures
FIGURE_DPI = 300

# Font sizes
FONT_SIZE_TITLE = 16
FONT_SIZE_LABEL = 14
FONT_SIZE_TICK = 12
FONT_SIZE_LEGEND = 12


# ============================================================================
# PERFORMANCE BENCHMARKS
# ============================================================================

# Minimum performance requirements for deployment
DEPLOYMENT_BENCHMARKS = {
    'accuracy': 0.75,
    'precision': 0.70,
    'recall': 0.75,
    'f1': 0.70,
    'roc_auc': 0.80
}

# Warning thresholds (suspiciously high - check for data leakage)
DATA_LEAKAGE_WARNING_THRESHOLDS = {
    'accuracy': 0.98,
    'precision': 0.98,
    'recall': 0.98,
    'f1': 0.98,
    'roc_auc': 0.99
}


# ============================================================================
# DATA VALIDATION RULES
# ============================================================================

# Valid ranges for features
VALID_RANGES = {
    'Year': (1980, 2025),
    'Month': (1, 12),
    'DayOfWeek': (0, 6),
    'Quarter': (1, 4),
    'Weather_Risk_Score': (1, 6),
    'Flight_Phase_Risk': (1, 5),
    'Aircraft_Risk': (1, 5),
    'Safety_Score': (0, 7),
    'Aircraft_Age_Proxy': (0, 50)
}

# Expected data types
EXPECTED_DTYPES = {
    'Year': 'int64',
    'Month': 'int64',
    'DayOfWeek': 'int64',
    'Severe_Accident': 'int64'
}

# Maximum acceptable missing value percentage
MAX_MISSING_PERCENTAGE = 0.10  # 10%

# Maximum acceptable duplicate percentage
MAX_DUPLICATE_PERCENTAGE = 0.01  # 1%


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = PROJECT_ROOT / 'project.log'


# ============================================================================
# API & DEPLOYMENT
# ============================================================================

# API settings
API_HOST = '0.0.0.0'
API_PORT = 8000
API_WORKERS = 4

# Model serving
MODEL_CACHE_SIZE = 100  # Number of prediction results to cache
PREDICTION_TIMEOUT = 5  # Seconds

# Monitoring
ENABLE_MONITORING = True
MONITORING_INTERVAL = 3600  # Seconds (1 hour)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_all_valid_features():
    """Get flattened list of all valid pre-flight features"""
    features = []
    for category, feature_list in VALID_FEATURE_CATEGORIES.items():
        features.extend(feature_list)
    return features


def is_leaked_feature(feature_name):
    """Check if a feature is a leaked (post-accident) feature"""
    return feature_name in LEAKED_FEATURES


def validate_feature_set(feature_list):
    """Validate that feature list doesn't contain leaked features"""
    leaked_found = [f for f in feature_list if is_leaked_feature(f)]
    
    if leaked_found:
        raise ValueError(
            f"❌ DATA LEAKAGE DETECTED! Found {len(leaked_found)} leaked features: {leaked_found}"
        )
    
    return True


def get_risk_level(probability):
    """Convert probability to risk level"""
    if probability < LOW_RISK_THRESHOLD:
        return 'Low'
    elif probability < MEDIUM_RISK_THRESHOLD:
        return 'Medium'
    else:
        return 'High'


def calculate_business_impact(tp, fp, fn, tn):
    """Calculate business impact metrics"""
    total_cost = (
        fn * COST_FALSE_NEGATIVE +
        fp * COST_FALSE_POSITIVE -
        tp * BENEFIT_TRUE_POSITIVE
    )
    
    total_benefit = (
        tp * (BENEFIT_TRUE_POSITIVE + BENEFIT_REPUTATION + BENEFIT_INSURANCE_SAVINGS)
    )
    
    net_impact = total_benefit - total_cost
    
    return {
        'total_cost': total_cost,
        'total_benefit': total_benefit,
        'net_impact': net_impact,
        'roi': (net_impact / total_cost * 100) if total_cost > 0 else float('inf')
    }


# ============================================================================
# VERSION INFORMATION
# ============================================================================

PROJECT_VERSION = '1.0.0'
DATA_VERSION = '2023.1'
MODEL_VERSION = '1.0.0-clean'  # Incremented after fixing data leakage


if __name__ == '__main__':
    """Test configuration"""
    print("=" * 60)
    print("🔧 AVIATION SAFETY PROJECT CONFIGURATION")
    print("=" * 60)
    
    print(f"\n📁 Project Root: {PROJECT_ROOT}")
    print(f"📊 Data Directory: {DATA_DIR}")
    print(f"🤖 Models Directory: {MODELS_DIR}")
    print(f"📄 Papers Directory: {PAPERS_DIR}")
    
    print(f"\n🎯 Random State: {RANDOM_STATE}")
    print(f"🔄 CV Folds: {CV_FOLDS}")
    print(f"📊 Test Size: {TEST_SIZE}")
    
    print(f"\n⚠️  Leaked Features: {len(LEAKED_FEATURES)}")
    print(f"✅ Valid Feature Categories: {len(VALID_FEATURE_CATEGORIES)}")
    print(f"✅ Total Valid Features: {len(get_all_valid_features())}")
    
    print(f"\n💰 Cost of False Negative: ${COST_FALSE_NEGATIVE:,}")
    print(f"💰 Cost of False Positive: ${COST_FALSE_POSITIVE:,}")
    print(f"💰 Benefit of True Positive: ${BENEFIT_TRUE_POSITIVE:,}")
    
    print(f"\n📊 Deployment Benchmarks:")
    for metric, threshold in DEPLOYMENT_BENCHMARKS.items():
        print(f"  • {metric}: ≥ {threshold:.2f}")
    
    print(f"\n✅ Configuration loaded successfully!")
    print("=" * 60)
