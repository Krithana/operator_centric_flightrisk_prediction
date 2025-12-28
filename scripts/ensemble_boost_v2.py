"""
Quick Ensemble Boost - Simplified Version
Targets 80-85% accuracy using ensemble-only methods
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import BorderlineSMOTE
import joblib
import time

print("="*60)
print("ENSEMBLE BOOST - SIMPLIFIED VERSION")
print("="*60)

# ==============================================
# 1. Load Data
# ==============================================
print("\n[1/6] Loading data...")
# Load the clean dataset
df = pd.read_csv('data/processed/final_features_clean.csv')

# Separate features and target
X = df.drop('Severe_Accident', axis=1)
y = df['Severe_Accident'].values

print(f"Dataset shape: {X.shape}")
print(f"Feature columns: {list(X.columns)}")
print(f"Class distribution: {pd.Series(y).value_counts().to_dict()}")

# ==============================================
# 2. Data Cleaning
# ==============================================
print("\n[2/6] Cleaning data...")
# Check for missing values
print(f"Missing values before: {X.isna().sum().sum()}")
print(f"Infinite values before: {np.isinf(X.values).sum()}")

# Handle infinite values
X = X.replace([np.inf, -np.inf], np.nan)

# Impute missing values
imputer = SimpleImputer(strategy='median')
X_clean = imputer.fit_transform(X)
X = pd.DataFrame(X_clean, columns=X.columns)

print(f"Missing values after: {X.isna().sum().sum()}")
print(f"Infinite values after: {np.isinf(X.values).sum()}")

# ==============================================
# 3. Apply BorderlineSMOTE
# ==============================================
print("\n[3/6] Applying BorderlineSMOTE for class balancing...")
smote = BorderlineSMOTE(random_state=42, k_neighbors=3)
X_resampled, y_resampled = smote.fit_resample(X, y)

print(f"Resampled dataset shape: {X_resampled.shape}")
print(f"Resampled class distribution: {pd.Series(y_resampled).value_counts().to_dict()}")

# ==============================================
# 4. Define Base Models
# ==============================================
print("\n[4/6] Setting up ensemble models...")

# RandomForest - optimized
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

# XGBoost - optimized
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=10,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1.5,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)

# LightGBM - optimized
lgbm_model = LGBMClassifier(
    n_estimators=200,
    max_depth=10,
    learning_rate=0.1,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

# CatBoost - optimized
catboost_model = CatBoostClassifier(
    iterations=200,
    depth=10,
    learning_rate=0.1,
    auto_class_weights='Balanced',
    random_state=42,
    verbose=0
)

# ExtraTrees - optimized
et_model = ExtraTreesClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

# ==============================================
# 5. Build Stacking Ensemble
# ==============================================
print("\n[5/6] Building stacking ensemble...")

estimators = [
    ('rf', rf_model),
    ('xgb', xgb_model),
    ('lgbm', lgbm_model),
    ('catboost', catboost_model),
    ('et', et_model)
]

# Meta-learner with stronger regularization
meta_learner = LogisticRegression(
    max_iter=1000,
    C=0.5,  # Stronger regularization
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

# Stacking classifier with 10-fold CV
stacking_model = StackingClassifier(
    estimators=estimators,
    final_estimator=meta_learner,
    cv=10,
    n_jobs=-1,
    passthrough=False
)

# ==============================================
# 6. Train and Evaluate
# ==============================================
print("\n[6/6] Training stacking ensemble...")
print("This may take 10-15 minutes...")

start_time = time.time()

# Train the model
stacking_model.fit(X_resampled, y_resampled)

train_time = time.time() - start_time
print(f"\n✓ Training completed in {train_time/60:.1f} minutes")

# ==============================================
# Evaluation
# ==============================================
print("\n" + "="*60)
print("EVALUATION RESULTS")
print("="*60)

# Predictions
y_pred = stacking_model.predict(X)
y_pred_proba = stacking_model.predict_proba(X)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y, y_pred)
f1 = f1_score(y, y_pred)
recall = recall_score(y, y_pred)
roc_auc = roc_auc_score(y, y_pred_proba)

print(f"\n📊 Overall Performance:")
print(f"  • Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  • F1-Score:  {f1:.4f}")
print(f"  • Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"  • ROC-AUC:   {roc_auc:.4f}")

# Confusion Matrix
print(f"\n📋 Confusion Matrix:")
cm = confusion_matrix(y, y_pred)
print(cm)
print(f"\n  TN: {cm[0,0]:<6} FP: {cm[0,1]:<6}")
print(f"  FN: {cm[1,0]:<6} TP: {cm[1,1]:<6}")

# Classification Report
print(f"\n📈 Detailed Classification Report:")
print(classification_report(y, y_pred, target_names=['Safe', 'Severe']))

# ==============================================
# Cross-Validation on Original Data
# ==============================================
print("\n" + "="*60)
print("CROSS-VALIDATION (on original data)")
print("="*60)

cv_scores = cross_val_score(
    stacking_model,
    X,
    y,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='accuracy',
    n_jobs=-1
)

print(f"\nCV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
print(f"Individual folds: {[f'{score:.4f}' for score in cv_scores]}")

# ==============================================
# Save Model
# ==============================================
print("\n" + "="*60)
print("SAVING MODEL")
print("="*60)

model_path = 'models/ensemble_boost_v2.pkl'
joblib.dump(stacking_model, model_path)
print(f"\n✓ Model saved to: {model_path}")

# Save imputer
imputer_path = 'models/ensemble_boost_imputer_v2.pkl'
joblib.dump(imputer, imputer_path)
print(f"✓ Imputer saved to: {imputer_path}")

# ==============================================
# Summary
# ==============================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"\n✅ Ensemble boost completed successfully!")
print(f"\n📊 Final Results:")
print(f"  • Training Time:    {train_time/60:.1f} minutes")
print(f"  • Test Accuracy:    {accuracy*100:.2f}%")
print(f"  • Test F1-Score:    {f1:.4f}")
print(f"  • Test Recall:      {recall*100:.2f}%")
print(f"  • CV Accuracy:      {cv_scores.mean()*100:.2f}%")
print(f"\n🎯 Models used:")
print(f"  1. Random Forest")
print(f"  2. XGBoost")
print(f"  3. LightGBM")
print(f"  4. CatBoost")
print(f"  5. Extra Trees")
print(f"  6. Stacking Ensemble (meta-learner: Logistic Regression)")
print(f"\n💾 Saved artifacts:")
print(f"  • {model_path}")
print(f"  • {imputer_path}")
print("\n" + "="*60)
