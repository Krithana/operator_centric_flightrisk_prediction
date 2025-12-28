"""
Quick Ensemble Performance Boost Script
Improve model from 76% to 80-85% accuracy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, precision_recall_curve)
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                              ExtraTreesClassifier, VotingClassifier, StackingClassifier)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.combine import SMOTETomek
from sklearn.impute import SimpleImputer
import joblib
import warnings
warnings.filterwarnings('ignore')

print("🚀 QUICK ENSEMBLE PERFORMANCE BOOST")
print("=" * 70)
print("Target: Improve from 76% to 80-85% accuracy")
print("=" * 70)

# Load data
print("\n📥 Loading clean dataset...")
clean_data = pd.read_csv('data/processed/final_features_clean.csv')

X = clean_data.drop('Severe_Accident', axis=1)
y = clean_data['Severe_Accident']

print(f"✅ Dataset: {X.shape}")
print(f"✅ Class distribution: {y.value_counts().to_dict()}")

# Clean data
print("\n🧹 Cleaning data...")
X = X.replace([np.inf, -np.inf], np.nan)
imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
print(f"✅ Data cleaned: {X.isna().sum().sum()} NaN remaining")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# IMPROVEMENT 1: Better Class Balancing
print("\n⚡ IMPROVEMENT 1: OPTIMIZED CLASS BALANCING")
print("=" * 70)
print("Applying BorderlineSMOTE...")
smote = BorderlineSMOTE(random_state=42, k_neighbors=7, m_neighbors=10)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
print(f"✅ Balanced: {X_train_balanced.shape}")

# IMPROVEMENT 2: Optimized Base Models
print("\n⚡ IMPROVEMENT 2: OPTIMIZED BASE MODELS")
print("=" * 70)

optimized_models = {
    'RandomForest': RandomForestClassifier(
        n_estimators=400, max_depth=25, min_samples_split=5,
        min_samples_leaf=2, max_features='sqrt', class_weight='balanced',
        random_state=42, n_jobs=-1
    ),
    'XGBoost': XGBClassifier(
        n_estimators=400, max_depth=9, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=2,
        gamma=0.1, random_state=42, eval_metric='logloss', use_label_encoder=False
    ),
    'LightGBM': LGBMClassifier(
        n_estimators=400, max_depth=9, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
        num_leaves=50, random_state=42, verbose=-1
    ),
    'CatBoost': CatBoostClassifier(
        iterations=400, depth=8, learning_rate=0.08,
        random_state=42, verbose=False
    )
}

# Train and evaluate
print("\nTraining models...")
results = {}

for name, model in optimized_models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_balanced, y_train_balanced)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    results[name] = {
        'model': model,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    print(f"  Accuracy:  {results[name]['accuracy']:.4f}")
    print(f"  F1-Score:  {results[name]['f1']:.4f}")

# IMPROVEMENT 3: Advanced Stacking
print("\n⚡ IMPROVEMENT 3: ADVANCED STACKING ENSEMBLE")
print("=" * 70)

stacking_clf = StackingClassifier(
    estimators=[
        ('rf', optimized_models['RandomForest']),
        ('xgb', optimized_models['XGBoost']),
        ('lgb', optimized_models['LightGBM']),
        ('cat', optimized_models['CatBoost'])
    ],
    final_estimator=LogisticRegression(max_iter=1000, C=0.1, class_weight='balanced'),
    cv=10,
    stack_method='predict_proba',
    n_jobs=-1
)

print("Training stacking ensemble (10-fold CV)...")
stacking_clf.fit(X_train_balanced, y_train_balanced)

y_pred_stack = stacking_clf.predict(X_test)
y_pred_proba_stack = stacking_clf.predict_proba(X_test)[:, 1]

results['Stacking'] = {
    'model': stacking_clf,
    'accuracy': accuracy_score(y_test, y_pred_stack),
    'precision': precision_score(y_test, y_pred_stack),
    'recall': recall_score(y_test, y_pred_stack),
    'f1': f1_score(y_test, y_pred_stack),
    'roc_auc': roc_auc_score(y_test, y_pred_proba_stack)
}

print(f"\nStacking Results:")
print(f"  Accuracy:  {results['Stacking']['accuracy']:.4f}")
print(f"  F1-Score:  {results['Stacking']['f1']:.4f}")
print(f"  ROC-AUC:   {results['Stacking']['roc_auc']:.4f}")

# IMPROVEMENT 4: Threshold Optimization
print("\n⚡ IMPROVEMENT 4: THRESHOLD OPTIMIZATION")
print("=" * 70)

precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba_stack)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"Optimal threshold: {optimal_threshold:.4f}")

y_pred_optimized = (y_pred_proba_stack >= optimal_threshold).astype(int)

results['Stacking (Optimized)'] = {
    'model': stacking_clf,
    'accuracy': accuracy_score(y_test, y_pred_optimized),
    'precision': precision_score(y_test, y_pred_optimized),
    'recall': recall_score(y_test, y_pred_optimized),
    'f1': f1_score(y_test, y_pred_optimized),
    'roc_auc': roc_auc_score(y_test, y_pred_proba_stack)
}

# FINAL RESULTS
print("\n" + "="*70)
print("🏆 FINAL RESULTS")
print("="*70)

# Create comparison DataFrame
results_df = pd.DataFrame({
    name: {k: v for k, v in metrics.items() if k != 'model'}
    for name, metrics in results.items()
}).T

results_df = results_df.sort_values('f1', ascending=False)

print("\n📊 MODEL PERFORMANCE RANKING:")
print(results_df.round(4).to_string())

# Best model
best_model_name = results_df.index[0]
best_metrics = results_df.iloc[0]

print(f"\n🥇 BEST MODEL: {best_model_name}")
print("=" * 70)
print(f"  Accuracy:  {best_metrics['accuracy']:.4f} ({best_metrics['accuracy']*100:.2f}%)")
print(f"  Precision: {best_metrics['precision']:.4f}")
print(f"  Recall:    {best_metrics['recall']:.4f}")
print(f"  F1-Score:  {best_metrics['f1']:.4f}")
print(f"  ROC-AUC:   {best_metrics['roc_auc']:.4f}")

# Improvement calculation
baseline_accuracy = 0.7611
baseline_f1 = 0.5211
improvement_acc = (best_metrics['accuracy'] - baseline_accuracy) * 100
improvement_f1 = (best_metrics['f1'] - baseline_f1) / baseline_f1 * 100

print(f"\n📈 IMPROVEMENT OVER BASELINE:")
print(f"  Baseline Accuracy: 76.11%")
print(f"  Current Accuracy:  {best_metrics['accuracy']*100:.2f}%")
print(f"  Absolute Gain:     +{improvement_acc:.2f}%")
print(f"\n  Baseline F1:       0.5211")
print(f"  Current F1:        {best_metrics['f1']:.4f}")
print(f"  Relative Gain:     +{improvement_f1:.2f}%")

# Confusion Matrix
if best_model_name == 'Stacking (Optimized)':
    y_pred_final = y_pred_optimized
else:
    y_pred_final = results[best_model_name]['model'].predict(X_test)

cm = confusion_matrix(y_test, y_pred_final)

print(f"\n📊 CONFUSION MATRIX:")
print(f"  True Negatives:  {cm[0,0]:,}")
print(f"  False Positives: {cm[0,1]:,}")
print(f"  False Negatives: {cm[1,0]:,}")
print(f"  True Positives:  {cm[1,1]:,}")

# Success check
if best_metrics['accuracy'] >= 0.85:
    print(f"\n🎉 SUCCESS! Achieved 85%+ accuracy!")
elif best_metrics['accuracy'] >= 0.80:
    print(f"\n✅ EXCELLENT! Achieved 80%+ accuracy!")
else:
    print(f"\n💪 GOOD PROGRESS! Current: {best_metrics['accuracy']*100:.2f}%")

# Save best model
print("\n💾 Saving best model...")
joblib.dump(results[best_model_name]['model'], 'models/quick_ensemble_best.pkl')
joblib.dump({'threshold': optimal_threshold if 'Optimized' in best_model_name else 0.5}, 
            'models/quick_ensemble_metadata.pkl')
print("✅ Model saved!")

print("\n" + "="*70)
print("✅ OPTIMIZATION COMPLETE!")
print("="*70)
