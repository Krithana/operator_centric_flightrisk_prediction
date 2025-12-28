# # src/model_training.py
# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from xgboost import XGBClassifier
# from sklearn.model_selection import cross_val_score, GridSearchCV
# from sklearn.metrics import classification_report, confusion_matrix
# import joblib
# from utils import ResearchConfig, save_model

# class ModelTrainer:
#     def __init__(self):
#         self.models = {}
#         self.best_model = None
#         self.best_score = 0
        
#     def initialize_models(self):
#         """Initialize multiple models for comparison"""
#         print("🤖 Initializing models...")
        
#         self.models = {
#             'random_forest': RandomForestClassifier(
#                 n_estimators=100,
#                 random_state=ResearchConfig.RANDOM_STATE,
#                 class_weight='balanced'
#             ),
#             'xgboost': XGBClassifier(
#                 n_estimators=100,
#                 random_state=ResearchConfig.RANDOM_STATE,
#                 eval_metric='logloss',
#                 use_label_encoder=False
#             ),
#             'gradient_boosting': GradientBoostingClassifier(
#                 n_estimators=100,
#                 random_state=ResearchConfig.RANDOM_STATE
#             ),
#             'logistic_regression': LogisticRegression(
#                 random_state=ResearchConfig.RANDOM_STATE,
#                 class_weight='balanced',
#                 max_iter=1000
#             ),
#             'svm': SVC(
#                 random_state=ResearchConfig.RANDOM_STATE,
#                 class_weight='balanced',
#                 probability=True
#             )
#         }
    
#     def train_models(self, X_train, y_train):
#         """Train all models and evaluate performance"""
#         print("🎯 Training models...")
        
#         results = {}
        
#         for name, model in self.models.items():
#             print(f"   Training {name}...")
            
#             # Cross-validation
#             cv_scores = cross_val_score(model, X_train, y_train, 
#                                       cv=ResearchConfig.CV_FOLDS, 
#                                       scoring='f1_macro')
            
#             # Train model
#             model.fit(X_train, y_train)
            
#             results[name] = {
#                 'model': model,
#                 'cv_mean': cv_scores.mean(),
#                 'cv_std': cv_scores.std(),
#                 'cv_scores': cv_scores
#             }
            
#             print(f"   ✅ {name}: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            
#             # Update best model
#             if cv_scores.mean() > self.best_score:
#                 self.best_score = cv_scores.mean()
#                 self.best_model = model
#                 self.best_model_name = name
        
#         print(f"\n🏆 Best model: {self.best_model_name} (F1: {self.best_score:.4f})")
        
#         return results
    
#     def hyperparameter_tuning(self, X_train, y_train):
#         """Perform hyperparameter tuning for best model"""
#         print("🎛️  Performing hyperparameter tuning...")
        
#         if self.best_model_name == 'random_forest':
#             param_grid = {
#                 'n_estimators': [100, 200, 300],
#                 'max_depth': [10, 20, None],
#                 'min_samples_split': [2, 5, 10],
#                 'min_samples_leaf': [1, 2, 4]
#             }
#         elif self.best_model_name == 'xgboost':
#             param_grid = {
#                 'n_estimators': [100, 200, 300],
#                 'max_depth': [3, 6, 9],
#                 'learning_rate': [0.01, 0.1, 0.2],
#                 'subsample': [0.8, 0.9, 1.0]
#             }
#         else:
#             print("   ⚠️  Hyperparameter tuning not implemented for this model")
#             return self.best_model
        
#         grid_search = GridSearchCV(
#             self.best_model,
#             param_grid,
#             cv=ResearchConfig.CV_FOLDS,
#             scoring='f1_macro',
#             n_jobs=-1,
#             verbose=1
#         )
        
#         grid_search.fit(X_train, y_train)
        
#         print(f"   Best parameters: {grid_search.best_params_}")
#         print(f"   Best score: {grid_search.best_score_:.4f}")
        
#         self.best_model = grid_search.best_estimator_
        
#         return self.best_model
    
#     def evaluate_models(self, results, X_test, y_test):
#         """Evaluate all models on test set"""
#         print("\n📊 Model Evaluation on Test Set:")
#         print("="*50)
        
#         test_results = {}
        
#         for name, result in results.items():
#             model = result['model']
#             y_pred = model.predict(X_test)
#             y_pred_proba = model.predict_proba(X_test)[:, 1]
            
#             # Calculate metrics
#             from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
            
#             test_results[name] = {
#                 'precision': precision_score(y_test, y_pred),
#                 'recall': recall_score(y_test, y_pred),
#                 'f1': f1_score(y_test, y_pred),
#                 'roc_auc': roc_auc_score(y_test, y_pred_proba)
#             }
            
#             print(f"\n{name.upper():<20}")
#             print(f"  Precision:  {test_results[name]['precision']:.4f}")
#             print(f"  Recall:     {test_results[name]['recall']:.4f}")
#             print(f"  F1-Score:   {test_results[name]['f1']:.4f}")
#             print(f"  ROC-AUC:    {test_results[name]['roc_auc']:.4f}")
        
#         return test_results
    
#     def save_trained_models(self, results):
#         """Save all trained models"""
#         print("\n💾 Saving trained models...")
        
#         for name, result in results.items():
#             save_model(result['model'], f"{name}_model")
        
#         # Save best model separately
#         save_model(self.best_model, "flight_risk_model")
#         print(f"🏆 Best model saved as: flight_risk_model.pkl")

# def main():
#     """Main model training pipeline"""
#     from feature_engineering import main as feature_main
    
#     # Get feature-engineered data
#     X_train, X_test, y_train, y_test, features = feature_main()
    
#     # Initialize and train models
#     trainer = ModelTrainer()
#     trainer.initialize_models()
    
#     # Train models
#     results = trainer.train_models(X_train, y_train)
    
#     # Hyperparameter tuning
#     trainer.hyperparameter_tuning(X_train, y_train)
    
#     # Evaluate models
#     test_results = trainer.evaluate_models(results, X_test, y_test)
    
#     # Save models
#     trainer.save_trained_models(results)
    
#     # Save feature names
#     joblib.dump(features, '../models/feature_names.pkl')
    
#     print(f"\n🎉 Model training completed!")
#     print(f"   Best model: {trainer.best_model_name}")
#     print(f"   Test F1-Score: {test_results[trainer.best_model_name]['f1']:.4f}")
    
#     return trainer.best_model, test_results, features

# if __name__ == "__main__":
#     best_model, test_results, features = main()

# src/model_training.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix,f1_score
import joblib
from utils import ResearchConfig, save_model
from uncertainty_quantification import UncertaintyQuantifier

class EnhancedModelTrainer:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_score = 0
        self.uncertainty_quantifier = None
        
    def initialize_models(self):
        """Initialize multiple models for comparison including ensemble"""
        print("🤖 Initializing enhanced models...")
        
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                random_state=ResearchConfig.RANDOM_STATE,
                class_weight='balanced',
                max_depth=20
            ),
            'xgboost': XGBClassifier(
                n_estimators=200,
                random_state=ResearchConfig.RANDOM_STATE,
                eval_metric='logloss',
                use_label_encoder=False,
                max_depth=6,
                learning_rate=0.1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=200,
                random_state=ResearchConfig.RANDOM_STATE,
                max_depth=5
            ),
            'logistic_regression': LogisticRegression(
                random_state=ResearchConfig.RANDOM_STATE,
                class_weight='balanced',
                max_iter=1000,
                C=0.1
            ),
            'svm': SVC(
                random_state=ResearchConfig.RANDOM_STATE,
                class_weight='balanced',
                probability=True,
                kernel='rbf',
                C=1.0
            ),
            'naive_bayes': GaussianNB()
        }
        
        # Create stacked ensemble
        self.models['stacked_ensemble'] = self.create_stacked_ensemble()
    
    def create_stacked_ensemble(self):
        """Create stacked ensemble model"""
        base_models = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=ResearchConfig.RANDOM_STATE)),
            ('xgb', XGBClassifier(random_state=ResearchConfig.RANDOM_STATE, use_label_encoder=False)),
            ('gb', GradientBoostingClassifier(random_state=ResearchConfig.RANDOM_STATE))
        ]
        
        stack_model = StackingClassifier(
            estimators=base_models,
            final_estimator=LogisticRegression(),
            cv=5
        )
        
        return stack_model
    
    def time_series_cross_validation(self, X, y, model):
        """Time-series aware cross-validation"""
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            scores.append(f1_score(y_test, y_pred))
        
        return np.mean(scores), np.std(scores)
    
    def train_models(self, X_train, y_train):
        """Train all models with enhanced validation"""
        print("🎯 Training models with enhanced validation...")
        
        results = {}
        
        for name, model in self.models.items():
            print(f"   Training {name}...")
            
            # Standard cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, 
                                      cv=ResearchConfig.CV_FOLDS, 
                                      scoring='f1_macro')
            
            # Time-series cross-validation for temporal data
            try:
                ts_mean, ts_std = self.time_series_cross_validation(X_train, y_train, model)
            except:
                ts_mean, ts_std = cv_scores.mean(), cv_scores.std()
            
            # Train model
            model.fit(X_train, y_train)
            
            results[name] = {
                'model': model,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_scores': cv_scores,
                'ts_cv_mean': ts_mean,
                'ts_cv_std': ts_std
            }
            
            print(f"   ✅ {name}: Standard CV = {cv_scores.mean():.4f}, Time-series CV = {ts_mean:.4f}")
            
            # Update best model
            current_score = (cv_scores.mean() + ts_mean) / 2  # Combined score
            if current_score > self.best_score:
                self.best_score = current_score
                self.best_model = model
                self.best_model_name = name
        
        print(f"\n🏆 Best model: {self.best_model_name} (Combined Score: {self.best_score:.4f})")
        
        return results
    
    def hyperparameter_tuning(self, X_train, y_train):
        """Enhanced hyperparameter tuning with multiple strategies"""
        print("🎛️  Performing enhanced hyperparameter tuning...")
        
        if self.best_model_name == 'random_forest':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            }
        elif self.best_model_name == 'xgboost':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
        elif self.best_model_name == 'stacked_ensemble':
            print("   ⚠️  Hyperparameter tuning for stacked ensemble is complex, using default")
            return self.best_model
        else:
            print(f"   ⚠️  Hyperparameter tuning not implemented for {self.best_model_name}")
            return self.best_model
        
        # Use time-series cross-validation for temporal data
        cv_strategy = TimeSeriesSplit(n_splits=5)
        
        grid_search = GridSearchCV(
            self.best_model,
            param_grid,
            cv=cv_strategy,
            scoring='f1_macro',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"   Best parameters: {grid_search.best_params_}")
        print(f"   Best score: {grid_search.best_score_:.4f}")
        
        self.best_model = grid_search.best_estimator_
        
        return self.best_model
    
    def setup_uncertainty_quantification(self, X_train, y_train):
        """Setup uncertainty quantification for the best model"""
        print("🔮 Setting up uncertainty quantification...")
        
        self.uncertainty_quantifier = UncertaintyQuantifier(self.best_model, n_bootstraps=50)
        self.uncertainty_quantifier.fit_bootstrap_models(X_train, y_train)
        
        return self.uncertainty_quantifier
    
    def evaluate_models(self, results, X_test, y_test):
        """Enhanced model evaluation with uncertainty"""
        print("\n📊 Enhanced Model Evaluation on Test Set:")
        print("=" * 60)
        
        test_results = {}
        
        for name, result in results.items():
            model = result['model']
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate comprehensive metrics
            from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
            
            test_results[name] = {
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'average_precision': average_precision_score(y_test, y_pred_proba),
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'model': model
            }
            
            print(f"\n{name.upper():<20}")
            print(f"  Precision:  {test_results[name]['precision']:.4f}")
            print(f"  Recall:     {test_results[name]['recall']:.4f}")
            print(f"  F1-Score:   {test_results[name]['f1']:.4f}")
            print(f"  ROC-AUC:    {test_results[name]['roc_auc']:.4f}")
            print(f"  Avg Precision: {test_results[name]['average_precision']:.4f}")
        
        return test_results
    
    def save_trained_models(self, results):
        """Save all trained models with metadata"""
        print("\n💾 Saving trained models with metadata...")
        
        for name, result in results.items():
            save_model(result['model'], f"{name}_model")
        
        # Save best model separately
        save_model(self.best_model, "flight_risk_model")
        
        # Save uncertainty quantifier if available
        if self.uncertainty_quantifier:
            save_model(self.uncertainty_quantifier, "uncertainty_quantifier")
        
        print(f"🏆 Best model saved as: flight_risk_model.pkl")

def main():
    """Main enhanced model training pipeline"""
    from feature_engineering import main as feature_main
    
    # Get feature-engineered data
    X_train, X_test, y_train, y_test, features = feature_main()
    
    # Initialize and train models
    trainer = EnhancedModelTrainer()
    trainer.initialize_models()
    
    # Train models
    results = trainer.train_models(X_train, y_train)
    
    # Hyperparameter tuning
    trainer.hyperparameter_tuning(X_train, y_train)
    
    # Setup uncertainty quantification
    trainer.setup_uncertainty_quantification(X_train, y_train)
    
    # Evaluate models
    test_results = trainer.evaluate_models(results, X_test, y_test)
    
    # Save models
    trainer.save_trained_models(results)
    
    # Save feature names
    joblib.dump(features, '../models/feature_names.pkl')
    
    print(f"\n🎉 Enhanced model training completed!")
    print(f"   Best model: {trainer.best_model_name}")
    print(f"   Test F1-Score: {test_results[trainer.best_model_name]['f1']:.4f}")
    
    return trainer.best_model, test_results, features, trainer.uncertainty_quantifier

if __name__ == "__main__":
    best_model, test_results, features, uncertainty_quantifier = main()