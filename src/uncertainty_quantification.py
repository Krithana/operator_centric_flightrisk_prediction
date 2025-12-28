# src/uncertainty_quantification.py
import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

class UncertaintyQuantifier:
    """Quantify model uncertainty using bootstrapping and other methods"""
    
    def __init__(self, base_model, n_bootstraps=100):
        self.base_model = base_model
        self.n_bootstraps = n_bootstraps
        self.bootstrap_models = []
        self.bootstrap_scores = []
    
    def fit_bootstrap_models(self, X, y):
        """Fit multiple models on bootstrap samples"""
        print(f"🔄 Training {self.n_bootstraps} bootstrap models...")
        
        for i in range(self.n_bootstraps):
            # Create bootstrap sample
            X_boot, y_boot = resample(X, y, random_state=i)
            
            # Clone and train model
            model = self._clone_model(self.base_model)
            model.fit(X_boot, y_boot)
            self.bootstrap_models.append(model)
            
            # Calculate bootstrap score
            y_pred = model.predict(X)
            score = accuracy_score(y, y_pred)
            self.bootstrap_scores.append(score)
            
            if (i + 1) % 20 == 0:
                print(f"   Completed {i + 1}/{self.n_bootstraps} bootstrap models")
    
    def predict_with_uncertainty(self, X):
        """Predict with confidence intervals"""
        if not self.bootstrap_models:
            raise ValueError("Bootstrap models not fitted. Call fit_bootstrap_models first.")
        
        all_predictions = []
        all_probabilities = []
        
        for model in self.bootstrap_models:
            pred = model.predict(X)
            proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else pred
            
            all_predictions.append(pred)
            all_probabilities.append(proba)
        
        # Convert to arrays
        predictions_array = np.array(all_predictions)
        probabilities_array = np.array(all_probabilities)
        
        # Calculate statistics
        mean_prediction = predictions_array.mean(axis=0)
        mean_probability = probabilities_array.mean(axis=0)
        std_probability = probabilities_array.std(axis=0)

        # Compute majority (mode) prediction per sample for binary predictions
        # and calculate the fraction of bootstrap models that agree with the majority.
        # This yields a per-sample confidence in [0,1].
        try:
            majority_prediction = (predictions_array.mean(axis=0) >= 0.5).astype(int)
            prediction_confidence = np.mean(predictions_array == majority_prediction[None, :], axis=0)
        except Exception:
            # Fallback: compute agreement with rounded mean_prediction
            majority_prediction = np.round(mean_prediction).astype(int)
            prediction_confidence = np.mean(predictions_array == majority_prediction[None, :], axis=0)
        
        # Confidence intervals
        ci_lower = np.percentile(probabilities_array, 2.5, axis=0)
        ci_upper = np.percentile(probabilities_array, 97.5, axis=0)
        
        return {
            'mean_prediction': mean_prediction,
            'mean_probability': mean_probability,
            'std_probability': std_probability,
            'prediction_confidence': prediction_confidence,
            'confidence_interval_lower': ci_lower,
            'confidence_interval_upper': ci_upper,
            'uncertainty_score': std_probability,  # Higher std = more uncertainty
            'all_predictions': predictions_array,
            'all_probabilities': probabilities_array
        }
    
    def calculate_calibration_metrics(self, y_true, y_proba, n_bins=10):
        """Calculate calibration metrics"""
        # Create bins
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_proba, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        calibration_data = []
        
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_mean_prob = y_proba[mask].mean()
                bin_actual_rate = y_true[mask].mean()
                bin_count = mask.sum()
                
                calibration_data.append({
                    'bin': i + 1,
                    'mean_predicted_prob': bin_mean_prob,
                    'actual_positive_rate': bin_actual_rate,
                    'count': bin_count,
                    'calibration_error': abs(bin_mean_prob - bin_actual_rate)
                })
        
        calibration_df = pd.DataFrame(calibration_data)
        expected_calibration_error = (calibration_df['calibration_error'] * 
                                    calibration_df['count']).sum() / calibration_df['count'].sum()
        
        return {
            'calibration_df': calibration_df,
            'expected_calibration_error': expected_calibration_error,
            'perfect_calibration': expected_calibration_error < 0.05
        }
    
    def adversarial_robustness_test(self, X, y, noise_level=0.1):
        """Test model robustness to adversarial examples"""
        # Add noise to create adversarial examples
        X_adversarial = X + np.random.normal(0, noise_level, X.shape)
        
        # Get predictions on adversarial examples
        adversarial_predictions = self.predict_with_uncertainty(X_adversarial)
        original_predictions = self.predict_with_uncertainty(X)
        
        # Calculate robustness metrics
        prediction_change = np.mean(adversarial_predictions['mean_prediction'] != 
                                  original_predictions['mean_prediction'])
        
        probability_change = np.mean(np.abs(adversarial_predictions['mean_probability'] - 
                                          original_predictions['mean_probability']))
        
        return {
            'prediction_change_rate': prediction_change,
            'mean_probability_change': probability_change,
            'robustness_score': 1 - prediction_change,
            'adversarial_predictions': adversarial_predictions
        }
    
    def _clone_model(self, model):
        """Create a copy of the model"""
        from sklearn.base import clone
        return clone(model)
    
    def get_uncertainty_summary(self, X, y_true=None):
        """Get comprehensive uncertainty summary"""
        predictions = self.predict_with_uncertainty(X)
        
        summary = {
            'mean_uncertainty': np.mean(predictions['std_probability']),
            'high_uncertainty_count': np.sum(predictions['std_probability'] > 0.2),
            'high_uncertainty_proportion': np.mean(predictions['std_probability'] > 0.2),
            'bootstrap_score_mean': np.mean(self.bootstrap_scores),
            'bootstrap_score_std': np.std(self.bootstrap_scores)
        }
        
        if y_true is not None:
            calibration = self.calculate_calibration_metrics(y_true, predictions['mean_probability'])
            summary.update(calibration)
        
        return summary