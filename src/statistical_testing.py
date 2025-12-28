# src/statistical_testing.py
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportions_ztest
import warnings
warnings.filterwarnings('ignore')

class StatisticalTester:
    """Comprehensive statistical testing for model evaluation"""
    
    def mcnemar_test(self, y_true, y_pred1, y_pred2):
        """McNemar's test for comparing two models"""
        # Create contingency table
        n00 = np.sum((y_pred1 == y_true) & (y_pred2 == y_true))
        n01 = np.sum((y_pred1 == y_true) & (y_pred2 != y_true))
        n10 = np.sum((y_pred1 != y_true) & (y_pred2 == y_true))
        n11 = np.sum((y_pred1 != y_true) & (y_pred2 != y_true))
        
        contingency_table = [[n00, n01], [n10, n11]]
        
        # Perform McNemar's test
        result = mcnemar(contingency_table, exact=False)
        
        return {
            'statistic': result.statistic,
            'p_value': result.pvalue,
            'significant': result.pvalue < 0.05,
            'contingency_table': contingency_table
        }
    
    def paired_t_test(self, scores1, scores2):
        """Paired t-test for model performance scores"""
        t_stat, p_value = stats.ttest_rel(scores1, scores2)
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'mean_difference': np.mean(scores1) - np.mean(scores2)
        }
    
    def confidence_intervals(self, scores, confidence=0.95):
        """Calculate confidence intervals for performance metrics"""
        n = len(scores)
        mean = np.mean(scores)
        std_err = stats.sem(scores)
        
        ci = stats.t.interval(confidence, n-1, loc=mean, scale=std_err)
        
        return {
            'mean': mean,
            'std_error': std_err,
            f'ci_{int(confidence*100)}': ci,
            'ci_lower': ci[0],
            'ci_upper': ci[1]
        }
    
    def proportion_test(self, success1, total1, success2, total2):
        """Test for difference in proportions"""
        count = np.array([success1, success2])
        nobs = np.array([total1, total2])
        
        z_stat, p_value = proportions_ztest(count, nobs)
        
        return {
            'z_statistic': z_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'proportion1': success1/total1,
            'proportion2': success2/total2
        }
    
    def comprehensive_model_comparison(self, model_results, X_test, y_test):
        """Comprehensive statistical comparison of multiple models"""
        comparisons = {}
        
        model_names = list(model_results.keys())
        
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                model1 = model_names[i]
                model2 = model_names[j]
                
                # Get predictions
                y_pred1 = model_results[model1]['predictions']
                y_pred2 = model_results[model2]['predictions']
                
                # McNemar's test
                mcnemar_result = self.mcnemar_test(y_test, y_pred1, y_pred2)
                
                # Accuracy comparison
                acc1 = np.mean(y_pred1 == y_test)
                acc2 = np.mean(y_pred2 == y_test)
                prop_test = self.proportion_test(
                    int(acc1 * len(y_test)), len(y_test),
                    int(acc2 * len(y_test)), len(y_test)
                )
                
                comparisons[f"{model1}_vs_{model2}"] = {
                    'mcnemar': mcnemar_result,
                    'proportion_test': prop_test,
                    'accuracy_difference': acc1 - acc2
                }
        
        return comparisons

def perform_ablation_study(X, y, feature_groups, model_class, test_size=0.2):
    """Perform ablation study with different feature groups"""
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score, accuracy_score
    
    results = {}
    
    # Baseline with all features
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    baseline_model = model_class()
    baseline_model.fit(X_train, y_train)
    y_pred = baseline_model.predict(X_test)
    
    results['all_features'] = {
        'f1': f1_score(y_test, y_pred),
        'accuracy': accuracy_score(y_test, y_pred),
        'features_used': X.shape[1]
    }
    
    # Test each feature group
    for group_name, features in feature_groups.items():
        X_subset = X[features]
        X_train, X_test, y_train, y_test = train_test_split(X_subset, y, test_size=test_size, random_state=42)
        
        model = model_class()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        results[group_name] = {
            'f1': f1_score(y_test, y_pred),
            'accuracy': accuracy_score(y_test, y_pred),
            'features_used': len(features)
        }
    
    return pd.DataFrame(results).T