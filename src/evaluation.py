# # src/evaluation.py
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import (classification_report, confusion_matrix, 
#                            roc_curve, auc, precision_recall_curve, 
#                            precision_score, recall_score)
# import shap
# import lime
# import lime.lime_tabular
# from utils import ResearchConfig, load_model, save_research_plot

# class ModelEvaluator:
#     def __init__(self, model, feature_names):
#         self.model = model
#         self.feature_names = feature_names
#         self.explainer = None
        
#     def comprehensive_evaluation(self, X_test, y_test, y_pred, y_pred_proba):
#         """Perform comprehensive model evaluation"""
#         print("📈 Comprehensive Model Evaluation")
#         print("="*50)
        
#         # Classification report
#         print("\n📋 Classification Report:")
#         print(classification_report(y_test, y_pred))
        
#         # Confusion matrix
#         self.plot_confusion_matrix(y_test, y_pred)
        
#         # ROC Curve
#         self.plot_roc_curve(y_test, y_pred_proba)
        
#         # Precision-Recall Curve
#         self.plot_precision_recall_curve(y_test, y_pred_proba)
        
#         # Feature importance
#         self.plot_feature_importance()
        
#         # Model interpretability
#         self.model_interpretability(X_test, y_test)
    
#     def plot_confusion_matrix(self, y_test, y_pred):
#         """Plot confusion matrix"""
#         cm = confusion_matrix(y_test, y_pred)
        
#         plt.figure(figsize=(8, 6))
#         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
#                    xticklabels=['Not Severe', 'Severe'],
#                    yticklabels=['Not Severe', 'Severe'])
#         plt.title('Confusion Matrix')
#         plt.ylabel('Actual')
#         plt.xlabel('Predicted')
#         save_research_plot('confusion_matrix.png')
#         plt.show()
    
#     def plot_roc_curve(self, y_test, y_pred_proba):
#         """Plot ROC curve"""
#         fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
#         roc_auc = auc(fpr, tpr)
        
#         plt.figure(figsize=(8, 6))
#         plt.plot(fpr, tpr, color='darkorange', lw=2, 
#                 label=f'ROC curve (AUC = {roc_auc:.4f})')
#         plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#         plt.xlim([0.0, 1.0])
#         plt.ylim([0.0, 1.05])
#         plt.xlabel('False Positive Rate')
#         plt.ylabel('True Positive Rate')
#         plt.title('Receiver Operating Characteristic (ROC) Curve')
#         plt.legend(loc="lower right")
#         save_research_plot('roc_curve.png')
#         plt.show()
    
#     def plot_precision_recall_curve(self, y_test, y_pred_proba):
#         """Plot precision-recall curve"""
#         precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        
#         plt.figure(figsize=(8, 6))
#         plt.plot(recall, precision, color='blue', lw=2)
#         plt.xlabel('Recall')
#         plt.ylabel('Precision')
#         plt.title('Precision-Recall Curve')
#         plt.xlim([0.0, 1.0])
#         plt.ylim([0.0, 1.05])
#         save_research_plot('precision_recall_curve.png')
#         plt.show()
    
#     def plot_feature_importance(self):
#         """Plot feature importance"""
#         if hasattr(self.model, 'feature_importances_'):
#             importance = self.model.feature_importances_
#             feature_imp = pd.DataFrame({
#                 'feature': self.feature_names,
#                 'importance': importance
#             }).sort_values('importance', ascending=False)
            
#             plt.figure(figsize=(10, 8))
#             sns.barplot(data=feature_imp.head(15), x='importance', y='feature')
#             plt.title('Top 15 Feature Importance')
#             plt.xlabel('Importance')
#             save_research_plot('feature_importance.png')
#             plt.show()
            
#             print("\n🔝 Top 10 Most Important Features:")
#             print(feature_imp.head(10))
    
#     def model_interpretability(self, X_test, y_test):
#         """Provide model interpretability using SHAP and LIME"""
#         print("\n🔍 Model Interpretability Analysis")
        
#         # SHAP Analysis
#         self.shap_analysis(X_test)
        
#         # LIME Analysis
#         self.lime_analysis(X_test, y_test)
    
#     def shap_analysis(self, X_test):
#         """Perform SHAP analysis"""
#         print("   📊 SHAP Analysis...")
        
#         # Create SHAP explainer
#         explainer = shap.TreeExplainer(self.model)
#         shap_values = explainer.shap_values(X_test)
        
#         # Summary plot
#         plt.figure(figsize=(10, 8))
#         shap.summary_plot(shap_values, X_test, feature_names=self.feature_names, show=False)
#         save_research_plot('shap_summary.png')
#         plt.show()
        
#         # Force plot for first prediction
#         plt.figure(figsize=(12, 6))
#         shap.force_plot(explainer.expected_value, shap_values[0,:], 
#                        X_test[0,:], feature_names=self.feature_names, 
#                        matplotlib=True, show=False)
#         save_research_plot('shap_force_plot.png')
#         plt.show()
    
#     def lime_analysis(self, X_test, y_test):
#         """Perform LIME analysis"""
#         print("   🍋 LIME Analysis...")
        
#         # Create LIME explainer
#         self.explainer = lime.lime_tabular.LimeTabularExplainer(
#             X_test,
#             feature_names=self.feature_names,
#             class_names=['Not Severe', 'Severe'],
#             mode='classification'
#         )
        
#         # Explain first instance
#         exp = self.explainer.explain_instance(
#             X_test[0], 
#             self.model.predict_proba, 
#             num_features=10
#         )
        
#         # Save explanation as plot
#         plt.figure(figsize=(10, 6))
#         exp.as_pyplot_figure()
#         plt.tight_layout()
#         save_research_plot('lime_explanation.png')
#         plt.show()
        
#         print("   ✅ LIME explanation generated for first test instance")
    
#     def risk_threshold_analysis(self, y_test, y_pred_proba):
#         """Analyze different risk thresholds"""
#         print("\n🎯 Risk Threshold Analysis")
        
#         thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
#         results = []
        
#         for threshold in thresholds:
#             y_pred_thresh = (y_pred_proba >= threshold).astype(int)
            
#             from sklearn.metrics import precision_score, recall_score, f1_score
            
#             results.append({
#                 'threshold': threshold,
#                 'precision': precision_score(y_test, y_pred_thresh),
#                 'recall': recall_score(y_test, y_pred_thresh),
#                 'f1': f1_score(y_test, y_pred_thresh),
#                 'positive_rate': y_pred_thresh.mean()
#             })
        
#         results_df = pd.DataFrame(results)
#         print(results_df.round(4))
        
#         return results_df

# def main():
#     """Main evaluation pipeline"""
#     from model_training import main as training_main
    
#     # Get trained model and test data
#     best_model, test_results, features = training_main()
    
#     # Load test data
#     from feature_engineering import main as feature_main
#     X_train, X_test, y_train, y_test, features = feature_main()
    
#     # Make predictions
#     y_pred = best_model.predict(X_test)
#     y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
#     # Comprehensive evaluation
#     evaluator = ModelEvaluator(best_model, features)
#     evaluator.comprehensive_evaluation(X_test, y_test, y_pred, y_pred_proba)
    
#     # Risk threshold analysis
#     threshold_results = evaluator.risk_threshold_analysis(y_test, y_pred_proba)
    
#     print(f"\n🎉 Model evaluation completed!")
#     print(f"   Best model: {type(best_model).__name__}")
#     print(f"   Test F1-Score: {test_results['flight_risk_model']['f1']:.4f}")
    
#     return evaluator, threshold_results

# if __name__ == "__main__":
#     evaluator, threshold_results = main()

# src/evaluation.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_curve, auc, precision_recall_curve, precision_score,
                           recall_score, average_precision_score)
import shap
import lime
import lime.lime_tabular
import joblib
from utils import ResearchConfig, load_model, save_research_plot
from statistical_testing import StatisticalTester
from cost_benefit_analysis import CostBenefitAnalyzer
# from uncertainty_quantification import UncertaintyQuantifier

class EnhancedModelEvaluator:
    def __init__(self, model, feature_names, uncertainty_quantifier=None):
        self.model = model
        self.feature_names = feature_names
        self.uncertainty_quantifier = uncertainty_quantifier
        self.statistical_tester = StatisticalTester()
        self.cost_benefit_analyzer = CostBenefitAnalyzer()
        self.explainer = None
        
    def comprehensive_evaluation(self, X_test, y_test, y_pred, y_pred_proba):
        """Perform comprehensive model evaluation with enhanced metrics"""
        print("📈 Enhanced Comprehensive Model Evaluation")
        print("=" * 60)
        
        # Basic classification report
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Not Severe', 'Severe']))
        
        # Enhanced metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        
        print("🎯 Enhanced Performance Metrics:")
        print(f"  • Accuracy:          {accuracy:.4f}")
        print(f"  • Precision:         {precision:.4f}")
        print(f"  • Recall:            {recall:.4f}")
        print(f"  • F1-Score:          {f1:.4f}")
        print(f"  • ROC-AUC:           {roc_auc:.4f}")
        print(f"  • Average Precision: {avg_precision:.4f}")
        
        # Confusion matrix
        self.plot_enhanced_confusion_matrix(y_test, y_pred)

        # ROC Curve (call defensively to avoid AttributeError if method missing)
        if hasattr(self, 'plot_roc_curve'):
            try:
                self.plot_roc_curve(y_test, y_pred_proba)
            except Exception as e:
                print(f"[WARN] plot_roc_curve raised an exception: {e}")
        else:
            print("[WARN] plot_roc_curve() not available on evaluator; skipping ROC plot.")

        # Precision-Recall Curve (defensive)
        if hasattr(self, 'plot_precision_recall_curve'):
            try:
                self.plot_precision_recall_curve(y_test, y_pred_proba)
            except Exception as e:
                print(f"[WARN] plot_precision_recall_curve raised an exception: {e}")
        else:
            print("[WARN] plot_precision_recall_curve() not available on evaluator; skipping PR plot.")
        
        # Feature importance
        self.plot_enhanced_feature_importance()
        
        # Cost-benefit analysis
        financial_impact = self.cost_benefit_analyzer.calculate_financial_impact(y_test, y_pred, y_pred_proba)
        print("\n💰 Financial Impact Analysis:")
        print(f"  • Net Impact: ${financial_impact['net_impact']:,.2f}")
        print(f"  • ROI: {financial_impact['roi']:.1%}")
        print(f"  • Cost per Prevented Accident: ${financial_impact['cost_per_prevented_accident']:,.2f}")
        
        # Uncertainty analysis
        if self.uncertainty_quantifier:
            self.analyze_uncertainty(X_test, y_test)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'avg_precision': avg_precision,
            'financial_impact': financial_impact
        }
    
    def plot_enhanced_confusion_matrix(self, y_test, y_pred):
        """Plot enhanced confusion matrix with percentages"""
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Not Severe', 'Severe'],
                   yticklabels=['Not Severe', 'Severe'])
        
        # Add percentage annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j+0.5, i+0.3, f"{cm_percent[i, j]:.1f}%",
                        ha="center", va="center", color="red", fontsize=12)
        
        plt.title('Enhanced Confusion Matrix (Counts with Percentages)')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        save_research_plot('enhanced_confusion_matrix.png')
        plt.show()

    def plot_roc_curve(self, y_test, y_pred_proba):
        """Plot ROC curve (enhanced)"""
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        save_research_plot('enhanced_roc_curve.png')
        plt.show()

    def plot_precision_recall_curve(self, y_test, y_pred_proba):
        """Plot Precision-Recall curve (enhanced)"""
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                 label=f'AP = {avg_precision:.4f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.legend(loc='lower left')
        save_research_plot('enhanced_precision_recall_curve.png')
        plt.show()
    
    def plot_enhanced_feature_importance(self, top_k=15):
        """Plot enhanced feature importance with confidence intervals"""
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            
            # Create feature importance dataframe
            feature_imp = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False).head(top_k)
            
            # Plot with enhanced styling
            plt.figure(figsize=(12, 8))
            bars = plt.barh(range(len(feature_imp)), feature_imp['importance'])
            plt.yticks(range(len(feature_imp)), feature_imp['feature'])
            plt.xlabel('Feature Importance')
            plt.title(f'Top {top_k} Feature Importance')
            plt.gca().invert_yaxis()
            
            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(width, bar.get_y() + bar.get_height()/2, 
                        f'{width:.4f}', ha='left', va='center')
            
            save_research_plot('enhanced_feature_importance.png')
            plt.show()
            
            print(f"\n🔝 Top {top_k} Most Important Features:")
            print(feature_imp)
    
    def analyze_uncertainty(self, X_test, y_test):
        """Analyze model uncertainty"""
        if not self.uncertainty_quantifier:
            print("⚠️ Uncertainty quantifier not available")
            return
        
        print("\n🔮 Uncertainty Analysis:")
        
        # Get predictions with uncertainty
        uncertainty_results = self.uncertainty_quantifier.predict_with_uncertainty(X_test)
        
        # Calibration analysis
        calibration = self.uncertainty_quantifier.calculate_calibration_metrics(
            y_test, uncertainty_results['mean_probability']
        )
        
        print(f"  • Expected Calibration Error: {calibration['expected_calibration_error']:.4f}")
        print(f"  • Mean Uncertainty: {np.mean(uncertainty_results['std_probability']):.4f}")
        print(f"  • High Uncertainty Predictions: {np.mean(uncertainty_results['std_probability'] > 0.2):.1%}")
        
        # Plot uncertainty distribution
        plt.figure(figsize=(10, 6))
        plt.hist(uncertainty_results['std_probability'], bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Prediction Uncertainty (Standard Deviation)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Prediction Uncertainty')
        save_research_plot('uncertainty_distribution.png')
        plt.show()
        
        # Plot calibration curve
        calibration_df = calibration['calibration_df']
        plt.figure(figsize=(8, 6))
        plt.plot(calibration_df['mean_predicted_prob'], calibration_df['actual_positive_rate'], 'o-')
        plt.plot([0, 1], [0, 1], '--', color='gray')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Actual Positive Rate')
        plt.title('Calibration Curve')
        plt.grid(True, alpha=0.3)
        save_research_plot('calibration_curve.png')
        plt.show()
    
    def risk_threshold_analysis(self, y_test, y_pred_proba):
        """Enhanced risk threshold analysis with financial impact"""
        print("\n🎯 Enhanced Risk Threshold Analysis")
        
        # Financial optimization
        optimization = self.cost_benefit_analyzer.optimize_threshold(y_test, y_pred_proba)
        optimal_threshold = optimization['optimal_threshold']
        
        print(f"💰 Financially Optimal Threshold: {optimal_threshold:.3f}")
        print(f"📈 Optimal Net Impact: ${optimization['optimal_metric_value']:,.2f}")
        
        # Plot threshold analysis
        sensitivity_df = optimization['sensitivity_analysis']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Net Impact vs Threshold
        axes[0,0].plot(sensitivity_df['threshold'], sensitivity_df['net_impact'], marker='o')
        axes[0,0].axvline(optimal_threshold, color='red', linestyle='--', alpha=0.7)
        axes[0,0].set_xlabel('Threshold')
        axes[0,0].set_ylabel('Net Impact ($)')
        axes[0,0].set_title('Net Financial Impact vs Threshold')
        axes[0,0].grid(True, alpha=0.3)
        
        # ROI vs Threshold
        axes[0,1].plot(sensitivity_df['threshold'], sensitivity_df['roi'], marker='o', color='green')
        axes[0,1].axvline(optimal_threshold, color='red', linestyle='--', alpha=0.7)
        axes[0,1].set_xlabel('Threshold')
        axes[0,1].set_ylabel('ROI')
        axes[0,1].set_title('Return on Investment vs Threshold')
        axes[0,1].grid(True, alpha=0.3)
        
        # Alert Rate vs Threshold
        axes[1,0].plot(sensitivity_df['threshold'], sensitivity_df['alert_rate'], marker='o', color='orange')
        axes[1,0].axvline(optimal_threshold, color='red', linestyle='--', alpha=0.7)
        axes[1,0].set_xlabel('Threshold')
        axes[1,0].set_ylabel('Alert Rate')
        axes[1,0].set_title('Alert Rate vs Threshold')
        axes[1,0].grid(True, alpha=0.3)
        
        # Trade-off: Precision vs Recall
        precision_vals = []
        recall_vals = []
        thresholds = np.linspace(0.1, 0.9, 50)
        
        for threshold in thresholds:
            y_pred_thresh = (y_pred_proba >= threshold).astype(int)
            precision_vals.append(precision_score(y_test, y_pred_thresh))
            recall_vals.append(recall_score(y_test, y_pred_thresh))
        
        axes[1,1].plot(recall_vals, precision_vals, marker='o')
        axes[1,1].set_xlabel('Recall')
        axes[1,1].set_ylabel('Precision')
        axes[1,1].set_title('Precision-Recall Trade-off')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_research_plot('enhanced_threshold_analysis.png')
        plt.show()
        
        return optimization
    
    def model_interpretability(self, X_test, y_test):
        """Enhanced model interpretability with multiple techniques"""
        print("\n🔍 Enhanced Model Interpretability Analysis")
        
        # SHAP Analysis (wrap in try-except for robustness)
        try:
            self.shap_analysis(X_test)
        except Exception as e:
            print(f"⚠️ SHAP analysis failed: {e}")
        
        # LIME Analysis
        try:
            self.lime_analysis(X_test, y_test)
        except Exception as e:
            print(f"⚠️ LIME analysis failed: {e}")
        
        # Partial Dependence Plots (if tree-based model)
        if hasattr(self.model, 'feature_importances_'):
            try:
                self.partial_dependence_analysis(X_test)
            except Exception as e:
                print(f"⚠️ Partial dependence analysis failed: {e}")
    
    def shap_analysis(self, X_test):
        """Perform SHAP analysis for model interpretability"""
        print("   📊 SHAP Analysis...")
        
        try:
            # Create SHAP explainer (tree-based models)
            if hasattr(self.model, 'feature_importances_'):
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(X_test[:100])  # Use first 100 samples for speed
                
                # Summary plot
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values, X_test[:100], feature_names=self.feature_names, show=False)
                save_research_plot('shap_summary.png')
                plt.close()
                
                print("   ✅ SHAP summary plot saved")
            else:
                print("   ⚠️ SHAP not supported for this model type")
        except Exception as e:
            print(f"   ❌ SHAP analysis error: {e}")
    
    def lime_analysis(self, X_test, y_test):
        """Perform LIME analysis for individual predictions"""
        print("   🍋 LIME Analysis...")
        
        try:
            # Create LIME explainer
            self.explainer = lime.lime_tabular.LimeTabularExplainer(
                X_test,
                feature_names=self.feature_names,
                class_names=['Not Severe', 'Severe'],
                mode='classification'
            )
            
            # Explain first instance
            exp = self.explainer.explain_instance(
                X_test[0], 
                self.model.predict_proba, 
                num_features=10
            )
            
            # Save explanation as plot
            plt.figure(figsize=(10, 6))
            exp.as_pyplot_figure()
            plt.tight_layout()
            save_research_plot('lime_explanation.png')
            plt.close()
            
            print("   ✅ LIME explanation generated for first test instance")
        except Exception as e:
            print(f"   ❌ LIME analysis error: {e}")
    
    def partial_dependence_analysis(self, X_test, top_features=5):
        """Partial dependence analysis for top features"""
        from sklearn.inspection import PartialDependenceDisplay
        
        
        # Get top features
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            top_indices = np.argsort(importance)[-top_features:][::-1]
            top_feature_names = [self.feature_names[i] for i in top_indices]
            
            print(f"📊 Partial Dependence Analysis for Top {top_features} Features:")
            
            # Create PDP plot
            fig, ax = plt.subplots(figsize=(12, 8))
            PartialDependenceDisplay.from_estimator(
                self.model, X_test, features=top_feature_names,
                ax=ax, grid_resolution=20
            )
            plt.title('Partial Dependence Plots for Top Features')
            plt.tight_layout()
            save_research_plot('partial_dependence_plots.png')
            plt.show()
    
    def statistical_model_comparison(self, test_results, y_test):
        """Statistical comparison of multiple models"""
        print("\n📊 Statistical Model Comparison")
        
        # Prepare model predictions for comparison
        model_predictions = {}
        for name, results in test_results.items():
            model_predictions[name] = results['predictions']
        
        # Perform comprehensive statistical testing
        statistical_comparisons = self.statistical_tester.comprehensive_model_comparison(
            test_results, test_results[list(test_results.keys())[0]]['predictions'], y_test
        )
        
        return statistical_comparisons

def main():
    """Main enhanced evaluation pipeline"""
    from model_training import main as training_main
    
    # Get trained model and test data
    best_model, test_results, features, uncertainty_quantifier = training_main()
    
    # Load test data
    from feature_engineering import main as feature_main
    X_train, X_test, y_train, y_test, features = feature_main()
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    # Comprehensive evaluation
    evaluator = EnhancedModelEvaluator(best_model, features, uncertainty_quantifier)
    evaluation_results = evaluator.comprehensive_evaluation(X_test, y_test, y_pred, y_pred_proba)
    
    # Risk threshold analysis
    threshold_results = evaluator.risk_threshold_analysis(y_test, y_pred_proba)
    
    # Model interpretability
    evaluator.model_interpretability(X_test, y_test)
    
    print(f"\n🎉 Enhanced model evaluation completed!")
    print(f"   Best model: {type(best_model).__name__}")
    print(f"   Test F1-Score: {evaluation_results['f1']:.4f}")
    print(f"   Net Financial Impact: ${evaluation_results['financial_impact']['net_impact']:,.2f}")
    
    return evaluator, evaluation_results, threshold_results

if __name__ == "__main__":
    evaluator, evaluation_results, threshold_results = main()