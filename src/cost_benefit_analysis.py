# src/cost_benefit_analysis.py
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

class CostBenefitAnalyzer:
    """Comprehensive cost-benefit analysis for aviation safety"""
    
    def __init__(self, cost_config=None):
        self.cost_config = cost_config or {
            'false_negative_cost': 10000000,  # Cost of missed severe accident
            'false_positive_cost': 50000,     # Cost of unnecessary prevention measures
            'true_positive_benefit': 2000000, # Benefit of prevented accident
            'operational_disruption_cost': 100000,  # Cost of flight cancellation
            'reputation_damage_cost': 500000, # Cost of reputation damage
            'insurance_savings': 300000       # Insurance savings per prevented accident
        }
    
    def calculate_financial_impact(self, y_true, y_pred, y_proba=None, threshold=0.5):
        """Calculate comprehensive financial impact"""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calculate costs and benefits
        fn_cost = fn * self.cost_config['false_negative_cost']
        fp_cost = fp * self.cost_config['false_positive_cost']
        tp_benefit = tp * (self.cost_config['true_positive_benefit'] + 
                          self.cost_config['insurance_savings'])
        operational_cost = (tp + fp) * self.cost_config['operational_disruption_cost']
        reputation_savings = tp * self.cost_config['reputation_damage_cost']
        
        total_cost = fn_cost + fp_cost + operational_cost
        total_benefit = tp_benefit + reputation_savings
        net_impact = total_benefit - total_cost
        
        # Calculate ROI
        total_investment = operational_cost + fp_cost
        roi = (total_benefit - total_investment) / total_investment if total_investment > 0 else float('inf')
        
        return {
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'true_negatives': tn,
            'false_negative_cost': fn_cost,
            'false_positive_cost': fp_cost,
            'true_positive_benefit': tp_benefit,
            'operational_cost': operational_cost,
            'reputation_savings': reputation_savings,
            'total_cost': total_cost,
            'total_benefit': total_benefit,
            'net_impact': net_impact,
            'roi': roi,
            'cost_per_prevented_accident': total_investment / tp if tp > 0 else float('inf'),
            'break_even_prevention_rate': self.calculate_break_even_rate()
        }
    
    def calculate_break_even_rate(self):
        """Calculate break-even prevention rate"""
        prevention_cost = (self.cost_config['false_positive_cost'] + 
                          self.cost_config['operational_disruption_cost'])
        accident_cost = (self.cost_config['false_negative_cost'] - 
                        self.cost_config['true_positive_benefit'] - 
                        self.cost_config['insurance_savings'])
        
        break_even_rate = prevention_cost / accident_cost
        return break_even_rate
    
    def threshold_sensitivity_analysis(self, y_true, y_proba, thresholds=None):
        """Analyze financial impact across different thresholds"""
        if thresholds is None:
            thresholds = np.linspace(0.1, 0.9, 17)
        
        results = []
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            financial_impact = self.calculate_financial_impact(y_true, y_pred)
            financial_impact['threshold'] = threshold
            financial_impact['alert_rate'] = (financial_impact['true_positives'] + 
                                             financial_impact['false_positives']) / len(y_true)
            results.append(financial_impact)
        
        return pd.DataFrame(results)
    
    def optimize_threshold(self, y_true, y_proba, metric='net_impact'):
        """Find optimal threshold based on financial metrics"""
        sensitivity_df = self.threshold_sensitivity_analysis(y_true, y_proba)
        optimal_idx = sensitivity_df[metric].idxmax()
        optimal_threshold = sensitivity_df.loc[optimal_idx, 'threshold']
        
        return {
            'optimal_threshold': optimal_threshold,
            'optimal_metric_value': sensitivity_df.loc[optimal_idx, metric],
            'sensitivity_analysis': sensitivity_df
        }
    
    def risk_stratification_analysis(self, y_proba, n_strata=5):
        """Analyze risk stratification for decision support"""
        strata_edges = np.linspace(0, 1, n_strata + 1)
        strata_info = []
        
        for i in range(n_strata):
            low = strata_edges[i]
            high = strata_edges[i + 1]
            mask = (y_proba >= low) & (y_proba < high)
            count = mask.sum()
            proportion = count / len(y_proba)
            
            strata_info.append({
                'stratum': i + 1,
                'risk_range': f'{low:.2f}-{high:.2f}',
                'count': count,
                'proportion': proportion,
                'recommended_action': self.get_stratum_action(i, n_strata)
            })
        
        return pd.DataFrame(strata_info)
    
    def get_stratum_action(self, stratum_idx, total_strata):
        """Get recommended action for each risk stratum"""
        if stratum_idx == 0:
            return "Standard procedures"
        elif stratum_idx == 1:
            return "Enhanced monitoring"
        elif stratum_idx == 2:
            return "Additional safety checks"
        elif stratum_idx == 3:
            return "Consider alternatives"
        else:
            return "Preventive action required"