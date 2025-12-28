# streamlit/utils.py
"""Utility functions for the Streamlit app"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

class ResearchConfig:
    """Configuration for research constants"""
    
    MODEL_PATH = '../models/flight_risk_model.pkl'
    SCALER_PATH = '../models/feature_scaler.pkl'
    FEATURES_PATH = '../models/selected_features.pkl'
    
    RISK_THRESHOLDS = {
        'low': 0.3,
        'medium': 0.7,
        'high': 1.0
    }
    
    COST_CONFIG = {
        'false_negative_cost': 10000000,
        'false_positive_cost': 50000,
        'true_positive_benefit': 2000000,
        'operational_disruption_cost': 100000,
        'reputation_damage_cost': 500000,
        'insurance_savings': 300000
    }


def load_model(model_path=None):
    """Load the trained model"""
    if model_path is None:
        model_path = ResearchConfig.MODEL_PATH
    
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        raise ValueError(f"Error loading model from {model_path}: {e}")


def calculate_risk_level(probability):
    """Calculate risk level from probability"""
    if probability < ResearchConfig.RISK_THRESHOLDS['low']:
        return 'Low', 'success'
    elif probability < ResearchConfig.RISK_THRESHOLDS['medium']:
        return 'Medium', 'warning'
    else:
        return 'High', 'error'


def get_risk_recommendation(risk_level):
    """Get recommendation based on risk level"""
    recommendations = {
        'Low': {
            'action': 'Standard Procedures',
            'description': 'Flight appears safe. Follow standard operational procedures.',
            'details': [
                'Continue with normal pre-flight checks',
                'Monitor weather conditions',
                'Ensure all documentation is current'
            ]
        },
        'Medium': {
            'action': 'Enhanced Monitoring',
            'description': 'Moderate risk detected. Additional precautions recommended.',
            'details': [
                'Conduct additional safety briefing',
                'Review weather forecasts carefully',
                'Consider backup plans',
                'Enhanced pre-flight inspection',
                'Monitor aircraft systems closely'
            ]
        },
        'High': {
            'action': 'Preventive Action Required',
            'description': 'High risk detected. Consider postponing or canceling flight.',
            'details': [
                'Conduct thorough risk assessment',
                'Consider alternative arrangements',
                'Consult with safety officer',
                'Review all risk factors carefully',
                'Do not proceed without management approval',
                'Consider rescheduling if possible'
            ]
        }
    }
    
    return recommendations.get(risk_level, recommendations['Medium'])


class ModelOptimizer:
    """Optimize model thresholds for business objectives"""
    
    def __init__(self, cost_config=None):
        self.cost_config = cost_config or ResearchConfig.COST_CONFIG
    
    def calculate_expected_value(self, probability, threshold):
        """Calculate expected value of a decision"""
        if probability >= threshold:
            # Predict positive (preventive action)
            expected_cost = (
                self.cost_config['false_positive_cost'] +
                self.cost_config['operational_disruption_cost']
            )
            expected_benefit = (
                probability * (
                    self.cost_config['true_positive_benefit'] +
                    self.cost_config['insurance_savings'] +
                    self.cost_config['reputation_damage_cost']
                )
            )
            return expected_benefit - expected_cost
        else:
            # Predict negative (no action)
            expected_cost = probability * self.cost_config['false_negative_cost']
            return -expected_cost
    
    def optimize_threshold(self, probabilities, y_true=None):
        """Find optimal threshold based on expected value"""
        thresholds = np.linspace(0.1, 0.9, 17)
        best_threshold = 0.5
        best_value = float('-inf')
        
        for threshold in thresholds:
            total_value = sum(
                self.calculate_expected_value(prob, threshold)
                for prob in probabilities
            )
            
            if total_value > best_value:
                best_value = total_value
                best_threshold = threshold
        
        return best_threshold, best_value
    
    def get_optimal_decision(self, probability):
        """Get optimal decision for a given probability"""
        threshold = 0.5  # Default threshold
        ev_action = self.calculate_expected_value(probability, threshold)
        ev_no_action = self.calculate_expected_value(probability, 1.0)  # Never act
        
        if ev_action > ev_no_action:
            return 'Take Preventive Action', ev_action
        else:
            return 'Proceed with Caution', ev_no_action
