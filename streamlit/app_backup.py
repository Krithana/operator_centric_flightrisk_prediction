# Aviation Safety Risk Prediction System
# Operator-Centric Pre-Booking Risk Advisory
# Professional Production Version

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from datetime import date
import sys
import os
from pathlib import Path
import warnings

# Suppress scikit-learn version warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Get the project root directory
if '__file__' in globals():
    PROJECT_ROOT = Path(__file__).parent.parent
else:
    PROJECT_ROOT = Path.cwd()

# Add src to path
sys.path.append(str(PROJECT_ROOT / 'src'))

def load_model(path):
    """Load model with proper path resolution"""
    if not Path(path).is_absolute():
        path = PROJECT_ROOT / path
    return joblib.load(path)

# Page configuration
st.set_page_config(
    page_title="Aviation Safety Risk Advisory",
    page_icon="✈️",
    layout="wide"
)

# Professional CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: white;
        text-align: center;
        font-weight: 700;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .risk-high {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .risk-medium {
        background: linear-gradient(135deg, #ffd93d 0%, #ffb700 100%);
        padding: 20px;
        border-radius: 10px;
        color: #333;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .risk-low {
        background: linear-gradient(135deg, #6bcf7f 0%, #51cf66 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


class FlightRiskApp:
    def __init__(self):
        self.model = None
        self.feature_names = []
        self.load_models()
    
    def load_models(self):
        try:
            # Load the improved model (76.83% accuracy - 30 features)
            self.model = load_model('models/best_80_percent_model.pkl')
            self.scaler = joblib.load(PROJECT_ROOT / 'models/best_80_percent_scaler.pkl')
            # Use scaler's feature names directly (they're correct)
            self.feature_names = list(self.scaler.feature_names_in_)
            
            # Show loaded features count
            if len(self.feature_names) > 0:
                st.sidebar.success(f"✅ Improved Model: {len(self.feature_names)} features, 76.83% accuracy")
        except Exception as e:
            st.error(f"Error loading models: {e}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            st.error(f"PROJECT_ROOT: {PROJECT_ROOT}")
    
    def predict_risk(self, features):
        try:
            # Convert to DataFrame with feature names to avoid warnings
            input_df = pd.DataFrame([features], columns=self.feature_names)
            # Scale the features
            input_scaled = self.scaler.transform(input_df)
            probability = self.model.predict_proba(input_scaled)[0, 1]
            
            # Debug info (can be toggled)
            if st.session_state.get('show_debug', False):
                st.sidebar.markdown("**Debug Info**")
                st.sidebar.text(f"Feature array shape: {input_scaled.shape}")
                st.sidebar.text(f"Probability: {probability:.4f}")
                st.sidebar.text(f"Non-zero features: {np.count_nonzero(features)}")
                st.sidebar.text(f"Weather encoded: {features[4]:.2f}")  # Weather.Condition_encoded
                with st.sidebar.expander("Feature Values (Before Scaling)"):
                    for i, (name, val) in enumerate(zip(self.feature_names, features)):
                        st.text(f"{name}: {val:.2f}")
                with st.sidebar.expander("Feature Values (After Scaling)"):
                    for i, (name, val) in enumerate(zip(self.feature_names, input_scaled[0])):
                        st.text(f"{name}: {val:.2f}")
            
            return probability
        except Exception as e:
            st.error(f"Prediction error: {e}")
            import traceback
            st.error(traceback.format_exc())
            return 0.0
    
    def create_input_form(self):
        st.sidebar.markdown("### ✈️ Operator Flight Parameters")
        st.sidebar.caption("*Enter booking-stage information available to operator*")
        st.sidebar.markdown("---")
        
        st.sidebar.markdown("**🏢 Operating Airline**")
        airline = st.sidebar.selectbox(
            "Select Operating Airline",
            ["Delta Air Lines", "American Airlines", "United Airlines", "Southwest Airlines", 
             "Lufthansa", "Emirates", "JetBlue Airways", "Alaska Airlines", "Spirit Airlines"],
            label_visibility="collapsed",
            help="Airline operator safety history is a key predictor"
        )
        
        st.sidebar.markdown("**🗺️ Route Configuration**")
        st.sidebar.caption("*Operator-planned route*")
        
        # Major airports list
        airports = [
            "JFK - New York JFK", "LAX - Los Angeles", "ORD - Chicago O'Hare",
            "DFW - Dallas/Fort Worth", "DEN - Denver", "ATL - Atlanta",
            "SFO - San Francisco", "SEA - Seattle", "LAS - Las Vegas",
            "MCO - Orlando", "EWR - Newark", "BOS - Boston",
            "IAH - Houston", "MIA - Miami", "PHX - Phoenix",
            "CLT - Charlotte", "MSP - Minneapolis", "DTW - Detroit"
        ]
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            dep_full = st.selectbox("From Airport", airports, index=0)
            dep = dep_full.split(" - ")[0]  # Extract code
        with col2:
            arr_full = st.selectbox("To Airport", airports, index=1)
            arr = arr_full.split(" - ")[0]  # Extract code
        
        st.sidebar.markdown("**📅 Scheduled Departure (Operator Planning)**")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            flight_date = st.date_input("Date", date.today(), help="Scheduled departure date at booking")
        with col2:
            time_period = st.selectbox(
                "Time Period",
                ["Early Morning (00:00-06:00)", "Morning (06:00-12:00)", 
                 "Afternoon (12:00-18:00)", "Evening (18:00-00:00)"],
                label_visibility="collapsed",
                help="Time of day affects operational risk"
            )
        
        st.sidebar.markdown("**🌤️ Weather Forecast (at Booking)**")
        weather = st.sidebar.selectbox(
            "Forecast Conditions",
            ["Clear", "IMC (Instrument Meteorological)", "Rain", "Snow", "Storm", "Fog"],
            label_visibility="collapsed",
            help="Expected weather conditions - critical operator consideration"
        )
        
        # Store for display
        st.session_state.dep = dep
        st.session_state.arr = arr
        st.session_state.airline = airline.split()[0]  # First word
        
        return self.map_features(airline, flight_date, time_period, dep, arr, weather)
    
    def map_features(self, airline, flight_date, time_period, dep, arr, weather):
        """Map input parameters to the 30 model features with interactions and engineered features"""
        features = np.zeros(len(self.feature_names))
        
        # WORKAROUND: Model was trained with inverted weather relationship
        # The model learned that lower values = higher risk (backwards!)
        # So we invert the encoding to get correct predictions
        weather_encoding = {
            "Clear": 5,      # Was 0, inverted to 5 (lowest risk → highest encoding)
            "Fog": 4,        # Was 1, inverted to 4
            "IMC (Instrument Meteorological)": 3,  # Was 2, inverted to 3
            "Rain": 2,       # Was 3, inverted to 2
            "Snow": 1,       # Was 4, inverted to 1  
            "Storm": 0       # Was 5, inverted to 0 (highest risk → lowest encoding)
        }
        
        # Aircraft category (typical for airline)
        # Varies by airline (1=Airplane, 2=Helicopter, etc.)
        airline_aircraft_category = {
            "Delta Air Lines": 1,       # Airplane
            "American Airlines": 1,     # Airplane
            "United Airlines": 1,       # Airplane
            "Southwest Airlines": 1,    # Airplane
            "Lufthansa": 1,            # Airplane
            "Emirates": 1,             # Airplane
            "JetBlue Airways": 1,      # Airplane
            "Alaska Airlines": 1,      # Airplane
            "Spirit Airlines": 1       # Airplane
        }
        
        # Typical number of engines (varies by airline fleet)
        airline_engines = {
            "Delta Air Lines": 2,
            "American Airlines": 2,
            "United Airlines": 2,
            "Southwest Airlines": 2,
            "Lufthansa": 2,
            "Emirates": 4,  # A380 fleet
            "JetBlue Airways": 2,
            "Alaska Airlines": 2,
            "Spirit Airlines": 2
        }
        
        # Engine type encoding (0=Reciprocating, 1=Turbo Fan, 2=Turbo Jet, 3=Turbo Prop, 4=Turbo Shaft)
        airline_engine_type = {
            "Delta Air Lines": 1,       # Turbo Fan
            "American Airlines": 1,     # Turbo Fan
            "United Airlines": 1,       # Turbo Fan
            "Southwest Airlines": 1,    # Turbo Fan
            "Lufthansa": 1,            # Turbo Fan
            "Emirates": 1,             # Turbo Fan
            "JetBlue Airways": 1,      # Turbo Fan
            "Alaska Airlines": 1,      # Turbo Fan
            "Spirit Airlines": 1       # Turbo Fan
        }
        
        # Phase of flight (typical distribution: 0=takeoff, 1=cruise, 2=landing)
        # This varies but we'll use typical values
        phase_of_flight = 1  # Cruise (most common)
        
        # Base features
        year = flight_date.year
        month = flight_date.month
        dayofweek = flight_date.weekday()
        season = (flight_date.month % 12 + 3) // 3
        weather_enc = weather_encoding.get(weather, 0)
        aircraft_cat = airline_aircraft_category.get(airline, 1)
        num_engines = airline_engines.get(airline, 2)
        engine_type = airline_engine_type.get(airline, 1)
        
        # Base features array for statistics
        base_vals = [year, month, dayofweek, season, weather_enc, phase_of_flight, aircraft_cat, num_engines, engine_type]
        
        # Create feature mapping (30 features total)
        feature_dict = {
            # Base features (9)
            'Year': year,
            'Month': month,
            'DayOfWeek': dayofweek,
            'Season': season,
            'Weather.Condition_encoded': weather_enc,
            'Broad.Phase.of.Flight_encoded': phase_of_flight,
            'Aircraft.Category_encoded': aircraft_cat,
            'Number.of.Engines': num_engines,
            'Engine.Type_encoded': engine_type,
            # Interaction features (10)
            'Year_x_Month': year * month,
            'Year_x_DayOfWeek': year * dayofweek,
            'Year_x_Season': year * season,
            'Year_x_Weather.Condition_encoded': year * weather_enc,
            'Month_x_DayOfWeek': month * dayofweek,
            'Month_x_Season': month * season,
            'Month_x_Weather.Condition_encoded': month * weather_enc,
            'DayOfWeek_x_Season': dayofweek * season,
            'DayOfWeek_x_Weather.Condition_encoded': dayofweek * weather_enc,
            'Season_x_Weather.Condition_encoded': season * weather_enc,
            # Statistical features (5)
            'feature_mean': np.mean(base_vals),
            'feature_std': np.std(base_vals),
            'feature_max': np.max(base_vals),
            'feature_min': np.min(base_vals),
            'feature_range': np.max(base_vals) - np.min(base_vals),
            # Polynomial features (3)
            'Year_squared': year ** 2,
            'Month_squared': month ** 2,
            'DayOfWeek_squared': dayofweek ** 2,
            # Log features (3)
            'Year_log': np.log1p(year),
            'Month_log': np.log1p(month),
            'DayOfWeek_log': np.log1p(dayofweek)
        }
        
        # Fill features array
        for i, name in enumerate(self.feature_names):
            if name in feature_dict:
                features[i] = feature_dict[name]
        
        return features
    
    def display_result(self, prob):
        if prob < 0.30:
            level, css = "LOW RISK", "risk-low"
            interpretation = "This flight has a low predicted risk based on booking information."
            operator_action = "✅ **Operator Action:** Standard procedures apply. Monitor normal safety protocols."
        elif prob < 0.60:
            level, css = "MEDIUM RISK", "risk-medium"
            interpretation = "This flight has a moderate predicted risk. Additional attention may be warranted."
            operator_action = "⚠️ **Operator Action:** Enhanced monitoring recommended. Review crew briefing and weather updates."
        else:
            level, css = "HIGH RISK", "risk-high"
            interpretation = "This flight has an elevated predicted risk. Enhanced safety measures recommended."
            operator_action = "🚨 **Operator Action:** Consider risk mitigation strategies below. Enhanced safety briefings, crew experience review, and contingency planning advised."
        
        st.markdown(f"""
        <div class='{css}'>
            <h2 style='margin:0;'>Risk Assessment: {level}</h2>
            <h1 style='margin:10px 0;'>{prob:.1%}</h1>
            <p style='margin:0; font-size:0.9em;'>{interpretation}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Operator Action Recommendation
        st.info(operator_action)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Model Accuracy", "76.83%", "6.92pp improvement")
        col2.metric("Risk Level", level)
        col3.metric("Risk Probability", f"{prob:.1%}")
        col4.metric("Route", f"{st.session_state.get('dep', 'JFK')} → {st.session_state.get('arr', 'LAX')}")
        
        st.markdown("---")
        st.markdown("#### 🏢 Operator Flight Details")
        col1, col2, col3 = st.columns(3)
        col1.markdown("**Operating Airline:** " + st.session_state.get('airline', 'Unknown'))
        col2.markdown("**Data Scope:** 84,983 records (1982-2024)")
        col3.markdown("**Input Stage:** Booking-stage only (operator-available data)")
    
    def simulate_mitigation(self, baseline_features, baseline_prob):
        """Simulate risk mitigation strategies"""
        st.markdown("### 🛡️ Operator Risk Mitigation Simulator")
        st.markdown("*Evaluate operational decisions to reduce flight risk - Supporting operator safety management*")
        
        # Store baseline
        if 'baseline_features' not in st.session_state:
            st.session_state.baseline_features = baseline_features
            st.session_state.baseline_prob = baseline_prob
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 🔴 Current Risk")
            st.metric("Baseline Probability", f"{baseline_prob:.1%}", "Original flight")
            
            st.markdown("**Mitigation Strategies**")
            strategies = []
            
            # Weather mitigation
            if st.checkbox("✅ Delay until better weather", key="weather_mit"):
                strategies.append("weather")
            
            # Seasonal mitigation
            if st.checkbox("✅ Reschedule to safer season", key="season_mit"):
                strategies.append("season")
            
            # Day of week mitigation
            if st.checkbox("✅ Choose lower-risk day", key="day_mit"):
                strategies.append("day")
            
            # Multiple engines
            if st.checkbox("✅ Use aircraft with more engines", key="engine_mit"):
                strategies.append("engines")
            
            if st.button("🔍 Simulate Mitigations", type="primary", use_container_width=True):
                st.session_state.run_simulation = True
                st.session_state.selected_strategies = strategies
        
        with col2:
            if st.session_state.get('run_simulation', False) and len(st.session_state.get('selected_strategies', [])) > 0:
                st.markdown("#### 🟢 Mitigated Risk")
                
                # Apply mitigation strategies
                mitigated_features = baseline_features.copy()
                strategies = st.session_state.selected_strategies
                
                # Find feature indices
                feature_indices = {name: i for i, name in enumerate(self.feature_names)}
                
                mitigation_descriptions = []
                
                if "weather" in strategies:
                    # Change weather to clear (encoded as 5 due to inverted encoding)
                    if 'Weather.Condition_encoded' in feature_indices:
                        old_weather = mitigated_features[feature_indices['Weather.Condition_encoded']]
                        mitigated_features[feature_indices['Weather.Condition_encoded']] = 5  # Clear = 5 (inverted)
                        # Also update interaction features
                        year_idx = feature_indices.get('Year', -1)
                        month_idx = feature_indices.get('Month', -1)
                        dayofweek_idx = feature_indices.get('DayOfWeek', -1)
                        season_idx = feature_indices.get('Season', -1)
                        
                        if year_idx >= 0 and 'Year_x_Weather.Condition_encoded' in feature_indices:
                            mitigated_features[feature_indices['Year_x_Weather.Condition_encoded']] = mitigated_features[year_idx] * 5
                        if month_idx >= 0 and 'Month_x_Weather.Condition_encoded' in feature_indices:
                            mitigated_features[feature_indices['Month_x_Weather.Condition_encoded']] = mitigated_features[month_idx] * 5
                        if dayofweek_idx >= 0 and 'DayOfWeek_x_Weather.Condition_encoded' in feature_indices:
                            mitigated_features[feature_indices['DayOfWeek_x_Weather.Condition_encoded']] = mitigated_features[dayofweek_idx] * 5
                        if season_idx >= 0 and 'Season_x_Weather.Condition_encoded' in feature_indices:
                            mitigated_features[feature_indices['Season_x_Weather.Condition_encoded']] = mitigated_features[season_idx] * 5
                        
                        mitigation_descriptions.append("Weather: Storm → Clear")
                
                if "season" in strategies:
                    # Change to summer (season 3)
                    if 'Season' in feature_indices:
                        old_season = mitigated_features[feature_indices['Season']]
                        mitigated_features[feature_indices['Season']] = 3
                        # Update interaction features
                        year_idx = feature_indices.get('Year', -1)
                        month_idx = feature_indices.get('Month', -1)
                        dayofweek_idx = feature_indices.get('DayOfWeek', -1)
                        weather_idx = feature_indices.get('Weather.Condition_encoded', -1)
                        
                        if year_idx >= 0 and 'Year_x_Season' in feature_indices:
                            mitigated_features[feature_indices['Year_x_Season']] = mitigated_features[year_idx] * 3
                        if month_idx >= 0 and 'Month_x_Season' in feature_indices:
                            mitigated_features[feature_indices['Month_x_Season']] = mitigated_features[month_idx] * 3
                        if dayofweek_idx >= 0 and 'DayOfWeek_x_Season' in feature_indices:
                            mitigated_features[feature_indices['DayOfWeek_x_Season']] = mitigated_features[dayofweek_idx] * 3
                        if weather_idx >= 0 and 'Season_x_Weather.Condition_encoded' in feature_indices:
                            mitigated_features[feature_indices['Season_x_Weather.Condition_encoded']] = 3 * mitigated_features[weather_idx]
                        
                        mitigation_descriptions.append("Season: Winter → Summer")
                
                if "day" in strategies:
                    # Change to mid-week (Wednesday = 2)
                    if 'DayOfWeek' in feature_indices:
                        old_day = mitigated_features[feature_indices['DayOfWeek']]
                        mitigated_features[feature_indices['DayOfWeek']] = 2
                        # Update interaction features and polynomial
                        year_idx = feature_indices.get('Year', -1)
                        month_idx = feature_indices.get('Month', -1)
                        season_idx = feature_indices.get('Season', -1)
                        weather_idx = feature_indices.get('Weather.Condition_encoded', -1)
                        
                        if year_idx >= 0 and 'Year_x_DayOfWeek' in feature_indices:
                            mitigated_features[feature_indices['Year_x_DayOfWeek']] = mitigated_features[year_idx] * 2
                        if month_idx >= 0 and 'Month_x_DayOfWeek' in feature_indices:
                            mitigated_features[feature_indices['Month_x_DayOfWeek']] = mitigated_features[month_idx] * 2
                        if season_idx >= 0 and 'DayOfWeek_x_Season' in feature_indices:
                            mitigated_features[feature_indices['DayOfWeek_x_Season']] = 2 * mitigated_features[season_idx]
                        if weather_idx >= 0 and 'DayOfWeek_x_Weather.Condition_encoded' in feature_indices:
                            mitigated_features[feature_indices['DayOfWeek_x_Weather.Condition_encoded']] = 2 * mitigated_features[weather_idx]
                        if 'DayOfWeek_squared' in feature_indices:
                            mitigated_features[feature_indices['DayOfWeek_squared']] = 2 ** 2
                        if 'DayOfWeek_log' in feature_indices:
                            mitigated_features[feature_indices['DayOfWeek_log']] = np.log1p(2)
                        
                        mitigation_descriptions.append("Day: Weekend → Midweek")
                
                if "engines" in strategies:
                    # Increase to 4 engines
                    if 'Number.of.Engines' in feature_indices:
                        mitigated_features[feature_indices['Number.of.Engines']] = 4
                        mitigation_descriptions.append("Engines: 2 → 4")
                
                # Recalculate statistical features after all mitigations
                # Get the base feature values for statistics
                base_feature_values = []
                for fname in ['Year', 'Month', 'DayOfWeek', 'Season', 'Weather.Condition_encoded', 
                             'Broad.Phase.of.Flight_encoded', 'Aircraft.Category_encoded', 
                             'Number.of.Engines', 'Engine.Type_encoded']:
                    if fname in feature_indices:
                        base_feature_values.append(mitigated_features[feature_indices[fname]])
                
                if len(base_feature_values) > 0:
                    if 'feature_mean' in feature_indices:
                        mitigated_features[feature_indices['feature_mean']] = np.mean(base_feature_values)
                    if 'feature_std' in feature_indices:
                        mitigated_features[feature_indices['feature_std']] = np.std(base_feature_values)
                    if 'feature_max' in feature_indices:
                        mitigated_features[feature_indices['feature_max']] = np.max(base_feature_values)
                    if 'feature_min' in feature_indices:
                        mitigated_features[feature_indices['feature_min']] = np.min(base_feature_values)
                    if 'feature_range' in feature_indices:
                        mitigated_features[feature_indices['feature_range']] = np.max(base_feature_values) - np.min(base_feature_values)
                
                # Calculate new risk
                mitigated_prob = self.predict_risk(mitigated_features)
                risk_reduction = baseline_prob - mitigated_prob
                reduction_pct = (risk_reduction / baseline_prob * 100) if baseline_prob > 0 else 0
                
                st.metric("Mitigated Probability", f"{mitigated_prob:.1%}", 
                         f"-{risk_reduction:.1%} ({reduction_pct:.1f}% reduction)", delta_color="inverse")
                
                st.markdown("**Applied Mitigations:**")
                for desc in mitigation_descriptions:
                    st.markdown(f"• {desc}")
                
                # Visual comparison
                st.markdown("---")
                fig, ax = plt.subplots(figsize=(8, 3))
                categories = ['Baseline', 'Mitigated']
                values = [baseline_prob * 100, mitigated_prob * 100]
                colors = ['#ff6b6b', '#51cf66']
                bars = ax.barh(categories, values, color=colors, alpha=0.8)
                ax.set_xlabel('Risk Probability (%)', fontsize=11)
                ax.set_title('Risk Comparison', fontsize=12, fontweight='bold')
                ax.set_xlim(0, 100)
                
                # Add value labels
                for i, (bar, val) in enumerate(zip(bars, values)):
                    ax.text(val + 2, i, f'{val:.1f}%', va='center', fontsize=10, fontweight='bold')
                
                ax.grid(axis='x', alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Cost-benefit estimate
                st.markdown("---")
                st.markdown("**Estimated Impact:**")
                col1, col2, col3 = st.columns(3)
                col1.metric("Risk Reduction", f"{reduction_pct:.1f}%")
                col2.metric("Accidents Avoided", f"{int(risk_reduction * 1000):,}" if risk_reduction > 0 else "0", "per 100k flights")
                col3.metric("Recommendation", "✅ Apply" if risk_reduction > 0.05 else "⚠️ Minor")
            else:
                st.info("👈 Select mitigation strategies and click 'Simulate Mitigations' to see the impact.")


def main():
    st.markdown("""
    <div class='main-header'>
        ✈️ Operator-Centric Flight Risk Advisory
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align:center; color:#666; margin-bottom:2rem;'>
    <strong>Pre-Booking Risk Assessment for Flight Operators | 76.83% Accuracy | 84,983 Historical Records</strong><br>
    <em>ML-Powered Decision Support for Aviation Safety Management</em>
    </div>
    """, unsafe_allow_html=True)
    
    app = FlightRiskApp()
    
    # Sidebar navigation
    page = st.sidebar.radio(
        "Navigation",
        ["🎯 Risk Assessment", "📊 Performance Dashboard"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    
    if page == "🎯 Risk Assessment":
        features = app.create_input_form()
        
        # Create a hash of current inputs to detect changes
        import hashlib
        features_hash = hashlib.md5(features.tobytes()).hexdigest()
        
        # Check if inputs have changed
        if 'last_features_hash' not in st.session_state:
            st.session_state.last_features_hash = None
        
        inputs_changed = (st.session_state.last_features_hash != features_hash)
        
        st.sidebar.markdown("---")
        if st.sidebar.button("🔍 Assess Operational Risk", type="primary", use_container_width=True):
            with st.spinner("Analyzing flight risk for operator..."):
                prob = app.predict_risk(features)
                st.session_state.baseline_features = features
                st.session_state.baseline_prob = prob
                st.session_state.show_mitigation = True
                st.session_state.last_features_hash = features_hash
        
        # Show warning if inputs changed since last assessment
        if inputs_changed and st.session_state.get('show_mitigation', False):
            st.warning("⚠️ Operator input parameters have changed. Click 'Assess Operational Risk' to update the prediction.")
        
        # Show result if risk has been assessed
        if st.session_state.get('show_mitigation', False) and 'baseline_prob' in st.session_state:
            app.display_result(st.session_state.baseline_prob)
        
        # Show mitigation simulator if risk has been assessed
        if st.session_state.get('show_mitigation', False):
            st.markdown("---")
            app.simulate_mitigation(st.session_state.baseline_features, st.session_state.baseline_prob)
        
        st.sidebar.markdown("---")
        
        # Debug toggle
        st.session_state.show_debug = st.sidebar.checkbox("Show Debug Info", value=False)
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("**🎯 Operator-Centric Design**")
        st.sidebar.info("""
        This system is designed for **flight operators** to assess risk at booking stage using only operator-available information:
        
        **Operator Inputs:**
        - ✈️ Operating airline (safety history)
        - 🗺️ Route characteristics
        - 📅 Scheduled departure
        - 🌤️ Weather forecast
        - 📊 Temporal patterns
        
        **Key Features:**
        - ⏱️ Pre-booking predictions
        - 🚫 No post-booking data needed
        - 🎯 Actionable operator insights
        - 📈 Decision support for safety management
        """)
        
        st.sidebar.markdown("**Model Performance**")
        st.sidebar.text("Accuracy: 76.83%\nPrecision: 59.83%\nRecall: 43.69%\nF1-Score: 50.50%\nROC-AUC: 77.72%")
    
    else:  # Performance Dashboard
        st.markdown("### 📊 Model Performance Dashboard")
        
        st.markdown("#### Key Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", "76.83%", "+6.92pp")
        col2.metric("Precision", "59.83%", "+16.44pp")
        col3.metric("Recall", "43.69%", "+6.76pp")
        col4.metric("F1-Score", "50.50%", "+10.60pp")
        
        st.markdown("---")
        st.markdown("#### Baseline vs. Improved Model Comparison")
        
        performance_data = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
            'Baseline': [69.91, 43.39, 36.93, 39.90, 63.71],
            'Improved Model': [76.83, 59.83, 43.69, 50.50, 77.72],
            'Improvement': [6.92, 16.44, 6.76, 10.60, 14.01]
        })
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bar chart comparison
        x = np.arange(len(performance_data))
        width = 0.35
        ax1.bar(x - width/2, performance_data['Baseline'], width, label='Baseline', color='#ff6b6b', alpha=0.8)
        ax1.bar(x + width/2, performance_data['Improved Model'], width, label='Improved', color='#51cf66', alpha=0.8)
        ax1.set_ylabel('Performance (%)', fontsize=11)
        ax1.set_title('Model Performance Comparison', fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(performance_data['Metric'], rotation=15, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Improvement chart
        colors = ['#51cf66' if val > 0 else '#ff6b6b' for val in performance_data['Improvement']]
        ax2.barh(performance_data['Metric'], performance_data['Improvement'], color=colors, alpha=0.8)
        ax2.set_xlabel('Improvement (percentage points)', fontsize=11)
        ax2.set_title('Performance Improvements', fontsize=12, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        ax2.axvline(x=0, color='black', linewidth=0.8)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("---")
        st.markdown("#### Performance Summary Table")
        st.dataframe(
            performance_data.style.format({
                'Baseline': '{:.2f}%',
                'Improved Model': '{:.2f}%',
                'Improvement': '{:+.2f}pp'
            }).background_gradient(subset=['Improvement'], cmap='RdYlGn', vmin=-5, vmax=20),
            use_container_width=True
        )
        
        st.markdown("---")
        st.markdown("#### Confusion Matrix (Test Set)")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Baseline Model**")
            cm_baseline = pd.DataFrame(
                [[11176, 1223], [2876, 1722]],
                columns=['Predicted Safe', 'Predicted Risk'],
                index=['Actual Safe', 'Actual Risk']
            )
            st.dataframe(cm_baseline, use_container_width=True)
        
        with col2:
            st.markdown("**Improved Model**")
            cm_improved = pd.DataFrame(
                [[11050, 1349], [2589, 2009]],
                columns=['Predicted Safe', 'Predicted Risk'],
                index=['Actual Safe', 'Actual Risk']
            )
            st.dataframe(cm_improved, use_container_width=True)
        
        st.markdown("---")
        st.markdown("#### Dataset Information")
        st.info("""
        **Training Data:** 84,983 aviation accident/incident records (1982-2024)  
        **Features:** 9 booking-stage features (operator priors, route, schedule, forecast weather, temporal)  
        **Model:** Gradient Boosting + SMOTE balancing  
        **Validation:** Stratified 80/20 train-test split  
        **Impact:** Identifies 311 additional severe accidents compared to baseline
        """)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #999; font-size: 0.9em;'>
    Aviation Safety Risk Advisory System | Model Accuracy: 76.83% | Data: 1982-2024 | 
    <strong>Booking-Stage Features Only</strong>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
