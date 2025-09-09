import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, time
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Food Delivery Demand Predictor",
    page_icon="🍕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4ECDC4;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin: 1rem 0;
    }
    .prediction-number {
        font-size: 3rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #FF6B6B;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model and feature columns"""
    try:
        model_assets = joblib.load("hyperlocal_demand_predictor.pkl")
        return model_assets['model'], model_assets['columns']
    except FileNotFoundError:
        st.error("❌ Model file not found. Please ensure 'hyperlocal_demand_predictor.pkl' is in the same directory.")
        return None, None

def create_feature_vector(hour, city, zone_id, zone_type, weather_condition, feature_columns):
    """Create a feature vector from user inputs"""
    # Create a dataframe with user inputs
    input_data = pd.DataFrame({
        'hour': [hour],
        'city': [city],
        'zone_id': [zone_id],
        'zone_type': [zone_type],
        'weather_condition': [weather_condition]
    })
    
    # One-hot encode the categorical variables
    input_encoded = pd.get_dummies(input_data, columns=['city', 'zone_id', 'zone_type', 'weather_condition'], drop_first=True)
    
    # Create a dataframe with all feature columns initialized to 0
    feature_vector = pd.DataFrame(0, index=[0], columns=feature_columns)
    
    # Set the values for columns that exist in our input
    for col in input_encoded.columns:
        if col in feature_vector.columns:
            feature_vector[col] = input_encoded[col].values[0]
    
    return feature_vector

def main():
    # Header
    st.markdown('<h1 class="main-header">🍕 Food Delivery Demand Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Predict hourly food delivery orders for your business</p>', unsafe_allow_html=True)
    
    # Load model
    model, feature_columns = load_model()
    
    if model is None:
        st.stop()
    
    # Sidebar for inputs
    st.sidebar.markdown('<h2 class="sub-header">📊 Input Parameters</h2>', unsafe_allow_html=True)
    
    # Hour input
    hour = st.sidebar.slider("Hour of Day (24-hour format)", 0, 23, 12, help="Select the hour for prediction (0-23)")
    
    # City selection
    cities = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Hyderabad', 'Kolkata', 'Pune', 'Ahmedabad']
    city = st.sidebar.selectbox("City", cities, help="Select the city")
    
    # Zone ID - Updated with actual zone names from your dataset
    zone_options = [
        'Andheri',
        'Bandra', 
        'Connaught Place',
        'Hauz Khas',
        'Indiranagar',
        'Jayanagar',
        'Koramangala',
        'Powai',
        'Saket'
    ]
    zone_id = st.sidebar.selectbox("Zone ID", zone_options, help="Select the delivery zone")
    
    # Zone Type - Updated with actual zone types from your dataset
    zone_types = ['Commercial', 'Mixed', 'Residential', 'Student Area']
    zone_type = st.sidebar.selectbox("Zone Type", zone_types, help="Type of area")
    
    # Weather Condition - Updated with actual weather conditions from your dataset
    weather_conditions = ['Clear', 'Extreme Heat', 'Rainy']
    weather_condition = st.sidebar.selectbox("Weather Condition", weather_conditions, help="Current weather condition")
    
    # Prediction button
    predict_button = st.sidebar.button("🚀 Predict Demand", type="primary", use_container_width=True)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if predict_button:
            with st.spinner("Predicting demand..."):
                try:
                    # Create feature vector
                    feature_vector = create_feature_vector(hour, city, zone_id, zone_type, weather_condition, feature_columns)
                    
                    # Make prediction
                    prediction = model.predict(feature_vector)[0]
                    prediction = max(0, round(prediction))  # Ensure non-negative integer
                    
                    # Display prediction
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h2>📈 Predicted Orders</h2>
                        <div class="prediction-number">{prediction}</div>
                        <p>Expected orders for the next hour</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Additional insights
                    col_insight1, col_insight2, col_insight3 = st.columns(3)
                    
                    with col_insight1:
                        demand_level = "High" if prediction > 150 else "Medium" if prediction > 75 else "Low"
                        color = "#FF6B6B" if demand_level == "High" else "#FFD93D" if demand_level == "Medium" else "#6BCF7F"
                        st.markdown(f"""
                        <div style="text-align: center; padding: 1rem; background-color: {color}20; border-radius: 10px;">
                            <h3 style="color: {color};">Demand Level</h3>
                            <h2 style="color: {color};">{demand_level}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_insight2:
                        drivers_needed = max(1, round(prediction / 8))  # Assuming 8 orders per driver per hour
                        st.markdown(f"""
                        <div style="text-align: center; padding: 1rem; background-color: #4ECDC420; border-radius: 10px;">
                            <h3 style="color: #4ECDC4;">Drivers Needed</h3>
                            <h2 style="color: #4ECDC4;">{drivers_needed}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_insight3:
                        prep_time = "30 min" if demand_level == "High" else "45 min" if demand_level == "Medium" else "15 min"
                        st.markdown(f"""
                        <div style="text-align: center; padding: 1rem; background-color: #A8E6CF20; border-radius: 10px;">
                            <h3 style="color: #45B7D1;">Prep Time</h3>
                            <h2 style="color: #45B7D1;">{prep_time}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")
        
        # Information section
        st.markdown('<h2 class="sub-header">ℹ️ How to Use</h2>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        <b>Step 1:</b> Select the parameters in the sidebar (hour, city, zone, weather)<br>
        <b>Step 2:</b> Click "Predict Demand" to get the forecast<br>
        <b>Step 3:</b> Use the insights to optimize your operations
        </div>
        """, unsafe_allow_html=True)
        
        # Business insights
        st.markdown('<h2 class="sub-header">💡 Business Insights</h2>', unsafe_allow_html=True)
        
        insights_col1, insights_col2 = st.columns(2)
        
        with insights_col1:
            st.markdown("""
            **🕐 Peak Hours Optimization**
            - Lunch: 12-14 hours
            - Dinner: 19-21 hours
            - Late night: 22-24 hours
            
            **🌤️ Weather Impact**
            - Rain increases orders by ~20%
            - Clear weather: steady demand
            - Storms: significant increase
            """)
        
        with insights_col2:
            st.markdown("""
            **🏙️ Zone Strategy**
            - Metropolitan zones: higher base demand
            - Urban zones: lunch rush focused
            - Suburban zones: dinner focused
            
            **📊 Resource Planning**
            - High demand: Scale drivers & kitchens
            - Medium demand: Standard operations
            - Low demand: Reduce operational costs
            """)
    
    with col2:
        # Current parameters display
        st.markdown('<h3 class="sub-header">📋 Current Parameters</h3>', unsafe_allow_html=True)
        st.info(f"""
        **Hour:** {hour}:00  
        **City:** {city}  
        **Zone:** {zone_id}  
        **Zone Type:** {zone_type}  
        **Weather:** {weather_condition}
        """)
        
        # Quick facts
        st.markdown('<h3 class="sub-header">📈 Quick Facts</h3>', unsafe_allow_html=True)
        st.success("""
        ✅ **Model Accuracy:** ~85% R²  
        ✅ **Update Frequency:** Real-time  
        ✅ **Data Sources:** Multi-city delivery data  
        ✅ **Prediction Horizon:** Next hour
        """)
        
        # Contact info
        st.markdown('<h3 class="sub-header">📞 Support</h3>', unsafe_allow_html=True)
        st.markdown("""
        Need help or have questions?
        
        📧 **Email:** support@fooddelivery.com  
        📱 **Phone:** +91-XXX-XXXXXXX  
        🌐 **Website:** www.fooddelivery.com
        """)

if __name__ == "__main__":
    main()