import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import requests
import joblib
from pathlib import Path
from ucimlrepo import fetch_ucirepo

# Set page configuration
st.set_page_config(
    page_title="Air Pollution Forecast",
    page_icon="🌬️",
    layout="wide"
)

# Set light theme for the entire application
st.markdown("""
<style>
    /* Main content area */
    .main .block-container {
        background-color: #FFFFFF;
        color: #262730;
        padding: 2rem;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #F0F2F6;
        color: #262730;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #262730;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #F0F2F6;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F0F2F6;
        border-radius: 4px 4px 0 0;
        padding: 10px 16px;
        margin-right: 4px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF;
        border-bottom: 2px solid #FF4B4B;
    }
    
    /* Override dark theme elements */
    .stApp {
        background-color: #FFFFFF;
    }
    
    /* Text color */
    .stMarkdown, p, div {
        color: #262730;
    }
    
    /* Tab content */
    .stTabContent {
        background-color: #FFFFFF;
        padding: 1rem;
        border-radius: 0 0 4px 4px;
    }
    
    /* Button styling */
    .stButton button {
        background-color: #FF4B4B;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.title("Air Pollution Forecast")

# Sidebar for inputs
with st.sidebar:
    st.header("Settings")
    
    # City selector (default: Beijing)
    city = st.text_input("City", value="Beijing")
    
    # Forecast button
    forecast_button = st.button("Get Pollution Forecast", type="primary")

# Load the trained model
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / 'rf_pm25_model.pkl'
    if not model_path.exists():
        raise FileNotFoundError(
            "Model file 'rf_pm25_model.pkl' not found. "
            "Please ensure the model file is in the same directory as this script."
        )
    return joblib.load(model_path)

# Load historical data for bootstrapping predictions
@st.cache_data
def load_historical_data():
    beijing_pm2_5 = fetch_ucirepo(id=381)
    df = beijing_pm2_5.data.features
    df['pm2.5'] = beijing_pm2_5.data.targets
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df = df.set_index('datetime')
    df = df.sort_index()
    df['pm2.5'] = df['pm2.5'].ffill()
    return df[['pm2.5']].tail(24) # Return last 24 hours

# Load model and data
model = load_model()
historical_data = load_historical_data()

# Function to fetch weather forecast data
def fetch_weather_forecast(city):
    """Get weather forecast data from Open-Meteo API"""
    city_coordinates = {
        "beijing": {"lat": 39.9042, "lon": 116.4074},
        "shanghai": {"lat": 31.2304, "lon": 121.4737},
        "guangzhou": {"lat": 23.1291, "lon": 113.2644},
        "shenzhen": {"lat": 22.5431, "lon": 114.0579},
        "chengdu": {"lat": 30.5728, "lon": 104.0668},
    }
    
    try:
        city_lower = city.lower()
        coords = city_coordinates.get(city_lower, city_coordinates["beijing"])
        
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": coords["lat"],
            "longitude": coords["lon"],
            "hourly": [
                "temperature_2m", "relative_humidity_2m", "pressure_msl",
                "wind_speed_10m", "wind_direction_10m", "precipitation"
            ],
            "timezone": "auto",
            "forecast_days": 3
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching weather data: {e}")
        return None

def create_features(df, pm_data):
    """Create all features required by the model"""
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    
    # Combine with historical pm2.5 data
    full_df = pd.concat([pm_data, df], axis=0)
    full_df = full_df.sort_index()
    
    # Basic time features
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['day_of_week'] = df.index.dayofweek
    df['day_of_year'] = df.index.dayofyear
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # Time of day
    def get_time_of_day(hour):
        if 5 <= hour < 12: return 'morning'
        elif 12 <= hour < 17: return 'afternoon'
        elif 17 <= hour < 21: return 'evening'
        else: return 'night'
    df['time_of_day'] = df['hour'].apply(get_time_of_day)

    # Season
    def get_season(month):
        if 3 <= month <= 5: return 'spring'
        elif 6 <= month <= 8: return 'summer'
        elif 9 <= month <= 11: return 'fall'
        else: return 'winter'
    df['Season'] = df['month'].apply(get_season)

    # Wind components
    wind_dir_rad = np.radians(df['WindDirection'])
    df['WinDir_U'] = -df['WindSpeed'] * np.sin(wind_dir_rad)
    df['WinDir_V'] = -df['WindSpeed'] * np.cos(wind_dir_rad)

    # Weather features
    df['HoursOfRain'] = (df['Precipitation'] > 0).astype(int)
    df['HoursOfSnow'] = 0 # Simplified
    df['HoursOfRain_rolling'] = df['HoursOfRain'].rolling(window=3, min_periods=1).mean()
    df['HoursOfSnow_rolling'] = df['HoursOfSnow'].rolling(window=3, min_periods=1).mean()
    
    # Winsorize WindSpeed
    wind_speed_99 = df['WindSpeed'].quantile(0.99)
    df['WindSpeed_Winsorized'] = df['WindSpeed'].clip(upper=wind_speed_99)

    # Dew Point Approximation
    df['DewP'] = df['Temp'] - ((100 - df['Humidity']) / 5)
    
    # Lag and rolling features for pm2.5
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f'pm2.5_lag{lag}'] = full_df['pm2.5'].shift(lag).loc[df.index]
        
    df['pm2.5_roll24_mean'] = full_df['pm2.5'].rolling(window=24, min_periods=1).mean().loc[df.index]
    df['pm2.5_roll24_std'] = full_df['pm2.5'].rolling(window=24, min_periods=1).std().loc[df.index]

    # Placeholder extreme flags
    df['Extreme_PM2.5'] = 0
    df['Extreme_Event_VMD_shift1'] = 0
    
    return df

def iterative_prediction(weather_data, initial_pm_history):
    """Predict PM2.5 iteratively, using each prediction as history for the next"""
    
    hourly_data = weather_data['hourly']
    forecast_df = pd.DataFrame({
        'datetime': pd.to_datetime(hourly_data['time']),
        'Temp': hourly_data['temperature_2m'],
        'Humidity': hourly_data['relative_humidity_2m'],
        'Press': hourly_data['pressure_msl'],
        'WindSpeed': hourly_data['wind_speed_10m'],
        'WindDirection': hourly_data['wind_direction_10m'],
        'Precipitation': hourly_data['precipitation'],
    })
    
    pm_history = initial_pm_history.copy()
    predictions = []

    for i in range(len(forecast_df)):
        current_hour_data = forecast_df.iloc[[i]]
        
        # Create features for the current hour
        features_df = create_features(current_hour_data.copy(), pm_history)
        
        # Ensure all required columns are present and in order
        required_features = model.feature_names_in_
        features_df = features_df.reindex(columns=required_features, fill_value=0)

        # Predict
        prediction = model.predict(features_df)[0]
        predictions.append(prediction)
        
        # Add prediction to history for next iteration
        new_row = pd.DataFrame({'pm2.5': [prediction]}, index=[current_hour_data.iloc[0]['datetime']])
        pm_history = pd.concat([pm_history, new_row]).sort_index()

    forecast_df['pm2.5_prediction'] = predictions
    return forecast_df

def pm25_to_aqi_category(pm25):
    """Convert PM2.5 concentration to AQI category"""
    if pm25 <= 12: return "Good", "#00e400"
    elif pm25 <= 35.4: return "Moderate", "#ffff00"
    elif pm25 <= 55.4: return "Unhealthy for Sensitive Groups", "#ff7e00"
    elif pm25 <= 150.4: return "Unhealthy", "#ff0000"
    elif pm25 <= 250.4: return "Very Unhealthy", "#8f3f97"
    else: return "Hazardous", "#7e0023"

def generate_forecast_summary(forecast_df):
    """Generate a human-readable summary of the forecast"""
    if forecast_df.empty:
        return "No forecast data available."
    
    max_pm25 = forecast_df['pm2.5_prediction'].max()
    worst_aqi, _ = pm25_to_aqi_category(max_pm25)
    worst_time = forecast_df.loc[forecast_df['pm2.5_prediction'].idxmax(), 'datetime']
    avg_pm25 = forecast_df['pm2.5_prediction'].mean()
    
    summary = f"""    ## Forecast Summary
    - **Worst Air Quality**: {worst_aqi} (PM2.5: {max_pm25:.1f} µg/m³)
    - **Time of Worst Air Quality**: {worst_time.strftime('%Y-%m-%d %H:%M')}
    - **Average PM2.5**: {avg_pm25:.1f} µg/m³
    """
    
    if worst_aqi in ["Unhealthy", "Very Unhealthy", "Hazardous"]:
        summary += "\n**⚠️ Health Advisory**: Consider limiting outdoor activities."
    elif worst_aqi == "Unhealthy for Sensitive Groups":
        summary += "\n**ℹ️ Note**: Sensitive individuals may experience health effects."
    
    return summary

# Main app logic
if forecast_button:
    with st.spinner("Fetching weather data and generating forecast..."):
        weather_data = fetch_weather_forecast(city)
        
        if weather_data:
            forecast_df = iterative_prediction(weather_data, historical_data)
            
            tab1, tab2 = st.tabs(["Forecast Chart", "Detailed Data"])
            
            with tab1:
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(forecast_df['datetime'], forecast_df['pm2.5_prediction'], marker='o', linestyle='-', color='#1f77b4', label='Predicted PM2.5')
                
                aqi_levels = [0, 12, 35.4, 55.4, 150.4, 250.4, 500]
                aqi_colors = ['#00e400', '#ffff00', '#ff7e00', '#ff0000', '#8f3f97', '#7e0023']
                for i in range(len(aqi_levels) - 1):
                    ax.axhspan(aqi_levels[i], aqi_levels[i+1], color=aqi_colors[i], alpha=0.2)
                
                ax.set_title(f'PM2.5 Forecast for {city}', fontsize=16)
                ax.set_xlabel('Date & Time', fontsize=12)
                ax.set_ylabel('PM2.5 (µg/m³)', fontsize=12)
                ax.grid(True, linestyle='--', alpha=0.7)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                
                st.markdown(generate_forecast_summary(forecast_df), unsafe_allow_html=True)
                
                st.subheader("AQI Categories")
                aqi_info = [
                    {"Range": "0-12", "Category": "Good", "Color": "#00e400"},
                    {"Range": "12.1-35.4", "Category": "Moderate", "Color": "#ffff00"},
                    {"Range": "35.5-55.4", "Category": "Unhealthy for Sensitive Groups", "Color": "#ff7e00"},
                    {"Range": "55.5-150.4", "Category": "Unhealthy", "Color": "#ff0000"},
                    {"Range": "150.5-250.4", "Category": "Very Unhealthy", "Color": "#8f3f97"},
                    {"Range": "250.5+", "Category": "Hazardous", "Color": "#7e0023"}
                ]
                aqi_df = pd.DataFrame(aqi_info)
                st.dataframe(
                    aqi_df.style.apply(lambda x: [f'background-color: {x.Color}' for i in x], axis=1),
                    column_config={"Color": None, "Range": "PM2.5 (µg/m³)", "Category": "AQI Category"},
                    use_container_width=True, hide_index=True
                )

            with tab2:
                detailed_df = forecast_df[['datetime', 'Temp', 'Humidity', 'Press', 'WindSpeed', 'pm2.5_prediction']].copy()
                detailed_df['AQI_Category'] = detailed_df['pm2.5_prediction'].apply(lambda x: pm25_to_aqi_category(x)[0])
                detailed_df = detailed_df.rename(columns={
                    'datetime': 'Date & Time', 'Temp': 'Temperature (°C)', 'Humidity': 'Humidity (%)',
                    'Press': 'Pressure (hPa)', 'WindSpeed': 'Wind Speed (m/s)', 'pm2.5_prediction': 'PM2.5 Prediction'
                })
                detailed_df['Date & Time'] = detailed_df['Date & Time'].dt.strftime('%Y-%m-%d %H:%M')
                st.dataframe(detailed_df, use_container_width=True, hide_index=True)
        else:
            st.error("Failed to fetch or process weather data.")

# Footer
st.markdown("---")
st.markdown("### About")
st.markdown("This app provides air quality forecasts using a Random Forest model trained on historical data from Beijing.")