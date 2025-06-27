import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import requests
import json
import os
import joblib
from pathlib import Path

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
    
    # Try loading with joblib first
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.warning(f"Joblib load failed, trying pickle with specific encoding: {e}")
        import pickle
        with open(model_path, 'rb') as f:
            return pickle.load(f, encoding='latin1')

# Load model
model = load_model()

# Function to fetch weather forecast data
def fetch_weather_forecast(city):
    """Get weather forecast data from Open-Meteo API"""
    # Default coordinates for common cities
    city_coordinates = {
        "beijing": {"lat": 39.9042, "lon": 116.4074},
        "shanghai": {"lat": 31.2304, "lon": 121.4737},
        "guangzhou": {"lat": 23.1291, "lon": 113.2644},
        "shenzhen": {"lat": 22.5431, "lon": 114.0579},
        "chengdu": {"lat": 30.5728, "lon": 104.0668},
        "tianjin": {"lat": 39.3434, "lon": 117.3616},
        "wuhan": {"lat": 30.5928, "lon": 114.3055},
        "xian": {"lat": 34.3416, "lon": 108.9398},
    }
    
    try:
        # Get coordinates for the city (default to Beijing if not found)
        city_lower = city.lower()
        coords = city_coordinates.get(city_lower, city_coordinates["beijing"])
        
        # Open-Meteo API endpoint
        url = "https://api.open-meteo.com/v1/forecast"
        
        # Parameters for the API request
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
        
        # Make the API request
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching weather data: {e}")
        return None

def preprocess_weather_data(weather_data):
    """Transform weather API data into format expected by the model"""
    if not weather_data or 'hourly' not in weather_data:
        return pd.DataFrame()
    
    try:
        # Extract hourly data
        hourly = weather_data['hourly']
        times = hourly['time']
        
        # Create a DataFrame with the raw data
        df = pd.DataFrame({
            'datetime': pd.to_datetime(times),
            'Temp': hourly.get('temperature_2m', [None] * len(times)),
            'Humidity': hourly.get('relative_humidity_2m', [None] * len(times)),
            'Press': hourly.get('pressure_msl', [None] * len(times)),
            'WindSpeed': hourly.get('wind_speed_10m', [None] * len(times)),
            'WindDirection': hourly.get('wind_direction_10m', [None] * len(times)),
            'Precipitation': hourly.get('precipitation', [0] * len(times))
        })
        
        # Convert pressure from hPa to kPa if needed
        if 'Press' in df.columns:
            df['Press'] = df['Press'] / 10.0
            
        # Add time-based features
        df['hour'] = df['datetime'].dt.hour
        df['day'] = df['datetime'].dt.day
        df['month'] = df['datetime'].dt.month
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['day_of_year'] = df['datetime'].dt.dayofyear
        df['is_weekend'] = df['datetime'].dt.dayofweek.isin([5, 6]).astype(int)
        
        # Add time of day categories
        def get_time_of_day(hour):
            if 5 <= hour < 12:
                return 'morning'
            elif 12 <= hour < 17:
                return 'afternoon'
            elif 17 <= hour < 21:
                return 'evening'
            else:
                return 'night'
                
        df['time_of_day'] = df['hour'].apply(get_time_of_day)
        
        # Add season
        def get_season(month):
            if 3 <= month <= 5:
                return 'spring'
            elif 6 <= month <= 8:
                return 'summer'
            elif 9 <= month <= 11:
                return 'fall'
            else:
                return 'winter'
                
        df['Season'] = df['month'].apply(get_season)
        
        # Add wind direction components (U and V)
        def get_wind_components(wind_speed, wind_direction_deg):
            if pd.isna(wind_speed) or pd.isna(wind_direction_deg):
                return np.nan, np.nan
            wind_direction_rad = np.radians(wind_direction_deg)
            u = -wind_speed * np.sin(wind_direction_rad)
            v = -wind_speed * np.cos(wind_direction_rad)
            return u, v
            
        wind_components = df.apply(
            lambda row: get_wind_components(row['WindSpeed'], row['WindDirection']), 
            axis=1
        )
        
        df[['WinDir_U', 'WinDir_V']] = pd.DataFrame(
            wind_components.tolist(), 
            index=df.index, 
            columns=['WinDir_U', 'WinDir_V']
        )
        
        # Add precipitation as HoursOfRain (simplified - assuming any precipitation means rain)
        df['HoursOfRain'] = (df['Precipitation'] > 0).astype(int)
        df['HoursOfSnow'] = 0  # Simplified - would need temperature data to determine snow
        
        # Add rolling calculations
        for window in [3, 6, 12, 24]:
            df[f'HoursOfRain_rolling'] = df['HoursOfRain'].rolling(window=window, min_periods=1).mean()
            df[f'HoursOfSnow_rolling'] = df['HoursOfSnow'].rolling(window=window, min_periods=1).mean()
        
        # Winsorize wind speed (cap at 99th percentile)
        if 'WindSpeed' in df.columns:
            wind_speed_99 = df['WindSpeed'].quantile(0.99)
            df['WindSpeed_Winsorized'] = df['WindSpeed'].clip(upper=wind_speed_99)
        
        # Add placeholder PM2.5 lag features (in a real app, these would come from historical data)
        for lag in [1, 2, 3, 6, 12, 24]:
            df[f'pm2.5_lag{lag}'] = np.nan
        
        # Add rolling statistics (using placeholders)
        df['pm2.5_roll24_mean'] = np.nan
        df['pm2.5_roll24_std'] = np.nan
        
        # Add extreme PM2.5 flag (placeholder)
        df['Extreme_PM2.5'] = 0
        df['Extreme_Event_VMD_shift1'] = 0
        
        # Add DewP (Dew Point) - using approximation if not available
        if 'Temp' in df.columns and 'Humidity' in df.columns:
            # Using the Magnus formula approximation
            df['DewP'] = df.apply(
                lambda x: x['Temp'] - ((100 - x['Humidity']) / 5) if pd.notna(x['Temp']) and pd.notna(x['Humidity']) else np.nan,
                axis=1
            )
        
        return df
        
    except Exception as e:
        st.error(f"Error processing weather data: {e}")
        return pd.DataFrame()

def predict_pm25(weather_df):
    """
    Predict PM2.5 values using the trained model.
    
    Args:
        weather_df: DataFrame containing weather features
        
    Returns:
        Array of predicted PM2.5 values
    """
    # Required features for the model
    required_features = [
        'DewP', 'Temp', 'Press', 'WindSpeed', 'HoursOfSnow', 'HoursOfRain',
        'WindSpeed_Winsorized', 'HoursOfRain_rolling', 'HoursOfSnow_rolling',
        'WinDir_U', 'WinDir_V', 'day_of_week', 'day_of_year', 'is_weekend',
        'pm2.5_lag1', 'pm2.5_lag2', 'pm2.5_lag3', 'pm2.5_lag6', 'pm2.5_lag12',
        'pm2.5_lag24', 'pm2.5_roll24_mean', 'pm2.5_roll24_std', 'month', 'hour',
        'Extreme_PM2.5', 'time_of_day', 'Season', 'Extreme_Event_VMD_shift1'
    ]
    
    # Check for missing features
    missing_features = [feat for feat in required_features if feat not in weather_df.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {', '.join(missing_features)}")
    
    # Select only the required features in the correct order
    X = weather_df[required_features]
    
    # Make predictions
    predictions = model.predict(X)
    return predictions

def pm25_to_aqi_category(pm25):
    """Convert PM2.5 concentration to AQI category"""
    if pm25 <= 12:
        return "Good", "#00e400"  # Green
    elif pm25 <= 35.4:
        return "Moderate", "#ffff00"  # Yellow
    elif pm25 <= 55.4:
        return "Unhealthy for Sensitive Groups", "#ff7e00"  # Orange
    elif pm25 <= 150.4:
        return "Unhealthy", "#ff0000"  # Red
    elif pm25 <= 250.4:
        return "Very Unhealthy", "#8f3f97"  # Purple
    else:
        return "Hazardous", "#7e0023"  # Maroon

def generate_forecast_summary(weather_df, pm25_predictions):
    """Generate a human-readable summary of the forecast"""
    if weather_df.empty or pm25_predictions is None or len(pm25_predictions) == 0:
        return "No forecast data available."
    
    # Create a DataFrame with predictions
    forecast_df = weather_df.copy()
    forecast_df['pm25'] = pm25_predictions
    
    # Add AQI category
    forecast_df['aqi_category'] = forecast_df['pm25'].apply(lambda x: pm25_to_aqi_category(x)[0])
    
    # Find the worst AQI in the forecast
    max_pm25 = forecast_df['pm25'].max()
    worst_aqi = pm25_to_aqi_category(max_pm25)[0]
    
    # Find when the worst AQI occurs
    worst_time = forecast_df.loc[forecast_df['pm25'].idxmax(), 'datetime']
    
    # Calculate average PM2.5
    avg_pm25 = forecast_df['pm25'].mean()
    
    # Generate summary text
    summary = f"""
    ## Forecast Summary
    
    - **Worst Air Quality**: {worst_aqi} (PM2.5: {max_pm25:.1f} µg/m³)
    - **Time of Worst Air Quality**: {worst_time.strftime('%Y-%m-%d %H:%M')}
    - **Average PM2.5**: {avg_pm25:.1f} µg/m³
    """
    
    # Add health recommendations based on worst AQI
    if worst_aqi in ["Unhealthy", "Very Unhealthy", "Hazardous"]:
        summary += "\n**⚠️ Health Advisory**: Consider limiting outdoor activities, especially for sensitive groups."
    elif worst_aqi == "Unhealthy for Sensitive Groups":
        summary += "\n**ℹ️ Note**: Sensitive individuals may experience health effects."
    
    return summary

# Main app logic
if forecast_button:
    with st.spinner("Fetching weather data and generating forecast..."):
        try:
            # Fetch weather data
            weather_data = fetch_weather_forecast(city)
            if not weather_data:
                st.error("Failed to fetch weather data. Please try again later.")
                st.stop()
            
            # Preprocess weather data
            weather_df = preprocess_weather_data(weather_data)
            if weather_df.empty:
                st.error("Failed to process weather data. Please check the input data format.")
                st.stop()
            
            # Ensure we have all required historical PM2.5 data
            required_pm25_columns = [
                'pm2.5_lag1', 'pm2.5_lag2', 'pm2.5_lag3', 'pm2.5_lag6', 
                'pm2.5_lag12', 'pm2.5_lag24', 'pm2.5_roll24_mean', 'pm2.5_roll24_std'
            ]
            
            missing_pm25 = [col for col in required_pm25_columns if col not in weather_df.columns]
            if missing_pm25:
                st.error(
                    f"Missing required historical PM2.5 data: {', '.join(missing_pm25)}. "
                    "Please ensure historical PM2.5 data is available for accurate predictions."
                )
                st.stop()
            
            # Predict PM2.5 values
            pm25_predictions = predict_pm25(weather_df)
                
                if pm25_predictions is not None:
                    # Create tabs for different views
                    tab1, tab2 = st.tabs(["Forecast Chart", "Detailed Data"])
                    
                    with tab1:
                        # Create a plot of the predictions
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        # Plot PM2.5 predictions
                        ax.plot(
                            weather_df['datetime'], 
                            pm25_predictions, 
                            marker='o', 
                            linestyle='-', 
                            color='#1f77b4',
                            label='Predicted PM2.5'
                        )
                        
                        # Add AQI category bands
                        aqi_levels = [0, 12, 35.4, 55.4, 150.4, 250.4, 500]
                        aqi_colors = ['#00e400', '#ffff00', '#ff7e00', '#ff0000', '#8f3f97', '#7e0023']
                        aqi_labels = ['Good', 'Moderate', 'USG', 'Unhealthy', 'Very Unhealthy', 'Hazardous']
                        
                        for i in range(len(aqi_levels) - 1):
                            ax.axhspan(
                                aqi_levels[i], 
                                aqi_levels[i+1], 
                                color=aqi_colors[i], 
                                alpha=0.2, 
                                label=aqi_labels[i]
                            )
                        
                        # Customize the plot
                        ax.set_title(f'PM2.5 Forecast for {city}', fontsize=16)
                        ax.set_xlabel('Date & Time', fontsize=12)
                        ax.set_ylabel('PM2.5 (µg/m³)', fontsize=12)
                        ax.grid(True, linestyle='--', alpha=0.7)
                        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
                        
                        # Rotate x-axis labels for better readability
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        
                        # Display the plot
                        st.pyplot(fig)
                        
                        # Display forecast summary
                        st.markdown(generate_forecast_summary(weather_df, pm25_predictions), unsafe_allow_html=True)
                        
                        # Display AQI categories legend
                        st.subheader("AQI Categories")
                        aqi_info = [
                            {"Range": "0-12", "Category": "Good", "Color": "#00e400"},
                            {"Range": "12.1-35.4", "Category": "Moderate", "Color": "#ffff00"},
                            {"Range": "35.5-55.4", "Category": "Unhealthy for Sensitive Groups", "Color": "#ff7e00"},
                            {"Range": "55.5-150.4", "Category": "Unhealthy", "Color": "#ff0000"},
                            {"Range": "150.5-250.4", "Category": "Very Unhealthy", "Color": "#8f3f97"},
                            {"Range": "250.5+", "Category": "Hazardous", "Color": "#7e0023"}
                        ]
                        
                        # Create a DataFrame for the AQI legend
                        aqi_df = pd.DataFrame(aqi_info)
                        
                        # Display the AQI legend as a styled table
                        def color_cells(val):
                            color = aqi_df[aqi_df['Category'] == val]['Color'].values[0]
                            return f'background-color: {color}'
                        
                        st.dataframe(
                            aqi_df.style.apply(lambda x: [color_cells(x['Category']) for _ in x], axis=1),
                            column_config={
                                "Color": None,
                                "Range": "PM2.5 (µg/m³)",
                                "Category": "AQI Category"
                            },
                            use_container_width=True,
                            hide_index=True
                        )
                    
                    with tab2:
                        # Display detailed forecast data
                        detailed_df = weather_df[['datetime', 'Temp', 'Humidity', 'Press', 'WindSpeed']].copy()
                        detailed_df['PM2.5_Prediction'] = pm25_predictions
                        detailed_df['AQI_Category'] = detailed_df['PM2.5_Prediction'].apply(lambda x: pm25_to_aqi_category(x)[0])
                        
                        # Rename columns for display
                        detailed_df = detailed_df.rename(columns={
                            'datetime': 'Date & Time',
                            'Temp': 'Temperature (°C)',
                            'Humidity': 'Humidity (%)',
                            'Press': 'Pressure (kPa)',
                            'WindSpeed': 'Wind Speed (m/s)'
                        })
                        
                        # Format the datetime
                        detailed_df['Date & Time'] = detailed_df['Date & Time'].dt.strftime('%Y-%m-%d %H:%M')
                        
                        # Display the detailed data
                        st.dataframe(
                            detailed_df,
                            column_config={
                                "PM2.5_Prediction": st.column_config.NumberColumn(
                                    "PM2.5 (µg/m³)",
                                    format="%.1f"
                                ),
                                "Temperature (°C)": st.column_config.NumberColumn(
                                    "Temperature (°C)",
                                    format="%.1f"
                                ),
                                "Humidity (%)": st.column_config.NumberColumn(
                                    "Humidity (%)",
                                    format="%.1f"
                                ),
                                "Pressure (kPa)": st.column_config.NumberColumn(
                                    "Pressure (kPa)",
                                    format="%.1f"
                                ),
                                "Wind Speed (m/s)": st.column_config.NumberColumn(
                                    "Wind Speed (m/s)",
                                    format="%.1f"
                                )
                            },
                            use_container_width=True,
                            hide_index=True
                        )
            else:
                st.error("Failed to process weather data.")
        else:
            st.error("Failed to fetch weather data.")

# Footer
st.markdown("---")
st.markdown("### About")
st.markdown("""
This app provides air quality forecasts using machine learning. 
The model predicts PM2.5 concentrations based on weather forecasts.
""")
