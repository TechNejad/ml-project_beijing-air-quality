import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import requests
import joblib
import pytz
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------
st.set_page_config(page_title="Air Pollution Forecast", page_icon="🌬️", layout="wide")

MODEL_PATH = "rf_pm25_model.pkl"
FORECAST_HOURS = 72  # predict next 72 hours
HISTORY_HOURS = 24   # hours of real pm2.5 for lag features

# AQI color bands (µg/m³)
aqi_bands = [
    (0, 12, "#b7f4b0"), (12.1, 35.4, "#ffff9c"), (35.5, 55.4, "#ffcd96"),
    (55.5, 150.4, "#ff9d9d"), (150.5, 250.4, "#c99ee0"), (250.5, 500, "#a285c3")
]

# Weather API URLs
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

# Load model
try:
    import os
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_PATH)
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please ensure the model file is in the correct location.")
        st.stop()
    
    # Use specific version of joblib to match scikit-learn 1.6.1
    import joblib
    try:
        model = joblib.load(model_path)
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}\n\nThis might be due to a version mismatch between scikit-learn and the saved model.\nPlease ensure you're using scikit-learn version 1.6.1.")
        st.stop()
except Exception as e:
    st.error(f"Error during initialization: {str(e)}")
    st.stop()

# ---------------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------------
def create_features(df):
    # Time-based features
    df['hour'] = df.index.hour
    df['month'] = df.index.month
    df['day_of_week'] = df.index.dayofweek
    df['day_of_year'] = df.index.dayofyear
    df['is_weekend'] = df['day_of_week'].isin([5, 6])
    
    # Time of day categories
    def get_time_of_day(hour):
        if 5 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 17:
            return 'afternoon'
        elif 17 <= hour < 22:
            return 'evening'
        else:
            return 'night'
    df['time_of_day'] = df['hour'].apply(get_time_of_day)
    
    # Season categories
    def get_season(month):
        if month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        elif month in [9, 10, 11]:
            return 'fall'
        else:
            return 'winter'
    df['Season'] = df['month'].apply(get_season)
    
    # Lag features
    lags = [1, 2, 3, 6, 12, 24]
    for lag in lags:
        df[f'pm2.5_lag{lag}'] = df['pm2.5'].shift(lag)
    
    # Rolling features
    df['pm2.5_roll24_mean'] = df['pm2.5'].rolling(window=24).mean()
    df['pm2.5_roll24_std'] = df['pm2.5'].rolling(window=24).std()
    
    # Wind direction components
    df['WinDir_U'] = np.sin(np.deg2rad(df['WindDir']))
    df['WinDir_V'] = np.cos(np.deg2rad(df['WindDir']))
    
    # Categorical encoding
    le = LabelEncoder()
    df['time_of_day'] = le.fit_transform(df['time_of_day'])
    df['Season'] = le.fit_transform(df['Season'])
    
    return df

# ---------------------------------------------------------------
# API HELPERS
# ---------------------------------------------------------------
def get_weather_forecast(latitude, longitude):
    try:
        params = {
            'latitude': latitude,
            'longitude': longitude,
            'hourly': 'temperature_2m,dewpoint_2m,pressure_msl,wind_speed_10m,wind_direction_10m,relative_humidity_2m',
            'forecast_days': 3,
            'timezone': 'auto',
        }
        response = requests.get(WEATHER_URL, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching weather data: {str(e)}")
        return None

# ---------------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------------
def main():
    st.title("Air Quality Forecast 🌬️")
    
    # Sidebar inputs
    st.sidebar.header("Location")
    city = st.sidebar.text_input("Enter city name", "Beijing")
    
    if st.sidebar.button("Get Forecast"):
        # Get weather forecast
        try:
            # Get coordinates for the city
            geocode_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}"
            geocode_response = requests.get(geocode_url)
            geocode_data = geocode_response.json()
            
            if 'results' in geocode_data and len(geocode_data['results']) > 0:
                lat = geocode_data['results'][0]['latitude']
                lon = geocode_data['results'][0]['longitude']
                
                # Get weather forecast
                weather_data = get_weather_forecast(lat, lon)
                if weather_data and 'hourly' in weather_data:
                    # Create DataFrame from weather data
                    df = pd.DataFrame({
                        'time': pd.to_datetime(weather_data['hourly']['time']),
                        'Temp': weather_data['hourly']['temperature_2m'],
                        'DewP': weather_data['hourly']['dewpoint_2m'],
                        'Press': weather_data['hourly']['pressure_msl'],
                        'WindSpeed': weather_data['hourly']['wind_speed_10m'],
                        'WindDir': weather_data['hourly']['wind_direction_10m'],
                        'Humidity': weather_data['hourly']['relative_humidity_2m'],
                    }).set_index('time')
                    
                    # Create features
                    df = create_features(df)
                    
                    # Make predictions
                    predictions = model.predict(df)
                    df['predicted_pm25'] = predictions
                    
                    # Create AQI recommendations
                    recommendations = []
                    for pm25 in predictions:
                        if pm25 <= 12:
                            recommendations.append("Good air quality. Safe to go outside.")
                        elif pm25 <= 35.4:
                            recommendations.append("Moderate air quality. Generally safe, but sensitive groups should be cautious.")
                        elif pm25 <= 55.4:
                            recommendations.append("Unhealthy for sensitive groups. Reduce outdoor activities.")
                        else:
                            recommendations.append("Unhealthy air quality. Avoid outdoor activities.")
                    
                    # Plot results
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # Add AQI color bands
                    for band in aqi_bands:
                        ax.axhspan(band[0], band[1], facecolor=band[2], alpha=0.2)
                    
                    # Plot predictions
                    ax.plot(df.index, df['predicted_pm25'], 'b-', label='Predicted PM2.5')
                    
                    # Format plot
                    ax.set_title(f"Air Quality Forecast for {city}")
                    ax.set_ylabel("PM2.5 (µg/m³)")
                    ax.grid(True, alpha=0.3)
                    plt.xticks(rotation=45)
                    
                    # Show plot
                    st.pyplot(fig)
                    
                    # Show recommendations
                    st.subheader("Hourly Recommendations")
                    for i, (time, rec) in enumerate(zip(df.index, recommendations)):
                        st.write(f"{time.strftime('%Y-%m-%d %H:%M')} - {rec}")
                    
            else:
                st.error("City not found. Please try another location.")
                
        except Exception as e:
            st.error(f"Error processing forecast: {str(e)}")

if __name__ == "__main__":
    main()
