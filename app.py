

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import requests
import joblib

# Set page configuration
st.set_page_config(
    page_title="Air Pollution Forecast",
    page_icon="🌬️",
    layout="wide"
)

# Load the trained model
try:
    model = joblib.load('rf_pm25_model.pkl')
except FileNotFoundError:
    st.error("Error: Model file 'rf_pm25_model.pkl' not found. Please make sure the model file is in the same directory as the app.")
    st.stop()
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

st.title("Air Pollution Forecast")

with st.sidebar:
    st.header("Settings")
    city = st.text_input("City", value="Beijing")
    forecast_button = st.button("Get Pollution Forecast", type="primary")

def fetch_weather_forecast(city):
    city_coordinates = {
        "beijing": {"lat": 39.9042, "lon": 116.4074},
        "shanghai": {"lat": 31.2304, "lon": 121.4737},
        "guangzhou": {"lat": 23.1291, "lon": 113.2644}
    }
    city_key = city.lower().replace(" ", "")
    lat = city_coordinates.get(city_key, city_coordinates["beijing"])["lat"]
    lon = city_coordinates.get(city_key, city_coordinates["beijing"])["lon"]
    
    try:
        url = "https://api.open-meteo.com/v1/cma"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,dew_point_2m,pressure_msl,precipitation,wind_speed_10m,wind_direction_10m",
            "forecast_days": 3,
            "timezone": "auto"
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        return process_open_meteo_data(response.json())
    except Exception as e:
        st.error(f"Error fetching weather data: {e}")
        return None

def process_open_meteo_data(data):
    hourly = data.get("hourly", {})
    time_values = hourly.get("time", [])
    if not time_values:
        return pd.DataFrame()

    df = pd.DataFrame({
        'datetime': pd.to_datetime(time_values),
        'Temp': hourly.get("temperature_2m"),
        'DewP': hourly.get("dew_point_2m"),
        'Press': hourly.get("pressure_msl"),
        'WindSpeed': hourly.get("wind_speed_10m"),
        'WinDir': hourly.get("wind_direction_10m"),
        'HoursOfRain': hourly.get("precipitation"),
        'HoursOfSnow': 0
    })
    return df

def preprocess_for_model(df):
    df.set_index('datetime', inplace=True)

    # --- Replicate ALL feature engineering from the notebook ---
    df['WindSpeed_Winsorized'] = df['WindSpeed'] 
    df['HoursOfRain_rolling'] = df['HoursOfRain'].rolling(window=24, min_periods=1).sum()
    df['HoursOfSnow_rolling'] = df['HoursOfSnow'].rolling(window=24, min_periods=1).sum()

    def degrees_to_cardinal(d):
        dirs = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
        return dirs[int(round(d / (360. / len(dirs)))) % len(dirs)]
    df['WinDir'] = df['WinDir'].apply(degrees_to_cardinal)

    wind_direction_mapping = {
        'n': 0, 'nne': 22.5, 'ne': 45, 'ene': 67.5, 'e': 90, 'ese': 112.5, 'se': 135, 'sse': 157.5,
        's': 180, 'ssw': 202.5, 'sw': 225, 'wsw': 247.5, 'w': 270, 'wnw': 292.5, 'nw': 315, 'nnw': 337.5,
    }
    df['WinDir_degrees'] = df['WinDir'].str.lower().map(wind_direction_mapping).fillna(0)
    df['WinDir_U'] = np.sin(np.radians(df['WinDir_degrees'])) * df['WindSpeed']
    df['WinDir_V'] = np.cos(np.radians(df['WinDir_degrees'])) * df['WindSpeed']

    df['day_of_week'] = df.index.dayofweek
    df['day_of_year'] = df.index.dayofyear
    df['month'] = df.index.month
    df['hour'] = df.index.hour
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    def get_season(month):
        if month in [12, 1, 2]: return 'Winter'
        elif month in [3, 4, 5]: return 'Spring'
        elif month in [6, 7, 8]: return 'Summer'
        else: return 'Fall'
    df['Season'] = df['month'].apply(get_season)

    def get_time_of_day(hour):
        if 0 <= hour < 6: return 'Night'
        elif 6 <= hour < 12: return 'Morning'
        elif 12 <= hour < 18: return 'Afternoon'
        else: return 'Evening'
    df['time_of_day'] = df['hour'].apply(get_time_of_day)

    # *** FIX: Add the missing columns the model expects ***
    # For a live forecast, we assume these events are not occurring by default.
    df['Extreme_PM2.5'] = False
    df['Extreme_Event_VMD_shift1'] = False

    # Create placeholder lag/rolling features for PM2.5 as we don't have historical data
    # In a real-world scenario, you'd fetch the last 24 hours of actual PM2.5 data
    df['pm2.5_lag1'] = 50 # Placeholder value
    df['pm2.5_lag2'] = 50
    df['pm2.5_lag3'] = 50
    df['pm2.5_lag6'] = 50
    df['pm2.5_lag12'] = 50
    df['pm2.5_lag24'] = 50
    df['pm2.5_roll24_mean'] = 50
    df['pm2.5_roll24_std'] = 5

    return df

def predict_pm25(weather_df):
    """Use the loaded model pipeline to predict PM2.5 values."""
    # The pipeline handles column selection and transformations internally
    predictions = model.predict(weather_df)
    return predictions

def pm25_to_aqi_category(pm25):
    if pm25 <= 12: return "Good", "#00e400"
    elif pm25 <= 35.4: return "Moderate", "#ffff00"
    elif pm25 <= 55.4: return "Unhealthy for Sensitive Groups", "#ff7e00"
    elif pm25 <= 150.4: return "Unhealthy", "#ff0000"
    elif pm25 <= 250.4: return "Very Unhealthy", "#8f3f97"
    else: return "Hazardous", "#7e0023"

def generate_forecast_summary(weather_df, pm25_predictions):
    if weather_df.empty or not pm25_predictions.any():
        return "No forecast data available."
    
    forecast_df = pd.DataFrame({"datetime": weather_df.index, "pm25": pm25_predictions})
    forecast_df["aqi_category"], _ = zip(*forecast_df["pm25"].apply(pm25_to_aqi_category))
    
    worst_idx = forecast_df["pm25"].idxmax()
    worst_category = forecast_df.loc[worst_idx, "aqi_category"]
    worst_time = forecast_df.loc[worst_idx, "datetime"]
    
    date_str = "Today" if worst_time.date() == datetime.date.today() else "Tomorrow" if worst_time.date() == datetime.date.today() + datetime.timedelta(days=1) else worst_time.strftime("%A, %B %d")
    
    hour = worst_time.hour
    if 6 <= hour < 12: period = "morning"
    elif 12 <= hour < 18: period = "afternoon"
    else: period = "evening" if hour >= 18 else "night"
    
    summary = [f"Air quality will be worst on {date_str} during the {period}, reaching {worst_category} levels."]
    if worst_category in ["Unhealthy", "Very Unhealthy", "Hazardous"]: summary.append("\n⛔ Outdoor activity is not recommended.")
    elif worst_category == "Unhealthy for Sensitive Groups": summary.append("\n⚠️ Sensitive individuals should limit outdoor activity.")
    
    return "\n\n".join(summary)

if forecast_button:
    with st.spinner("Fetching weather data and generating forecast..."):
        weather_data = fetch_weather_forecast(city)
        if weather_data is not None and not weather_data.empty:
            weather_df_processed = preprocess_for_model(weather_data.copy())
            if not weather_df_processed.empty:
                pm25_predictions = predict_pm25(weather_df_processed)
                if pm25_predictions is not None:
                    tab1, tab2 = st.tabs(["Forecast Chart", "Detailed Data"])
                    with tab1:
                        plot_df = pd.DataFrame({"datetime": weather_df_processed.index, "pm25": pm25_predictions})
                        fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
                        ax.set_facecolor('white')
                        ax.plot(plot_df["datetime"], plot_df["pm25"], marker='o', color='black', linewidth=2)
                        max_pm25 = max(plot_df["pm25"]) * 1.2
                        y_max = max(60, max_pm25)
                        ax.axhspan(0, 12, alpha=0.3, color='#00e400')
                        ax.axhspan(12, 35.4, alpha=0.3, color='#ffff00')
                        ax.axhspan(35.4, 55.4, alpha=0.3, color='#ff7e00')
                        if y_max > 55.4: ax.axhspan(55.4, 150.4, alpha=0.3, color='#ff0000')
                        if y_max > 150.4: ax.axhspan(150.4, 250.4, alpha=0.3, color='#8f3f97')
                        if y_max > 250.4: ax.axhspan(250.4, y_max, alpha=0.3, color='#7e0023')
                        ax.set_ylim(0, y_max)
                        ax.set_xlabel('Time', color='black', fontsize=12)
                        ax.set_ylabel('PM2.5 (μg/m³)', color='black', fontsize=12)
                        ax.set_title(f'Predicted Air Pollution Levels for {city}', color='black', fontsize=14, pad=20)
                        date_format = plt.matplotlib.dates.DateFormatter('%m-%d\n%H:00')
                        ax.xaxis.set_major_formatter(date_format)
                        ax.xaxis.set_major_locator(plt.matplotlib.dates.HourLocator(interval=12))
                        plt.xticks(color='black', rotation=0)
                        ax.grid(True, linestyle='--', alpha=0.3, color='gray')
                        plt.yticks(color='black')
                        plt.tight_layout()
                        st.pyplot(fig)
                        st.header("Forecast Summary")
                        st.markdown(generate_forecast_summary(weather_df_processed, pm25_predictions))
                    with tab2:
                        detailed_df = pd.DataFrame({
                            "Date & Time": weather_df_processed.index,
                            "Temperature (°C)": weather_df_processed["Temp"],
                            "Wind Speed (km/h)": weather_df_processed["WindSpeed"],
                            "Pressure (hPa)": weather_df_processed["Press"],
                            "Predicted PM2.5 (μg/m³)": pm25_predictions
                        })
                        detailed_df["AQI Category"] = [pm25_to_aqi_category(p)[0] for p in pm25_predictions]
                        st.dataframe(detailed_df, hide_index=True)
                else: st.error("Failed to generate PM2.5 predictions.")
            else: st.error("Failed to process weather data.")
        else: st.error("Failed to fetch weather data.")

st.markdown("---")
