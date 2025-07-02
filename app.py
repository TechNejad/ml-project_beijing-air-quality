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
    city = st.text_input("Enter any city", value="Beijing")
    forecast_button = st.button("Get Pollution Forecast", type="primary")

def get_city_coords(city_name):
    """Get latitude and longitude for a city using Nominatim API."""
    try:
        url = f"https://nominatim.openstreetmap.org/search?q={city_name}&format=json&limit=1"
        headers = {'User-Agent': 'Mozilla/5.0'} # Nominatim requires a user-agent
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        if data:
            return float(data[0]['lat']), float(data[0]['lon'])
        else:
            st.error(f"Could not find coordinates for city: {city_name}")
            return None, None
    except Exception as e:
        st.error(f"Geocoding error: {e}")
        return None, None

def fetch_weather_forecast(city):
    lat, lon = get_city_coords(city)
    if lat is None or lon is None:
        return None
    
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

    df['Extreme_PM2.5'] = False
    df['Extreme_Event_VMD_shift1'] = False

    df['pm2.5_lag1'] = 50
    df['pm2.5_lag2'] = 50
    df['pm2.5_lag3'] = 50
    df['pm2.5_lag6'] = 50
    df['pm2.5_lag12'] = 50
    df['pm2.5_lag24'] = 50
    df['pm2.5_roll24_mean'] = 50
    df['pm2.5_roll24_std'] = 5

    return df

def predict_pm25(weather_df):
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

st.header("AQI Categories Legend")
aqi_categories = [
    {"name": "Good", "color": "#00e400", "range": "0-12", "description": "Air quality is satisfactory, poses little or no risk."},
    {"name": "Moderate", "color": "#ffff00", "range": "12.1-35.4", "description": "Acceptable air quality, but moderate health concern for very sensitive individuals."},
    {"name": "Unhealthy for Sensitive Groups", "color": "#ff7e00", "range": "35.5-55.4", "description": "Members of sensitive groups may experience health effects."},
    {"name": "Unhealthy", "color": "#ff0000", "range": "55.5-150.4", "description": "Everyone may begin to experience health effects."},
    {"name": "Very Unhealthy", "color": "#8f3f97", "range": "150.5-250.4", "description": "Health alert: everyone may experience more serious health effects."},
    {"name": "Hazardous", "color": "#7e0023", "range": "250.5+", "description": "Health warning of emergency conditions, entire population likely affected."}
]

cols = st.columns(3)
for i, category in enumerate(aqi_categories):
    with cols[i % 3]:
        text_color = 'black' if category['name'] not in ['Unhealthy', 'Very Unhealthy', 'Hazardous'] else 'white'
        if category['name'] == 'Moderate': text_color = '#333333'
        st.markdown(
            f'''
            <div style="background-color: {category['color']}; padding: 15px; border-radius: 8px; margin-bottom: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); color: {text_color};">
                <h3 style="margin-top: 0; margin-bottom: 8px; font-weight: 600;">{category['name']}</h3>
                <div style="font-size: 1.0em; margin-bottom: 8px; font-weight: 500;">PM2.5: {category['range']} μg/m³</div>
                <div style="font-size: 0.9em; line-height: 1.4;">{category['description']}</div>
            </div>
            ''',
            unsafe_allow_html=True
        )