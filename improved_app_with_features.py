import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import requests
import json
import os
import joblib
import re
import pytz
import math
import time
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

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
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #262730;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF;
        border-radius: 4px 4px 0 0;
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
    
    /* Error message styling */
    .error-box {
        background-color: #FFEBEE;
        border-left: 5px solid #F44336;
        padding: 15px;
        margin: 10px 0;
        border-radius: 4px;
    }
    
    /* Warning message styling */
    .warning-box {
        background-color: #FFF8E1;
        border-left: 5px solid #FFC107;
        padding: 15px;
        margin: 10px 0;
        border-radius: 4px;
    }
    
    /* Info message styling */
    .info-box {
        background-color: #E3F2FD;
        border-left: 5px solid #2196F3;
        padding: 15px;
        margin: 10px 0;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Constants
MODEL_PATH = "rf_pm25_model.pkl"
FORECAST_HOURS = 72  # predict next 72 h
HISTORY_HOURS = 24   # hours of real pm2.5 for lag features
MAX_RETRIES = 3      # maximum number of API retry attempts
RETRY_BACKOFF = 2    # exponential backoff factor

# API URLs
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
AIRQUAL_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
HISTORICAL_WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"

# Fallback data paths (for when APIs are unreachable)
FALLBACK_DATA_DIR = "fallback_data"

# Map Open‑Meteo variable → column name used during model training
WEATHER_RENAME = {
    "temperature_2m": "Temp",
    "dew_point_2m": "DewP",
    "pressure_msl": "Press",
    "wind_speed_10m": "WindSpeed",
    "wind_direction_10m": "WindDir",
    "relative_humidity_2m": "Humidity",
    "precipitation": "precipitation",
    "snowfall": "snowfall"
}

# App title only
st.title("Air Pollution Forecast")

# Sidebar for inputs
with st.sidebar:
    st.header("Settings")
    
    # City selector (default: Beijing)
    city = st.text_input("City", value="Beijing")
    
    # Forecast button
    forecast_button = st.button("Get Pollution Forecast", type="primary")
    
    # Add a checkbox for using fallback data (for testing or when APIs are down)
    use_fallback = st.checkbox("Use fallback data (offline mode)", value=False, 
                              help="Enable this if you're having issues with API connectivity")

# ---------------------------------------------------------------
# API HELPERS WITH ERROR HANDLING
# ---------------------------------------------------------------

def create_session_with_retries():
    """Create a requests session with retry capabilities"""
    session = requests.Session()
    retry_strategy = Retry(
        total=MAX_RETRIES,
        backoff_factor=RETRY_BACKOFF,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def safe_api_request(url, params, timeout=15, fallback_data=None, fallback_file=None):
    """Make an API request with error handling and fallback options"""
    try:
        session = create_session_with_retries()
        r = session.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.SSLError:
        st.warning("""
        SSL connection error. This may be due to network restrictions in your environment.
        
        If you're running on Streamlit Cloud, some API connections might be restricted.
        Try using the fallback data option or run the app locally.
        """, icon="⚠️")
        if fallback_data:
            return fallback_data
        elif fallback_file and os.path.exists(fallback_file):
            with open(fallback_file, 'r') as f:
                return json.load(f)
        raise
    except requests.exceptions.ConnectionError:
        st.warning("""
        Connection error. Unable to reach the API server.
        
        This could be due to network issues or API service unavailability.
        Try using the fallback data option or try again later.
        """, icon="⚠️")
        if fallback_data:
            return fallback_data
        elif fallback_file and os.path.exists(fallback_file):
            with open(fallback_file, 'r') as f:
                return json.load(f)
        raise
    except requests.exceptions.Timeout:
        st.warning("""
        Request timed out. The API server is taking too long to respond.
        
        Try using the fallback data option or try again later.
        """, icon="⚠️")
        if fallback_data:
            return fallback_data
        elif fallback_file and os.path.exists(fallback_file):
            with open(fallback_file, 'r') as f:
                return json.load(f)
        raise
    except requests.exceptions.RequestException as e:
        st.error(f"""
        API request failed: {str(e)}
        
        Try using the fallback data option or try again later.
        """, icon="🚨")
        if fallback_data:
            return fallback_data
        elif fallback_file and os.path.exists(fallback_file):
            with open(fallback_file, 'r') as f:
                return json.load(f)
        raise

def geocode_city(city: str, use_fallback=False):
    """Get latitude, longitude, timezone, and proper name for a city"""
    # Fallback data for Beijing (most common use case)
    fallback_data = {
        "results": [{
            "name": "Beijing",
            "latitude": 39.9075,
            "longitude": 116.39723,
            "timezone": "Asia/Shanghai"
        }]
    }
    
    # If using fallback mode or if the city is Beijing, use fallback data
    if use_fallback or city.lower() == "beijing":
        return (fallback_data["results"][0]["latitude"], 
                fallback_data["results"][0]["longitude"], 
                fallback_data["results"][0]["timezone"], 
                fallback_data["results"][0]["name"])
    
    # Otherwise try the API
    fallback_file = os.path.join(FALLBACK_DATA_DIR, f"geocode_{city.lower().replace(' ', '_')}.json")
    
    try:
        response = safe_api_request(
            GEOCODE_URL, 
            params={"name": city, "count": 1}, 
            fallback_data=fallback_data if city.lower() == "beijing" else None,
            fallback_file=fallback_file
        )
        
        res = response.get("results")
        if not res:
            st.warning(f"City '{city}' not found. Using Beijing as default.", icon="⚠️")
            return (fallback_data["results"][0]["latitude"], 
                    fallback_data["results"][0]["longitude"], 
                    fallback_data["results"][0]["timezone"], 
                    fallback_data["results"][0]["name"])
        
        d = res[0]
        
        # Save this result for future fallback
        os.makedirs(FALLBACK_DATA_DIR, exist_ok=True)
        with open(fallback_file, 'w') as f:
            json.dump(response, f)
            
        return d["latitude"], d["longitude"], d["timezone"], d["name"]
    except Exception as e:
        st.error(f"Error geocoding city: {str(e)}", icon="🚨")
        return (fallback_data["results"][0]["latitude"], 
                fallback_data["results"][0]["longitude"], 
                fallback_data["results"][0]["timezone"], 
                fallback_data["results"][0]["name"])

def generate_datetime_range(hours, start_from_past=False):
    """Generate a range of datetime objects with proper timezone handling"""
    now = datetime.datetime.now().replace(minute=0, second=0, microsecond=0)
    
    if start_from_past:
        # Generate datetimes from past to now
        return [now - datetime.timedelta(hours=i) for i in range(hours, 0, -1)]
    else:
        # Generate datetimes from now to future
        return [now + datetime.timedelta(hours=i) for i in range(hours)]

def fetch_weather(lat, lon, tz, use_fallback=False):
    """Get weather forecast data from Open-Meteo API"""
    # Create fallback data filename based on coordinates
    fallback_file = os.path.join(FALLBACK_DATA_DIR, f"weather_{lat:.2f}_{lon:.2f}.json")
    
    # Generate proper datetime strings for fallback data
    future_datetimes = generate_datetime_range(FORECAST_HOURS)
    datetime_strings = [dt.isoformat() for dt in future_datetimes]
    
    # Beijing fallback data (pre-generated)
    beijing_fallback = {
        "hourly": {
            "time": datetime_strings,
            "temperature_2m": [np.random.normal(25, 5) for _ in range(FORECAST_HOURS)],
            "dew_point_2m": [np.random.normal(15, 3) for _ in range(FORECAST_HOURS)],
            "pressure_msl": [np.random.normal(1013, 5) for _ in range(FORECAST_HOURS)],
            "wind_speed_10m": [np.random.normal(10, 5) for _ in range(FORECAST_HOURS)],
            "wind_direction_10m": [np.random.randint(0, 360) for _ in range(FORECAST_HOURS)],
            "relative_humidity_2m": [np.random.normal(60, 10) for _ in range(FORECAST_HOURS)],
            "precipitation": [max(0, np.random.normal(0, 0.5)) for _ in range(FORECAST_HOURS)],
            "snowfall": [max(0, np.random.normal(0, 0.1)) for _ in range(FORECAST_HOURS)]
        }
    }
    
    if use_fallback:
        # Use the fallback data directly
        df = pd.DataFrame(beijing_fallback["hourly"]).rename(columns=WEATHER_RENAME)
        df["time"] = pd.to_datetime(df["time"])
        return df
    
    try:
        params = {
            "latitude": lat, 
            "longitude": lon, 
            "timezone": tz,
            "hourly": ",".join(WEATHER_RENAME.keys()),
            "forecast_hours": FORECAST_HOURS,
        }
        
        response = safe_api_request(
            WEATHER_URL, 
            params=params, 
            fallback_file=fallback_file
        )
        
        # Save this result for future fallback
        os.makedirs(FALLBACK_DATA_DIR, exist_ok=True)
        with open(fallback_file, 'w') as f:
            json.dump(response, f)
        
        j = response["hourly"]
        df = pd.DataFrame(j).rename(columns=WEATHER_RENAME)
        df["time"] = pd.to_datetime(df["time"])
        return df
    except Exception as e:
        st.warning(f"Error fetching weather data: {str(e)}. Using simulated data.", icon="⚠️")
        
        # Use the fallback data
        df = pd.DataFrame(beijing_fallback["hourly"]).rename(columns=WEATHER_RENAME)
        df["time"] = pd.to_datetime(df["time"])
        return df

def fetch_historical_weather(lat, lon, tz, use_fallback=False):
    """Get historical weather data from Open-Meteo Archive API"""
    # Create fallback data filename based on coordinates
    fallback_file = os.path.join(FALLBACK_DATA_DIR, f"historical_weather_{lat:.2f}_{lon:.2f}.json")
    
    # Generate proper datetime strings for fallback data
    past_datetimes = generate_datetime_range(72, start_from_past=True)
    datetime_strings = [dt.isoformat() for dt in past_datetimes]
    
    # Beijing fallback data (pre-generated)
    beijing_fallback = {
        "hourly": {
            "time": datetime_strings,
            "temperature_2m": [np.random.normal(25, 5) for _ in range(72)],
            "dew_point_2m": [np.random.normal(15, 3) for _ in range(72)],
            "pressure_msl": [np.random.normal(1013, 5) for _ in range(72)],
            "wind_speed_10m": [np.random.normal(10, 5) for _ in range(72)],
            "wind_direction_10m": [np.random.randint(0, 360) for _ in range(72)],
            "relative_humidity_2m": [np.random.normal(60, 10) for _ in range(72)],
            "precipitation": [max(0, np.random.normal(0, 0.5)) for _ in range(72)],
            "snowfall": [max(0, np.random.normal(0, 0.1)) for _ in range(72)]
        }
    }
    
    if use_fallback:
        # Use the fallback data directly
        df = pd.DataFrame(beijing_fallback["hourly"]).rename(columns=WEATHER_RENAME)
        df["time"] = pd.to_datetime(df["time"])
        return df
    
    try:
        # Calculate date range for historical data (3 days back)
        end_date = datetime.datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.datetime.now() - datetime.timedelta(days=3)).strftime("%Y-%m-%d")
        
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "timezone": tz,
            "hourly": ",".join(WEATHER_RENAME.keys())
        }
        
        response = safe_api_request(
            HISTORICAL_WEATHER_URL, 
            params=params, 
            fallback_file=fallback_file
        )
        
        # Save this result for future fallback
        os.makedirs(FALLBACK_DATA_DIR, exist_ok=True)
        with open(fallback_file, 'w') as f:
            json.dump(response, f)
        
        j = response["hourly"]
        df = pd.DataFrame(j).rename(columns=WEATHER_RENAME)
        df["time"] = pd.to_datetime(df["time"])
        return df
    except Exception as e:
        st.warning(f"Error fetching historical weather data: {str(e)}. Using simulated data.", icon="⚠️")
        
        # Use the fallback data
        df = pd.DataFrame(beijing_fallback["hourly"]).rename(columns=WEATHER_RENAME)
        df["time"] = pd.to_datetime(df["time"])
        return df

def fetch_pm25_history(lat, lon, tz, use_fallback=False):
    """Get historical PM2.5 data from Open-Meteo Air Quality API"""
    # Create fallback data filename based on coordinates
    fallback_file = os.path.join(FALLBACK_DATA_DIR, f"pm25_history_{lat:.2f}_{lon:.2f}.json")
    
    # Generate proper datetime strings for fallback data
    past_datetimes = generate_datetime_range(72, start_from_past=True)
    datetime_strings = [dt.isoformat() for dt in past_datetimes]
    
    # Beijing fallback data (pre-generated)
    beijing_fallback = {
        "hourly": {
            "time": datetime_strings,
            "pm2_5": [max(5, np.random.normal(50, 20)) for _ in range(72)]
        }
    }
    
    if use_fallback:
        # Use the fallback data directly
        df = pd.DataFrame(beijing_fallback["hourly"])
        df["time"] = pd.to_datetime(df["time"])
        # Rename pm2_5 to pm2.5 to match model training column name
        df = df.rename(columns={"pm2_5": "pm2.5"})
        return df
    
    try:
        params = {
            "latitude": lat, 
            "longitude": lon, 
            "timezone": tz,
            "hourly": "pm2_5", 
            "past_days": 3, 
            "forecast_hours": 0  # We only need historical data
        }
        
        response = safe_api_request(
            AIRQUAL_URL, 
            params=params, 
            fallback_file=fallback_file
        )
        
        # Save this result for future fallback
        os.makedirs(FALLBACK_DATA_DIR, exist_ok=True)
        with open(fallback_file, 'w') as f:
            json.dump(response, f)
        
        j = response["hourly"]
        df = pd.DataFrame(j)
        df["time"] = pd.to_datetime(df["time"])
        # Rename pm2_5 to pm2.5 to match model training column name
        df = df.rename(columns={"pm2_5": "pm2.5"})
        return df
    except Exception as e:
        st.warning(f"Error fetching PM2.5 history: {str(e)}. Using simulated data.", icon="⚠️")
        
        # Use the fallback data
        df = pd.DataFrame(beijing_fallback["hourly"])
        df["time"] = pd.to_datetime(df["time"])
        # Rename pm2_5 to pm2.5 to match model training column name
        df = df.rename(columns={"pm2_5": "pm2.5"})
        return df

# ---------------------------------------------------------------
# MODEL & FEATURE LOGIC
# ---------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    """Load the trained Random Forest model"""
    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found – upload rf_pm25_model.pkl")
        st.stop()
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

def get_season(month):
    """Determine season from month number"""
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:  # 9, 10, 11
        return "Fall"

def get_time_of_day(hour):
    """Categorize hour into time of day"""
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    else:  # 21-23, 0-4
        return "Night"

def calculate_wind_components(wind_speed, wind_direction):
    """Calculate U and V components of wind"""
    # Convert degrees to radians
    wind_dir_rad = np.radians(wind_direction)
    
    # Calculate U (east-west) and V (north-south) components
    # U is positive when wind is blowing from west to east
    # V is positive when wind is blowing from south to north
    u = -wind_speed * np.sin(wind_dir_rad)
    v = -wind_speed * np.cos(wind_dir_rad)
    
    return u, v

def winsorize(data, lower_percentile=0.05, upper_percentile=0.95):
    """Cap outliers at specified percentiles"""
    if isinstance(data, pd.Series):
        lower_bound = data.quantile(lower_percentile)
        upper_bound = data.quantile(upper_percentile)
        return data.clip(lower=lower_bound, upper=upper_bound)
    else:
        # Handle numpy arrays or lists
        data_array = np.array(data)
        lower_bound = np.quantile(data_array, lower_percentile)
        upper_bound = np.quantile(data_array, upper_percentile)
        return np.clip(data_array, lower_bound, upper_bound)

def calculate_hours_of_precipitation(precip_data, threshold=0.1):
    """Calculate hours of precipitation (rain or snow) based on threshold"""
    return sum(1 for p in precip_data if p >= threshold)

def is_extreme_pm25(pm25_value, threshold=150):
    """Determine if PM2.5 value is extreme"""
    return "Yes" if pm25_value >= threshold else "No"

def create_lag_features(df, history_pm25):
    """Create lag features for PM2.5 values"""
    # Create a copy of the dataframe
    result_df = df.copy()
    
    # Define lag periods
    lag_periods = [1, 2, 3, 6, 12, 24]
    
    # Add lag features
    for lag in lag_periods:
        col_name = f"pm2.5_lag{lag}"
        if len(history_pm25) >= lag:
            result_df[col_name] = history_pm25[-lag]
        else:
            # Use the first available value if not enough history
            result_df[col_name] = history_pm25[0] if history_pm25 else np.nan
    
    return result_df

def create_rolling_features(df, history_pm25):
    """Create rolling statistics features for PM2.5"""
    # Create a copy of the dataframe
    result_df = df.copy()
    
    # Calculate 24-hour rolling mean and std if enough history
    if len(history_pm25) > 0:
        window_size = min(24, len(history_pm25))
        result_df["pm2.5_roll24_mean"] = np.mean(history_pm25[-window_size:])
        result_df["pm2.5_roll24_std"] = np.std(history_pm25[-window_size:]) if window_size > 1 else 0
    else:
        result_df["pm2.5_roll24_mean"] = np.nan
        result_df["pm2.5_roll24_std"] = np.nan
    
    return result_df

def add_time_features(df):
    """Add time-based features to the dataframe"""
    # Create a copy of the dataframe
    result_df = df.copy()
    
    # Ensure time column is datetime
    if not pd.api.types.is_datetime64_any_dtype(result_df["time"]):
        result_df["time"] = pd.to_datetime(result_df["time"])
    
    # Extract datetime components
    result_df["Year"] = result_df["time"].dt.year
    result_df["month"] = result_df["time"].dt.month
    result_df["Day"] = result_df["time"].dt.day
    result_df["hour"] = result_df["time"].dt.hour
    result_df["day_of_week"] = result_df["time"].dt.dayofweek
    result_df["day_of_year"] = result_df["time"].dt.dayofyear
    result_df["is_weekend"] = result_df["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)
    result_df["time_of_day"] = result_df["hour"].apply(get_time_of_day)
    result_df["Season"] = result_df["month"].apply(get_season)
    
    return result_df

def add_weather_features(df, historical_weather_df=None):
    """Add weather-related features to the dataframe"""
    # Create a copy of the dataframe
    result_df = df.copy()
    
    # Calculate wind components
    u_components, v_components = [], []
    for i, row in result_df.iterrows():
        u, v = calculate_wind_components(row["WindSpeed"], row["WindDir"])
        u_components.append(u)
        v_components.append(v)
    
    result_df["WinDir_U"] = u_components
    result_df["WinDir_V"] = v_components
    
    # Winsorize wind speed
    result_df["WindSpeed_Winsorized"] = winsorize(result_df["WindSpeed"])
    
    # Calculate hours of precipitation
    # If historical weather data is available, use it for more accurate calculations
    if historical_weather_df is not None:
        # Calculate hours of rain and snow from historical data
        past_24h_precip = historical_weather_df["precipitation"].tail(24).tolist()
        past_24h_snow = historical_weather_df["snowfall"].tail(24).tolist()
        
        result_df["HoursOfRain"] = calculate_hours_of_precipitation(past_24h_precip)
        result_df["HoursOfSnow"] = calculate_hours_of_precipitation(past_24h_snow)
        
        # Calculate rolling features
        result_df["HoursOfRain_rolling"] = calculate_hours_of_precipitation(past_24h_precip)
        result_df["HoursOfSnow_rolling"] = calculate_hours_of_precipitation(past_24h_snow)
    else:
        # Fallback: estimate based on current conditions
        # This is less accurate but provides values when historical data is unavailable
        has_rain = 1 if result_df["precipitation"].iloc[0] > 0.1 else 0
        has_snow = 1 if result_df["snowfall"].iloc[0] > 0.1 else 0
        
        result_df["HoursOfRain"] = has_rain * 3  # Assume 3 hours if currently raining
        result_df["HoursOfSnow"] = has_snow * 3  # Assume 3 hours if currently snowing
        
        result_df["HoursOfRain_rolling"] = has_rain * 3
        result_df["HoursOfSnow_rolling"] = has_snow * 3
    
    return result_df

def add_extreme_event_features(df, history_pm25):
    """Add extreme event features to the dataframe"""
    # Create a copy of the dataframe
    result_df = df.copy()
    
    # Add extreme PM2.5 flag
    if len(history_pm25) > 0:
        last_pm25 = history_pm25[-1]
        result_df["Extreme_PM2.5"] = is_extreme_pm25(last_pm25)
        
        # Add shifted extreme event flag
        if len(history_pm25) > 1:
            prev_pm25 = history_pm25[-2]
            result_df["Extreme_Event_VMD_shift1"] = is_extreme_pm25(prev_pm25)
        else:
            result_df["Extreme_Event_VMD_shift1"] = "No"
    else:
        result_df["Extreme_PM2.5"] = "No"
        result_df["Extreme_Event_VMD_shift1"] = "No"
    
    return result_df

def prepare_features(wx_df, aq_df, historical_weather_df=None):
    """Prepare all features required by the model"""
    # Sort and get historical PM2.5 data
    aq_df = aq_df.sort_values("time")
    history_pm25 = aq_df["pm2.5"].tail(HISTORY_HOURS).tolist()
    
    # Create base dataframe with datetime
    result_df = pd.DataFrame({"time": wx_df["time"]})
    
    # Add weather variables
    for col in WEATHER_RENAME.values():
        if col in wx_df.columns:
            result_df[col] = wx_df[col]
    
    # Add time features
    result_df = add_time_features(result_df)
    
    # Add weather features
    result_df = add_weather_features(result_df, historical_weather_df)
    
    # Add PM2.5 lag features
    result_df = create_lag_features(result_df, history_pm25)
    
    # Add PM2.5 rolling features
    result_df = create_rolling_features(result_df, history_pm25)
    
    # Add extreme event features
    result_df = add_extreme_event_features(result_df, history_pm25)
    
    # Add datetime column for display
    result_df["datetime"] = result_df["time"]
    
    return result_df

def build_future_dataframe(wx_df, aq_df, historical_weather_df=None):
    """Build a dataframe with future predictions using recursive forecasting"""
    # Sort and get historical PM2.5 data
    aq_df = aq_df.sort_values("time")
    history_pm25 = aq_df["pm2.5"].tail(HISTORY_HOURS).tolist()
    
    # Load the model
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()
    
    rows = []
    for idx in range(FORECAST_HOURS):
        try:
            # For the first prediction, use the prepared features
            if idx == 0:
                # Prepare features for the first time step
                pred_df = prepare_features(wx_df.iloc[[idx]], aq_df, historical_weather_df)
                
                # Make prediction
                try:
                    yhat = float(model.predict(pred_df)[0])
                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    # Show the features that were available
                    st.write("Features available:")
                    st.write(pred_df.columns.tolist())
                    # Show the features that the model expects
                    try:
                        st.write("Features expected by model:")
                        if hasattr(model, 'feature_names_in_'):
                            st.write(model.feature_names_in_.tolist())
                        elif hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
                            st.write(model.named_steps['preprocessor'].feature_names_in_.tolist())
                    except:
                        pass
                    st.stop()
                    
                # Add prediction to history for next iteration
                history_pm25.append(yhat)
                
                # Add prediction to result
                pred_df["PM2.5_pred"] = yhat
                rows.append(pred_df.iloc[0].to_dict())
            else:
                # For subsequent predictions, update the features with the latest prediction
                row_time = wx_df.iloc[idx]["time"]
                
                # Create a new dataframe for this time step
                new_row_df = pd.DataFrame({"time": [row_time]})
                
                # Add weather variables
                for col in WEATHER_RENAME.values():
                    if col in wx_df.columns:
                        new_row_df[col] = wx_df.iloc[idx][col]
                
                # Add time features
                new_row_df = add_time_features(new_row_df)
                
                # Add weather features (without historical data for simplicity)
                new_row_df = add_weather_features(new_row_df)
                
                # Add PM2.5 lag features using updated history
                new_row_df = create_lag_features(new_row_df, history_pm25)
                
                # Add PM2.5 rolling features using updated history
                new_row_df = create_rolling_features(new_row_df, history_pm25)
                
                # Add extreme event features using updated history
                new_row_df = add_extreme_event_features(new_row_df, history_pm25)
                
                # Make prediction
                try:
                    yhat = float(model.predict(new_row_df)[0])
                except Exception as e:
                    st.error(f"Prediction error at step {idx}: {e}")
                    st.write("Features available:")
                    st.write(new_row_df.columns.tolist())
                    st.stop()
                    
                # Add prediction to history for next iteration
                history_pm25.append(yhat)
                
                # Add prediction to result
                new_row_df["PM2.5_pred"] = yhat
                rows.append(new_row_df.iloc[0].to_dict())
        except Exception as e:
            st.error(f"Error in prediction step {idx}: {str(e)}")
            # If we have at least one prediction, return what we have
            if rows:
                break
            else:
                st.stop()
    
    return pd.DataFrame(rows)

# Function to convert PM2.5 values to AQI categories
def pm25_to_aqi_category(pm25):
    """Convert PM2.5 concentration to AQI category and color"""
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

# Function to generate forecast summary
def generate_forecast_summary(df_future):
    """Generate a condensed human-friendly summary of the forecast"""
    if df_future.empty:
        return "No forecast data available."
    
    try:
        # Add AQI categories
        df_future["aqi_category"], df_future["aqi_color"] = zip(*df_future["PM2.5_pred"].apply(pm25_to_aqi_category))
        
        # Find the worst AQI category overall
        overall_worst_idx = df_future["PM2.5_pred"].idxmax()
        overall_worst_category = df_future.loc[overall_worst_idx, "aqi_category"]
        worst_time = df_future.loc[overall_worst_idx, "datetime"]
        
        # Ensure worst_time is a valid datetime
        if pd.isna(worst_time) or not isinstance(worst_time, (datetime.datetime, pd.Timestamp)):
            worst_time = datetime.datetime.now() + datetime.timedelta(hours=12)  # Default to 12 hours from now
        
        # Format the date and time
        try:
            date_str = "Today" if worst_time.date() == datetime.datetime.now().date() else (
                "Tomorrow" if worst_time.date() == (datetime.datetime.now() + datetime.timedelta(days=1)).date() else
                worst_time.strftime("%A, %B %d")
            )
            
            # Format time period
            hour = worst_time.hour
            if 6 <= hour < 12:
                period = "morning (6-12 AM)"
            elif 12 <= hour < 18:
                period = "afternoon (12-6 PM)"
            else:
                period = "evening (6-12 PM)" if hour >= 18 else "night (12-6 AM)"
        except Exception as e:
            # Fallback if datetime formatting fails
            date_str = "the forecast period"
            period = "peak hours"
            st.warning(f"Error formatting datetime: {str(e)}")
        
        # Create a single sentence summary
        summary = [f"Air quality will be worst on {date_str} during {period}, reaching {overall_worst_category} levels."]
        
        # Add health recommendation
        if overall_worst_category in ["Unhealthy", "Very Unhealthy", "Hazardous"]:
            summary.append("\n⛔ Outdoor activity is not recommended.")
        elif overall_worst_category == "Unhealthy for Sensitive Groups":
            summary.append("\n⚠️ Sensitive individuals should limit outdoor activity.")
        
        return "\n\n".join(summary)
    except Exception as e:
        st.error(f"Error generating forecast summary: {str(e)}")
        return "Forecast summary unavailable due to an error."

# Main app logic
if forecast_button:
    with st.spinner("Fetching weather data and generating forecast..."):
        try:
            # Create fallback data directory if it doesn't exist
            os.makedirs(FALLBACK_DATA_DIR, exist_ok=True)
            
            # Get city coordinates
            lat, lon, tz, proper_name = geocode_city(city, use_fallback=use_fallback)
            
            # Fetch weather, historical weather, and air quality data
            wx_df = fetch_weather(lat, lon, tz, use_fallback=use_fallback)
            historical_weather_df = fetch_historical_weather(lat, lon, tz, use_fallback=use_fallback)
            aq_df = fetch_pm25_history(lat, lon, tz, use_fallback=use_fallback)
            
            # Build forecast dataframe
            df_future = build_future_dataframe(wx_df, aq_df, historical_weather_df)
            
            if not df_future.empty:
                # Create tabs for different views
                st.markdown("""
                <style>
                .stTabs [data-baseweb="tab-list"] {
                    gap: 10px;
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
                </style>
                """, unsafe_allow_html=True)
                
                tabs = st.tabs(["Summary", "Chart", "Detailed Data"])
                
                # Summary tab
                with tabs[0]:
                    st.subheader(f"Air Quality Forecast for {proper_name}")
                    
                    # Generate summary
                    summary = generate_forecast_summary(df_future)
                    st.markdown(f"### Forecast Summary\n{summary}")
                    
                    # Display current conditions
                    current_pm25 = df_future.loc[0, "PM2.5_pred"]
                    current_category, current_color = pm25_to_aqi_category(current_pm25)
                    
                    st.markdown("### Current Conditions")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("PM2.5 Level", f"{current_pm25:.1f} µg/m³")
                    with col2:
                        st.markdown(f"<div style='background-color: {current_color}; padding: 10px; border-radius: 5px; color: black; text-align: center;'><strong>Air Quality:</strong> {current_category}</div>", unsafe_allow_html=True)
                
                # Chart tab
                with tabs[1]:
                    st.subheader(f"PM2.5 Forecast for {proper_name} - Next {FORECAST_HOURS} Hours")
                    
                    try:
                        # Create figure with AQI color bands
                        fig, ax = plt.subplots(figsize=(10, 5))
                        
                        # Add AQI color bands
                        aqi_bands = [
                            (0, 12, "#00e400"),
                            (12.1, 35.4, "#ffff00"),
                            (35.5, 55.4, "#ff7e00"),
                            (55.5, 150.4, "#ff0000"),
                            (150.5, 250.4, "#8f3f97"),
                            (250.5, 500, "#7e0023")
                        ]
                        
                        for lo, hi, col in aqi_bands:
                            ax.axhspan(lo, hi, color=col, alpha=0.25)
                        
                        # Plot PM2.5 predictions
                        ax.plot(df_future["datetime"], df_future["PM2.5_pred"], marker="o", color="black", linewidth=2)
                        
                        # Add labels and grid
                        ax.set_ylabel("PM2.5 (µg/m³)")
                        ax.set_xlabel("Time")
                        ax.grid(ls="--", alpha=0.3)
                        
                        # Format x-axis to show date and time
                        fig.autofmt_xdate()
                        
                        # Show the plot
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Error generating chart: {str(e)}")
                        st.write("Unable to display chart due to an error with the data.")
                    
                    # Add AQI legend
                    st.markdown("### Air Quality Index (AQI) Categories")
                    legend_cols = st.columns(3)
                    
                    aqi_categories = [
                        ("Good (0-12 µg/m³)", "#00e400"),
                        ("Moderate (12.1-35.4 µg/m³)", "#ffff00"),
                        ("Unhealthy for Sensitive Groups (35.5-55.4 µg/m³)", "#ff7e00"),
                        ("Unhealthy (55.5-150.4 µg/m³)", "#ff0000"),
                        ("Very Unhealthy (150.5-250.4 µg/m³)", "#8f3f97"),
                        ("Hazardous (>250.4 µg/m³)", "#7e0023")
                    ]
                    
                    for i, (label, color) in enumerate(aqi_categories):
                        col_idx = i % 3
                        with legend_cols[col_idx]:
                            st.markdown(f"<div style='background-color: {color}; padding: 5px; margin: 2px; border-radius: 3px; color: black;'>{label}</div>", unsafe_allow_html=True)
                
                # Detailed Data tab
                with tabs[2]:
                    st.subheader(f"Detailed PM2.5 Forecast Data for {proper_name}")
                    
                    try:
                        # Create a copy with formatted datetime and PM2.5 values
                        detailed_df = df_future[["datetime", "PM2.5_pred"]].copy()
                        detailed_df = detailed_df.rename(columns={"PM2.5_pred": "PM2.5 (µg/m³)"})
                        
                        # Format PM2.5 values to one decimal place
                        detailed_df["PM2.5 (µg/m³)"] = detailed_df["PM2.5 (µg/m³)"].apply(lambda x: f"{x:.1f}")
                        
                        # Add AQI category
                        detailed_df["AQI Category"] = [pm25_to_aqi_category(float(pm25))[0] for pm25 in detailed_df["PM2.5 (µg/m³)"]]
                        
                        # Display the detailed data
                        st.dataframe(detailed_df, hide_index=True)
                    except Exception as e:
                        st.error(f"Error displaying detailed data: {str(e)}")
                        st.write("Unable to display detailed data due to an error.")
                    
                # Add note about fallback data if used
                if use_fallback:
                    st.info("""
                    **Note:** This forecast is using fallback data since you selected offline mode.
                    For real forecasts, disable the "Use fallback data" option in the sidebar.
                    """, icon="ℹ️")
            else:
                st.error("Failed to generate forecast data.")
        except Exception as e:
            st.error(f"""
            Error generating forecast: {str(e)}
            
            If you're running this app on Streamlit Cloud and experiencing API connectivity issues,
            try enabling the "Use fallback data" option in the sidebar.
            """, icon="🚨")

# Footer with deployment information
st.markdown("---")
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center;">
    <div>Air Pollution Forecast App - Using Random Forest Model</div>
    <div style="text-align: right; font-size: 0.8em;">
        <span style="color: #888;">Running on: {}</span>
    </div>
</div>
""".format("Streamlit Cloud" if "STREAMLIT_SHARING" in os.environ else "Local Machine"), unsafe_allow_html=True)

# Add information about offline mode
with st.sidebar:
    st.markdown("---")
    st.markdown("""
    ### About Offline Mode
    
    If you're experiencing API connectivity issues (especially on Streamlit Cloud),
    enable the "Use fallback data" option above.
    
    This will use pre-generated data instead of making API calls,
    allowing the app to function without external dependencies.
    """)
