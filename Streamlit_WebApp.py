import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import requests
import os
import joblib  # NEW: for loading our trained model

# --------------------------------------------------
# 🔧  CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Air Pollution Forecast",
    page_icon="🌬️",
    layout="wide"
)

# Light theme / basic styling
st.markdown(
    """
    <style>
        .main .block-container {background-color:#FFFFFF;color:#262730;padding:2rem;}
        section[data-testid="stSidebar"]{background-color:#F0F2F6;color:#262730;}
        h1,h2,h3,h4,h5,h6{color:#262730;}
        .stTabs [data-baseweb="tab-list"]{gap:2px;background-color:#F0F2F6;}
        .stTabs [data-baseweb="tab"]{height:50px;background-color:#F0F2F6;border-radius:4px 4px 0 0;padding:10px 16px;color:#262730;}
        .stTabs [aria-selected="true"]{background-color:#FFFFFF;border-bottom:2px solid #FF4B4B;}
        .stApp{background-color:#FFFFFF;}
        .stButton button{background-color:#FF4B4B;color:white;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Air Pollution Forecast")

# --------------------------------------------------
# 📥  SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.header("Settings")
    city = st.text_input("City", value="Beijing")
    forecast_button = st.button("Get Pollution Forecast", type="primary")

# --------------------------------------------------
# 🌍  WEATHER DATA – Open‑Meteo API
# --------------------------------------------------

def fetch_weather_forecast(city: str):
    """Query Open‑Meteo CMA endpoint and return parsed (hourly) forecast."""
    city_coordinates = {
        "beijing": {"lat": 39.9042, "lon": 116.4074},
        "shanghai": {"lat": 31.2304, "lon": 121.4737},
        "guangzhou": {"lat": 23.1291, "lon": 113.2644},
        "shenzhen": {"lat": 22.5431, "lon": 114.0579},
        "chengdu": {"lat": 30.5728, "lon": 104.0668},
        "tianjin": {"lat": 39.3434, "lon": 117.3616},
        "wuhan": {"lat": 30.5928, "lon": 114.3055},
        "xian": {"lat": 34.3416, "lon": 108.9398},
        "hangzhou": {"lat": 30.2741, "lon": 120.1551},
        "nanjing": {"lat": 32.0603, "lon": 118.7969},
    }

    key = city.lower().replace(" ", "")
    lat, lon = city_coordinates.get(key, city_coordinates["beijing"]).values()

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,dew_point_2m,pressure_msl,wind_speed_10m,wind_direction_10m,precipitation",
        "forecast_days": 3,
        "timezone": "auto",
    }
    try:
        res = requests.get("https://api.open-meteo.com/v1/cma", params=params, timeout=15)
        res.raise_for_status()
        return process_open_meteo_data(res.json(), city)
    except Exception as e:
        st.error(f"Error fetching weather data ➜ {e}")
        return None


def process_open_meteo_data(api_json: dict, city: str):
    """Convert Open‑Meteo JSON to a structure similar to OpenWeather (for legacy code)."""
    hourly = api_json.get("hourly", {})
    times = hourly.get("time", [])
    forecast = []
    for idx, ts in enumerate(times):
        dt = datetime.datetime.fromisoformat(ts)
        forecast.append(
            {
                "dt": int(dt.timestamp()),
                "dt_txt": dt.strftime("%Y-%m-%d %H:%M:%S"),
                "main": {
                    "temp": hourly["temperature_2m"][idx],
                    "pressure": hourly["pressure_msl"][idx],
                    "humidity": hourly["relative_humidity_2m"][idx],
                },
                "wind": {
                    "speed": hourly["wind_speed_10m"][idx],
                    "deg": hourly["wind_direction_10m"][idx],
                },
                "dew_point": hourly["dew_point_2m"][idx],  # NEW
                "rain": {"3h": hourly["precipitation"][idx]} if hourly["precipitation"][idx] > 0 else {},
            }
        )

    return {
        "list": forecast,
        "city": {
            "name": city,
            "coord": {"lat": api_json.get("latitude"), "lon": api_json.get("longitude")},
        },
    }

# --------------------------------------------------
# 🛠️  DATA PRE‑PROCESSING
# --------------------------------------------------

def preprocess_weather_data(raw_weather: dict) -> pd.DataFrame:
    """Turn processed API json into a tidy DataFrame expected by the ML model."""
    rows = []
    for itm in raw_weather.get("list", []):
        dt = datetime.datetime.fromtimestamp(itm["dt"])
        main = itm["main"]
        wind = itm["wind"]
        rows.append(
            {
                "datetime": dt,
                "Temp": main["temp"],          # rename to match model training
                "Press": main["pressure"],
                "Humidity": main["humidity"],  # not used by model but handy for charts
                "WindSpeed": wind["speed"],
                "DewP": itm.get("dew_point"),
            }
        )
    df = pd.DataFrame(rows)
    # Extra time‑based features if we want to plot / provide summary
    if not df.empty:
        df["Hour"] = df["datetime"].dt.hour
        df["Day"] = df["datetime"].dt.day
        df["Month"] = df["datetime"].dt.month
    return df

# --------------------------------------------------
# 🤖  LOAD TRAINED RANDOM‑FOREST MODEL
# --------------------------------------------------
MODEL_PATH = "rf_pm25_model.pkl"  # Make sure this pickle sits next to this app

@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.stop(
            f"❗ Trained model not found at '{MODEL_PATH}'. Upload it or change MODEL_PATH.")
    return joblib.load(MODEL_PATH)


def predict_pm25(model, weather_df: pd.DataFrame):
    """Generate PM2.5 predictions using the pretrained sklearn pipeline."""
    feature_cols = ["DewP", "Temp", "Press", "WindSpeed"]  # model_without_rain features
    if any(col not in weather_df.columns for col in feature_cols):
        missing = [c for c in feature_cols if c not in weather_df.columns]
        st.error(f"Missing columns for prediction: {missing}")
        return []
    preds = model.predict(weather_df[feature_cols])
    return preds.tolist()

# --------------------------------------------------
# 🖍️  HELPER FUNCTIONS FOR UI
# --------------------------------------------------

def pm25_to_aqi_category(pm25):
    if pm25 <= 12:  # µg/m³
        return "Good", "#00e400"
    elif pm25 <= 35.4:
        return "Moderate", "#ffff00"
    elif pm25 <= 55.4:
        return "Unhealthy for Sensitive Groups", "#ff7e00"
    elif pm25 <= 150.4:
        return "Unhealthy", "#ff0000"
    elif pm25 <= 250.4:
        return "Very Unhealthy", "#8f3f97"
    else:
        return "Hazardous", "#7e0023"


def generate_forecast_summary(df: pd.DataFrame):
    if df.empty:
        return "No forecast data available."
    worst_idx = df["pm25"].idxmax()
    worst_time = df.loc[worst_idx, "datetime"]
    worst_cat = df.loc[worst_idx, "aqi_category"]

    today = datetime.date.today()
    if worst_time.date() == today:
        date_str = "today"
    elif worst_time.date() == today + datetime.timedelta(days=1):
        date_str = "tomorrow"
    else:
        date_str = worst_time.strftime("%A, %B %d")

    hour = worst_time.hour
    if 6 <= hour < 12:
        period = "morning"
    elif 12 <= hour < 18:
        period = "afternoon"
    else:
        period = "evening" if hour >= 18 else "night"

    rec = ""
    if worst_cat in {"Unhealthy", "Very Unhealthy", "Hazardous"}:
        rec = "\n⛔ Outdoor activity is not recommended."
    elif worst_cat == "Unhealthy for Sensitive Groups":
        rec = "\n⚠️ Sensitive individuals should limit outdoor activity."
    return f"Air quality will be worst {date_str} in the {period}, reaching **{worst_cat}** levels.{rec}"

# --------------------------------------------------
# 🚀  MAIN APP FLOW
# --------------------------------------------------
if forecast_button:
    with st.spinner("Fetching data & predicting …"):
        weather_json = fetch_weather_forecast(city)
        if weather_json:
            weather_df = preprocess_weather_data(weather_json)
            rf_model = load_model()
            preds = predict_pm25(rf_model, weather_df)
            if preds:
                weather_df["pm25"] = preds
                weather_df["aqi_category"], weather_df["aqi_color"] = zip(*weather_df["pm25"].apply(pm25_to_aqi_category))

                # Tabs for overview / chart / table
                tab_over, tab_chart, tab_table = st.tabs(["Summary", "PM2.5 Trend", "Raw data"])

                with tab_over:
                    st.markdown(generate_forecast_summary(weather_df))

                with tab_chart:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(weather_df["datetime"], weather_df["pm25"], marker="o", lw=1.5)
                    ax.set_ylabel("PM2.5 (µg/m³)")
                    ax.set_xlabel("Time")
                    ax.set_title("Predicted PM2.5 concentration – next 72 h")
                    ax.grid(True, linestyle="--", alpha=0.4)
                    st.pyplot(fig)

                with tab_table:
                    st.dataframe(weather_df.set_index("datetime"), use_container_width=True)
