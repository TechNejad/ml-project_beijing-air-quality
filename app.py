import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import requests, os, joblib, pytz

# ------------------------------------------------------------------
# 📄  PAGE CONFIG
# ------------------------------------------------------------------
st.set_page_config(page_title="Air Pollution Forecast", page_icon="🌬️", layout="wide")

st.title("Air Pollution Forecast")

# ------------------------------------------------------------------
# 🔧  CONSTANTS & HELPERS
# ------------------------------------------------------------------
MODEL_PATH = "rf_pm25_model.pkl"  # → same folder as this file
FORECAST_HOURS = 72              # how far ahead we predict
HISTORY_HOURS  = 24              # pm2.5 history window for lags

# -- AQI colour bands for quick chart shading
aqi_bands = [(0,12,"#b8f4a8"), (12.1,35.4,"#ffffa1"), (35.5,55.4,"#ffd48c"),
             (55.5,150.4,"#ff9999"), (150.5,250.4,"#c79ddf"), (250.5,500,"#a07cc8")]

# ------------------------------------------------------------------
# ☁️  OPEN‑METEO ENDPOINTS
# ------------------------------------------------------------------
GEOCODE_URL   = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL   = "https://api.open-meteo.com/v1/forecast"
AIRQUAL_URL   = "https://air-quality-api.open-meteo.com/v1/air-quality"

def geocode_city(city: str):
    resp = requests.get(GEOCODE_URL, params={"name": city, "count": 1})
    resp.raise_for_status()
    results = resp.json().get("results")
    if not results:
        raise ValueError("City not found in Open‑Meteo geocoder")
    r = results[0]
    return r["latitude"], r["longitude"], r["timezone"], r["name"]


def fetch_weather(lat, lon, timezone):
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,dew_point_2m,pressure_msl,wind_speed_10m,wind_direction_10m",
        "forecast_hours": FORECAST_HOURS,
        "timezone": timezone,
    }
    r = requests.get(WEATHER_URL, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def fetch_pm25_history(lat, lon, timezone):
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "pm2_5",
        "timezone": timezone,
        "past_days": 3,         # get 72 h back (max 3)
        "forecast_hours": FORECAST_HOURS,
    }
    r = requests.get(AIRQUAL_URL, params=params, timeout=15)
    r.raise_for_status()
    return r.json()

# ------------------------------------------------------------------
# 🤖  MODEL
# ------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found – upload rf_pm25_model.pkl")
        st.stop()
    return joblib.load(MODEL_PATH)

model = load_model()
pre_feat_names = list(model.named_steps["preprocessor"].feature_names_in_)
# Which lag / roll cols does the model need?
lag_cols  = [c for c in pre_feat_names if "lag"  in c]
roll_cols = [c for c in pre_feat_names if "roll" in c]

# Helpers to compute lags / rolls given history list
def make_lag_features(history):
    feats = {}
    for col in lag_cols:
        # expect name like "pm2.5_lag1" → lag=1
        lag = int(col.split("lag")[-1])
        feats[col] = history[-lag] if len(history) >= lag else history[0]
    for col in roll_cols:
        # expect name like "pm2.5_roll24" → window=24
        win = int(col.split("roll")[-1])
        feats[col] = float(np.mean(history[-win:])) if len(history) >= 1 else history[-1]
    return feats

# ------------------------------------------------------------------
# 🏗️  FEATURE CONSTRUCTION
# ------------------------------------------------------------------

def build_future_dataframe(weather_json, pm_json, tz):
    # Weather future block
    w_hr   = weather_json["hourly"]
    times  = [dt.datetime.fromisoformat(t).replace(tzinfo=pytz.timezone(tz)) for t in w_hr["time"]][:FORECAST_HOURS]
    # Past PM2.5 block (last HISTORY_HOURS)
    aq_hr  = pm_json["hourly"]
    past_times = [dt.datetime.fromisoformat(t).replace(tzinfo=pytz.timezone(tz)) for t in aq_hr["time"]]
    past_vals  = aq_hr["pm2_5"]
    # take the last HISTORY_HOURS actual values as history seed
    history_pm25 = past_vals[-HISTORY_HOURS:]

    rows, preds = [], []
    for idx, t in enumerate(times):
        # Base weather/time features
        row = {
            "datetime": t,
            "DewP":   w_hr["dew_point_2m"][idx],
            "Temp":   w_hr["temperature_2m"][idx],
            "Press":  w_hr["pressure_msl"][idx],
            "WindSpeed": w_hr["wind_speed_10m"][idx],
            "WindDir":   w_hr["wind_direction_10m"][idx],
            "Humidity":  w_hr["relative_humidity_2m"][idx],
            "Hour":  t.hour,
            "Month": t.month,
            "Weekday": t.weekday(),
        }
        # Lag / rolling features from current history list
        row.update(make_lag_features(history_pm25))

        # Predict pm25 for this hour
        df_one = pd.DataFrame([row])
        pred = float(model.predict(df_one)[0])
        row["pm25_pred"] = pred
        preds.append(pred)
        rows.append(row)

        # append pred to history for next step's lags
        history_pm25.append(pred)
    return pd.DataFrame(rows)

# ------------------------------------------------------------------
# 🖼️  UI
# ------------------------------------------------------------------
with st.sidebar:
    city_name = st.text_input("City", value="Beijing")
    go_btn = st.button("Get Pollution Forecast", type="primary")

if go_btn:
    try:
        lat, lon, tz, proper_city = geocode_city(city_name)
    except Exception as e:
        st.error(f"Geocoder error – {e}")
        st.stop()

    with st.spinner("Fetching weather & air‑quality data …"):
        try:
            w_json  = fetch_weather(lat, lon, tz)
            aq_json = fetch_pm25_history(lat, lon, tz)
        except Exception as e:
            st.error(f"API error – {e}")
            st.stop()

    df_future = build_future_dataframe(w_json, aq_json, tz)

    st.subheader(f"Predicted Air Pollution Levels for {proper_city}")
    # Chart with coloured band
    fig, ax = plt.subplots(figsize=(10,4))
    # colour background
    for low, high, col in aqi_bands:
        ax.axhspan(low, high, color=col, alpha=0.25)
    ax.plot(df_future["datetime"], df_future["pm25_pred"], marker="o", color="black")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.set_xlabel("Time")
    ax.grid(ls="--", alpha=0.3)
    st.pyplot(fig)

    # Data tab
    with st.expander("Detailed Data"):
        st.dataframe(df_future[["datetime","pm25_pred"]].rename(columns={"pm25_pred":"PM2.5 µg/m³"}), use_container_width=True)
