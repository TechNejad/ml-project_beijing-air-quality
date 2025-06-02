import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import requests, os, joblib, pytz, re

# ---------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------
st.set_page_config(page_title="Air Pollution Forecast", page_icon="🌬️", layout="wide")

MODEL_PATH      = "rf_pm25_model.pkl"
FORECAST_HOURS  = 72   # predict next 72 h
HISTORY_HOURS   = 24   # hours of real pm2.5 for lag features

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
AIRQUAL_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

# Background AQI colour bands (µg/m³)
aqi_bands = [
    (0,12,"#b7f4b0"), (12.1,35.4,"#ffff9c"), (35.5,55.4,"#ffcd96"),
    (55.5,150.4,"#ff9d9d"), (150.5,250.4,"#c99ee0"), (250.5,500,"#a285c3")
]

# Map Open‑Meteo variable → column name used during model training
WEATHER_RENAME = {
    "temperature_2m": "Temp",
    "dew_point_2m":  "DewP",
    "pressure_msl":  "Press",
    "wind_speed_10m":"WindSpeed",
    "wind_direction_10m":"WindDir",
    "relative_humidity_2m":"Humidity",
}

# ---------------------------------------------------------------
# API HELPERS
# ---------------------------------------------------------------

def geocode_city(city:str):
    r = requests.get(GEOCODE_URL, params={"name":city,"count":1}, timeout=15)
    r.raise_for_status()
    res = r.json().get("results")
    if not res:
        raise ValueError("City not found")
    d = res[0]
    return d["latitude"], d["longitude"], d["timezone"], d["name"]


def fetch_weather(lat, lon, tz):
    params = {
        "latitude":lat, "longitude":lon, "timezone":tz,
        "hourly": ",".join(WEATHER_RENAME.keys()),
        "forecast_hours": FORECAST_HOURS,
    }
    r = requests.get(WEATHER_URL, params=params, timeout=15)
    r.raise_for_status()
    j = r.json()["hourly"]
    df = pd.DataFrame(j).rename(columns=WEATHER_RENAME)
    df["time"] = pd.to_datetime(df["time"])
    return df


def fetch_pm25_history(lat, lon, tz):
    params = {
        "latitude":lat, "longitude":lon, "timezone":tz,
        "hourly":"pm2_5", "past_days":3, "forecast_hours":FORECAST_HOURS
    }
    r = requests.get(AIRQUAL_URL, params=params, timeout=15)
    r.raise_for_status()
    j = r.json()["hourly"]
    df = pd.DataFrame(j)
    df["time"] = pd.to_datetime(df["time"])
    return df

# ---------------------------------------------------------------
# MODEL & FEATURE LOGIC
# ---------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found – upload rf_pm25_model.pkl")
        st.stop()
    return joblib.load(MODEL_PATH)

model = load_model()
pre_feats = list(model.named_steps["preprocessor"].feature_names_in_)
lag_cols  = [c for c in pre_feats if "lag"  in c]
roll_cols = [c for c in pre_feats if "roll" in c]


def make_lag_features(history:list):
    """Return a dict of {col:value} for all lag/roll cols expected by the model."""
    feats = {}
    for col in lag_cols:
        lag = int(re.findall(r"\d+", col)[0])          # works for lag1 / lag_1 / lag1h
        feats[col] = history[-lag] if len(history) >= lag else history[0]
    for col in roll_cols:
        win = int(re.findall(r"\d+", col)[0])          # works for roll24 / roll24h
        feats[col] = float(np.mean(history[-win:])) if history else np.nan
    return feats

# ---------------------------------------------------------------
# DATAFRAME BUILDER (recursive forecast)
# ---------------------------------------------------------------

def build_future_dataframe(wx_df, aq_df, tz):
    # Merge to ensure we have past PM2.5 for HISTORY_HOURS
    aq_df = aq_df.sort_values("time")
    history_pm = aq_df["pm2_5"].tail(HISTORY_HOURS).tolist()

    rows = []
    for idx in range(FORECAST_HOURS):
        row_time = wx_df.loc[idx, "time"]
        base = {
            "datetime": row_time,
            "Year":  row_time.year,
            "Month": row_time.month,
            "Day":   row_time.day,
            "Hour":  row_time.hour,
            "Weekday": row_time.weekday(),
        }
        # attach weather columns (renamed already)
        for trg_col in WEATHER_RENAME.values():
            base[trg_col] = wx_df.loc[idx, trg_col]

        # add lag / rolling features
        base.update(make_lag_features(history_pm))

        # dataframe for single prediction
        pred_df = pd.DataFrame([base])
        yhat = float(model.predict(pred_df)[0])
        history_pm.append(yhat)
        base["PM2.5_pred"] = yhat
        rows.append(base)
    return pd.DataFrame(rows)

# ---------------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------------
with st.sidebar:
    city_inp = st.text_input("City", "Beijing")
    if st.button("Get Pollution Forecast", type="primary"):
        st.session_state["go"] = True

if st.session_state.get("go"):
    try:
        lat, lon, tz, proper = geocode_city(city_inp)
    except Exception as e:
        st.error(f"Geocoder error: {e}")
        st.stop()

    with st.spinner("Fetching data …"):
        try:
            wx_json = fetch_weather(lat, lon, tz)
            aq_json = fetch_pm25_history(lat, lon, tz)
        except Exception as e:
            st.error(f"API error: {e}")
            st.stop()

    df_future = build_future_dataframe(wx_json, aq_json, tz)

    st.subheader(f"Predicted PM2.5 for {proper} – next {FORECAST_HOURS} h")

    # Plot
    fig, ax = plt.subplots(figsize=(10,4))
    for lo, hi, col in aqi_bands:
        ax.axhspan(lo, hi, color=col, alpha=0.25)
    ax.plot(df_future["datetime"], df_future["PM2.5_pred"], marker="o", color="black")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.set_xlabel("Time")
    ax.grid(ls="--", alpha=0.3)
    st.pyplot(fig)

    with st.expander("Detailed data"):
        st.dataframe(df_future[["datetime","PM2.5_pred"]].rename(columns={"PM2.5_pred":"PM2.5 (µg/m³)"}), use_container_width=True)
