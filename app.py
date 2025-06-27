import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the trained model
with open('rf_pm25_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title('Beijing Air Quality Predictor')

st.sidebar.title('Input Features')

# Create input fields for all the features required by the model
def user_input_features():
    year = st.sidebar.slider('Year', 2013, 2017, 2015)
    month = st.sidebar.slider('Month', 1, 12, 6)
    day = st.sidebar.slider('Day', 1, 31, 15)
    hour = st.sidebar.slider('Hour', 0, 23, 12)
    season = st.sidebar.selectbox('Season', (1, 2, 3, 4), index=2)  # Assuming 1:Spring, 2:Summer, 3:Autumn, 4:Winter
    dehumidification = st.sidebar.number_input('DEWP', -40.0, 40.0, 10.0)
    temperature = st.sidebar.number_input('TEMP', -20.0, 50.0, 25.0)
    pressure = st.sidebar.number_input('PRES', 900.0, 1100.0, 1015.0)
    combined_wind_direction = st.sidebar.selectbox('Combined Wind Direction', ('NW', 'SE', 'NE', 'cv'), index=0)
    wind_speed = st.sidebar.number_input('Iws', 0.0, 600.0, 10.0)
    snow = st.sidebar.number_input('Is', 0, 50, 0)
    rain = st.sidebar.number_input('Ir', 0, 50, 0)

    data = {'year': year,
            'month': month,
            'day': day,
            'hour': hour,
            'season': season,
            'DEWP': dehumidification,
            'TEMP': temperature,
            'PRES': pressure,
            'cbwd_NW': 1 if combined_wind_direction == 'NW' else 0,
            'cbwd_SE': 1 if combined_wind_direction == 'SE' else 0,
            'cbwd_NE': 1 if combined_wind_direction == 'NE' else 0,
            'cbwd_cv': 1 if combined_wind_direction == 'cv' else 0,
            'Iws': wind_speed,
            'Is': snow,
            'Ir': rain}
    
    # The model expects these specific columns in this order
    feature_cols = ['year', 'month', 'day', 'hour', 'season', 'DEWP', 'TEMP', 'PRES', 
                    'Iws', 'Is', 'Ir', 'cbwd_NE', 'cbwd_NW', 'cbwd_SE', 'cbwd_cv']

    features = pd.DataFrame(data, index=[0])
    features = features[feature_cols] # Ensure correct column order

    return features

input_df = user_input_features()

st.subheader('User Input features')
st.write(input_df)

# Make prediction
prediction = model.predict(input_df)

st.subheader('Prediction')
st.write(f'Predicted PM2.5: {prediction[0]:.2f}')

