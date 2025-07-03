# Beijing Air Quality Forecast: A Machine Learning Project

This machine learning project explores air quality in Beijing from 2010 to 2014, focusing on predicting fine particulate matter (PM2.5) and how it evolves in relation to meteorological and seasonal factors. The dataset is a time series, with hourly readings of PM2.5 and weather conditions such as temperature, wind speed, and pressure. The project culminates in a live, interactive web application that provides real-time air quality information to users.

This was a joint project for our Fundamentals of Machine Learning and Applied Programming for Data Science courses as part of the Artificial Intelligence For Sustainable Socieities (AISS) Master's program.
---

## Live Demo

<a href="https://ml-projectbeijing-air-quality-gqgd3juiatrt7eop7cdrwb.streamlit.app/" target="_blank">
    <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Streamlit App"/>
</a>

---

## Project Overview

This project follows the CRISP-DM methodology to deliver a comprehensive analysis of Beijing's air quality. It includes:

*   **Data Cleaning and Preprocessing:** Handling missing values, outliers, and inconsistencies in the dataset.
*   **Exploratory Data Analysis (EDA):** Uncovering temporal and meteorological patterns in the data.
*   **Feature Engineering:** Creating new features to improve model performance, including lagged features, rolling windows, and VMD-based volatility flags.
*   **Machine Learning Modeling:** Training and evaluating multiple regression models, including RandomForest, XGBoost, SVR, and KNN.
*   **Hyperparameter Tuning:** Optimizing model performance using GridSearchCV and TimeSeriesSplit.
*   **Web Application:** A live, interactive Streamlit app that uses the best-performing model to provide real-time air quality forecasts.

---

## Key Technologies

*   **Python:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
*   **Machine Learning:** RandomForest, XGBoost, SVR, KNN
*   **Web Framework:** Streamlit
*   **Deployment:** Streamlit Community Cloud

---
