# Beijing Air Quality Forecast: A Machine Learning Project

This ML project analyzes Beijing's air quality from 2010 to 2014, focusing on predicting PM2.5 levels and their correlation with weather and seasonal factors. We trained and evaluated a Random Forest model, which we then used to create a real-time air quality web app demo providing real-time air quality information. This was a joint project for the 'Fundamentals of ML' and ' Applied Programming for Data Science' courses in the Artificial Intelligence for Sustainable Societies Master's program.

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
