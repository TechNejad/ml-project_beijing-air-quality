# Beijing Air Quality Forecast: A Machine Learning Project

This repository contains the complete work for a machine learning project that analyzes and forecasts air pollution in Beijing, using data from 2010 to 2014. The project culminates in a live, interactive web application that provides real-time air quality forecasts.

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

## Repository Structure

*   `Streamlit_WebApp_Demo.py`: The main Streamlit application file.
*   `rf_pm25_model.pkl`: The trained RandomForest model.
*   `ML_project_China_Air_Pollution_.ipynb`: The Jupyter Notebook containing the complete data analysis, feature engineering, and model training process.
*   `requirements.txt`: The Python dependencies for the project.
*   `runtime.txt`: The Python runtime version for deployment.

---

## How to Run the App Locally

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/TechNejad/ml-project_beijing-air-quality.git
    cd ml-project_beijing-air-quality
    ```

2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Streamlit app:**
    ```bash
    streamlit run Streamlit_WebApp_Demo.py
    ```
