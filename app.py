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
</style>
""", unsafe_allow_html=True)

# App title only
st.title("Air Pollution Forecast")

# Sidebar for inputs
with st.sidebar:
    st.header("Settings")
    
    # City selector (default: Beijing)
    city = st.text_input("City", value="Beijing")
    
    # Forecast button
    forecast_button = st.button("Get Pollution Forecast", type="primary")

# Function to fetch weather forecast data from Open-Meteo
def fetch_weather_forecast(city):
    """Get weather forecast data from Open-Meteo CMA API"""
    # Default coordinates for common cities
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
        "nanjing": {"lat": 32.0603, "lon": 118.7969}
    }
    
    # Use provided coordinates or look up by city name
    city_key = city.lower().replace(" ", "")
    if city_key in city_coordinates:
        lat = city_coordinates[city_key]["lat"]
        lon = city_coordinates[city_key]["lon"]
    else:
        # Default to Beijing if city not found
        lat = city_coordinates["beijing"]["lat"]
        lon = city_coordinates["beijing"]["lon"]
    
    try:
        # Define the API endpoint
        url = "https://api.open-meteo.com/v1/cma"
        
        # Define the parameters
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,relative_humidity_2m,dew_point_2m,pressure_msl,surface_pressure,precipitation,wind_speed_10m,wind_direction_10m",
            "forecast_days": 3,
            "timezone": "auto"
        }
        
        # Make the API request
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise exception for HTTP errors
        data = response.json()
        
        # Process the data into a format compatible with our model
        processed_data = process_open_meteo_data(data, city)
        return processed_data
    
    except Exception as e:
        st.error(f"Error fetching weather data: {e}")
        return None

def process_open_meteo_data(data, city):
    """
    Process Open-Meteo API response into a format compatible with our model
    """
    # Extract hourly data
    hourly = data.get("hourly", {})
    
    # Create a list of forecast items
    forecast_list = []
    
    # Get the time values
    time_values = hourly.get("time", [])
    
    for i in range(len(time_values)):
        # Convert ISO time string to timestamp
        dt_str = time_values[i]
        dt = datetime.datetime.fromisoformat(dt_str)
        
        # Extract weather variables for this time step
        temp = hourly.get("temperature_2m", [])[i]
        humidity = hourly.get("relative_humidity_2m", [])[i]
        dew_point = hourly.get("dew_point_2m", [])[i]
        pressure = hourly.get("pressure_msl", [])[i]
        precipitation = hourly.get("precipitation", [])[i]
        wind_speed = hourly.get("wind_speed_10m", [])[i]
        wind_direction = hourly.get("wind_direction_10m", [])[i]
        
        # Create forecast item in a format similar to OpenWeather API
        forecast_item = {
            "dt": dt,
            "Temp": temp,
            "DewP": dew_point,
            "Press": pressure,
            "WindSpeed": wind_speed,
            "WinDir": wind_direction,
            "HoursOfRain": precipitation,
            "HoursOfSnow": 0  # Assuming no snow data from this API
        }
        
        forecast_list.append(forecast_item)
    
    return forecast_list

# Function to preprocess weather data for model input
def preprocess_for_model(df):
    """Transform weather data into format expected by the model"""
    
    # Rename columns to match model's expected input
    df = df.rename(columns={
        "dt": "datetime",
    })
    df.set_index('datetime', inplace=True)

    # Feature Engineering from the notebook
    df['WindSpeed_Winsorized'] = df['WindSpeed'] # Simplified for the app
    df['HoursOfRain_rolling'] = df['HoursOfRain'].rolling(window=24, min_periods=1).sum()
    df['HoursOfSnow_rolling'] = df['HoursOfSnow'].rolling(window=24, min_periods=1).sum()

    wind_direction_mapping = {
        'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
        'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
        'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
        'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5,
        'cv': 0
    }
    
    # Convert wind direction degrees to categories for mapping
    def degrees_to_cardinal(d):
        dirs = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
        ix = int(round(d / (360. / len(dirs))))
        return dirs[ix % len(dirs)]

    df['WinDir_cat'] = df['WinDir'].apply(degrees_to_cardinal)
    df['WinDir_degrees'] = df['WinDir_cat'].str.lower().map(wind_direction_mapping)
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
    
    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=['Season', 'time_of_day'], drop_first=True)

    return df

# Function to predict PM2.5 values
def predict_pm25(weather_df):
    """Use the loaded model to predict PM2.5 values"""
    
    # The model expects a specific set of features.
    # We need to create a dataframe with these features.
    # The order of columns must match the training data.
    
    # These are the features the model was trained on
    model_features = [
        'DewP', 'Temp', 'Press', 'WindSpeed', 'HoursOfSnow', 'HoursOfRain',
        'WindSpeed_Winsorized', 'HoursOfRain_rolling', 'HoursOfSnow_rolling',
        'WinDir_U', 'WinDir_V', 'day_of_week', 'day_of_year', 'is_weekend',
        'month', 'hour', 'Season_Fall', 'Season_Spring', 'Season_Summer',
        'Season_Winter', 'time_of_day_Evening', 'time_of_day_Morning',
        'time_of_day_Night'
    ]

    # Create a dataframe with the correct columns, initialized to 0
    X = pd.DataFrame(0, index=weather_df.index, columns=model_features)

    # Fill in the values from the preprocessed weather_df
    for col in weather_df.columns:
        if col in X.columns:
            X[col] = weather_df[col]

    # Handle the one-hot encoded columns
    if 'Season_Fall' in weather_df.columns: X['Season_Fall'] = weather_df['Season_Fall']
    if 'Season_Spring' in weather_df.columns: X['Season_Spring'] = weather_df['Season_Spring']
    if 'Season_Summer' in weather_df.columns: X['Season_Summer'] = weather_df['Season_Summer']
    if 'Season_Winter' in weather_df.columns: X['Season_Winter'] = weather_df['Season_Winter']
    if 'time_of_day_Evening' in weather_df.columns: X['time_of_day_Evening'] = weather_df['time_of_day_Evening']
    if 'time_of_day_Morning' in weather_df.columns: X['time_of_day_Morning'] = weather_df['time_of_day_Morning']
    if 'time_of_day_Night' in weather_df.columns: X['time_of_day_Night'] = weather_df['time_of_day_Night']
    
    # Ensure all columns are present, fill missing with 0
    for col in model_features:
        if col not in X.columns:
            X[col] = 0
            
    # Reorder columns to match the model's training order
    X = X[model_features]

    # Make predictions
    predictions = model.predict(X)
    return predictions

# Function to convert PM2.5 values to AQI categories
def pm25_to_aqi_category(pm25):
    """Convert PM2.5 concentration to AQI category"""
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
def generate_forecast_summary(weather_df, pm25_predictions):
    """Generate a condensed human-friendly summary of the forecast"""
    if weather_df.empty or not pm25_predictions.any():
        return "No forecast data available."
    
    # Combine datetime and predictions
    forecast_df = pd.DataFrame({
        "datetime": weather_df.index,
        "pm25": pm25_predictions
    })
    
    # Add AQI categories
    forecast_df["aqi_category"], forecast_df["aqi_color"] = zip(*forecast_df["pm25"].apply(pm25_to_aqi_category))
    
    # Find the worst AQI category overall
    overall_worst_idx = forecast_df["pm25"].idxmax()
    overall_worst_category = forecast_df.loc[overall_worst_idx, "aqi_category"]
    worst_time = forecast_df.loc[overall_worst_idx, "datetime"]
    
    # Format the date and time
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
    
    # Create a single sentence summary
    summary = [f"Air quality will be worst on {date_str} during {period}, reaching {overall_worst_category} levels."]
    
    # Add health recommendation
    if overall_worst_category in ["Unhealthy", "Very Unhealthy", "Hazardous"]:
        summary.append("\n⛔ Outdoor activity is not recommended.")
    elif overall_worst_category == "Unhealthy for Sensitive Groups":
        summary.append("\n⚠️ Sensitive individuals should limit outdoor activity.")
    
    return "\n\n".join(summary)

# Main app logic
if forecast_button:
    with st.spinner("Fetching weather data and generating forecast..."):
            # Fetch weather data
            weather_data = fetch_weather_forecast(city)
            
            if weather_data:
                # Preprocess weather data
                weather_df = pd.DataFrame(weather_data)
                weather_df_processed = preprocess_for_model(weather_df.copy())
                
                if not weather_df_processed.empty:
                    # Predict PM2.5 values
                    pm25_predictions = predict_pm25(weather_df_processed)
                    
                    if pm25_predictions is not None:
                        # Create tabs for different views with custom CSS to fix visual bug
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
                        
                        tab1, tab2 = st.tabs(["Forecast Chart", "Detailed Data"])
                        
                        with tab1:
                            # Create a DataFrame for plotting
                            plot_df = pd.DataFrame({
                                "datetime": weather_df_processed.index,
                                "pm25": pm25_predictions
                            })
                            
                            # Create the plot with light theme
                            fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
                            ax.set_facecolor('white')
                            
                            # Plot the PM2.5 values
                            ax.plot(plot_df["datetime"], plot_df["pm25"], marker='o', color='black', linewidth=2)
                            
                            # Calculate dynamic y-axis range based on data
                            max_pm25 = max(plot_df["pm25"]) * 1.2  # Add 20% padding
                            
                            # Determine which AQI bands to show based on data range
                            if max_pm25 <= 60:  # If max is in Unhealthy for Sensitive Groups or lower
                                y_max = max(60, max_pm25)  # Show at least up to Unhealthy for Sensitive Groups
                                ax.axhspan(0, 12, alpha=0.3, color='#00e400', label='Good')
                                ax.axhspan(12, 35.4, alpha=0.3, color='#ffff00', label='Moderate')
                                ax.axhspan(35.4, 55.4, alpha=0.3, color='#ff7e00', label='Unhealthy for Sensitive Groups')
                                ax.set_ylim(0, y_max)
                            elif max_pm25 <= 155:  # If max is in Unhealthy range
                                y_max = max(155, max_pm25)  # Show at least up to Unhealthy
                                ax.axhspan(0, 12, alpha=0.3, color='#00e400', label='Good')
                                ax.axhspan(12, 35.4, alpha=0.3, color='#ffff00', label='Moderate')
                                ax.axhspan(35.4, 55.4, alpha=0.3, color='#ff7e00', label='Unhealthy for Sensitive Groups')
                                ax.axhspan(55.4, 150.4, alpha=0.3, color='#ff0000', label='Unhealthy')
                                ax.set_ylim(0, y_max)
                            elif max_pm25 <= 255:  # If max is in Very Unhealthy range
                                y_max = max(255, max_pm25)  # Show at least up to Very Unhealthy
                                ax.axhspan(0, 12, alpha=0.3, color='#00e400', label='Good')
                                ax.axhspan(12, 35.4, alpha=0.3, color='#ffff00', label='Moderate')
                                ax.axhspan(35.4, 55.4, alpha=0.3, color='#ff7e00', label='Unhealthy for Sensitive Groups')
                                ax.axhspan(55.4, 150.4, alpha=0.3, color='#ff0000', label='Unhealthy')
                                ax.axhspan(150.4, 250.4, alpha=0.3, color='#8f3f97', label='Very Unhealthy')
                                ax.set_ylim(0, y_max)
                            else:  # If max is in Hazardous range
                                y_max = max(500, max_pm25)  # Show at least up to 500 for Hazardous
                                ax.axhspan(0, 12, alpha=0.3, color='#00e400', label='Good')
                                ax.axhspan(12, 35.4, alpha=0.3, color='#ffff00', label='Moderate')
                                ax.axhspan(35.4, 55.4, alpha=0.3, color='#ff7e00', label='Unhealthy for Sensitive Groups')
                                ax.axhspan(55.4, 150.4, alpha=0.3, color='#ff0000', label='Unhealthy')
                                ax.axhspan(150.4, 250.4, alpha=0.3, color='#8f3f97', label='Very Unhealthy')
                                ax.axhspan(250.4, y_max, alpha=0.3, color='#7e0023', label='Hazardous')
                                ax.set_ylim(0, y_max)
                            
                            # Add labels and title with improved styling for light theme
                            ax.set_xlabel('Time', color='black', fontsize=12)
                            ax.set_ylabel('PM2.5 (μg/m³)', color='black', fontsize=12)
                            ax.set_title(f'Predicted Air Pollution Levels for {city}', color='black', fontsize=14, pad=20)
                            
                            # Improve x-axis readability
                            date_format = plt.matplotlib.dates.DateFormatter('%m-%d\n%H:00')
                            ax.xaxis.set_major_formatter(date_format)
                            ax.xaxis.set_major_locator(plt.matplotlib.dates.HourLocator(interval=12))
                            plt.xticks(color='black', rotation=0)
                            
                            # Add grid for better trend visibility
                            ax.grid(True, linestyle='--', alpha=0.3, color='gray')
                            
                            # Style y-axis ticks
                            plt.yticks(color='black')
                            
                            # Remove legend from the figure as requested
                            if ax.get_legend():
                                ax.get_legend().remove()
                            
                            # Add trend indicator arrow for overall trend
                            if len(plot_df) > 1:
                                first_val = plot_df["pm25"].iloc[0]
                                last_val = plot_df["pm25"].iloc[-1]
                                trend_text = "↗️ Rising" if last_val > first_val else "↘️ Falling" if last_val < first_val else "→ Stable"
                                ax.text(0.02, 0.98, trend_text, transform=ax.transAxes, 
                                       color='black', fontsize=12, verticalalignment='top',
                                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='lightgray'))
                            
                            # Adjust layout to make room for the legend
                            plt.tight_layout()
                            plt.subplots_adjust(right=0.75)
                            
                            # Display the plot
                            st.pyplot(fig)
                            
                            # Display forecast summary
                            st.header("Forecast Summary")
                            summary = generate_forecast_summary(weather_df_processed, pm25_predictions)
                            st.markdown(summary)
                            
                            # Display AQI categories legend with colorful boxes
                            st.header("AQI Categories Legend")
                            
                            # Create a more visually appealing legend with colored boxes
                            aqi_categories = [
                                {"name": "Good", "color": "#00e400", "range": "0-12", "description": "Air quality is satisfactory, poses little or no risk."},
                                {"name": "Moderate", "color": "#ffff00", "range": "12.1-35.4", "description": "Acceptable air quality, but moderate health concern for very sensitive individuals."},
                                {"name": "Unhealthy for Sensitive Groups", "color": "#ff7e00", "range": "35.5-55.4", "description": "Members of sensitive groups may experience health effects."},
                                {"name": "Unhealthy", "color": "#ff0000", "range": "55.5-150.4", "description": "Everyone may begin to experience health effects."},
                                {"name": "Very Unhealthy", "color": "#8f3f97", "range": "150.5-250.4", "description": "Health alert: everyone may experience more serious health effects."},
                                {"name": "Hazardous", "color": "#7e0023", "range": "250.5+", "description": "Health warning of emergency conditions, entire population likely affected."}
                            ]
                            
                            # Create a more visually appealing legend with colored boxes using Streamlit columns
                            cols = st.columns(3)
                            
                            for i, category in enumerate(aqi_categories):
                                col_idx = i % 3
                                with cols[col_idx]:
                                    # Determine text color based on background for optimal readability
                                    text_color = 'black'
                                    if category['name'] in ['Unhealthy', 'Very Unhealthy', 'Hazardous']:
                                        text_color = 'white'
                                    elif category['name'] == 'Moderate':
                                        text_color = '#333333'  # Darker text for yellow background
                                        
                                    # Create a colored box with category information - improved styling
                                    st.markdown(
                                        f"""
                                        <div style="background-color: {category['color']}; 
                                                    padding: 15px; 
                                                    border-radius: 8px; 
                                                    margin-bottom: 15px;
                                                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                                                    color: {text_color};">
                                            <h3 style="margin-top: 0; margin-bottom: 8px; font-weight: 600;">{category['name']}</h3>
                                            <div style="font-size: 1.0em; margin-bottom: 8px; font-weight: 500;">PM2.5: {category['range']} μg/m³</div>
                                            <div style="font-size: 0.9em; line-height: 1.4;">{category['description']}</div>
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )
                        
                        with tab2:
                            # Display detailed forecast data
                            detailed_df = pd.DataFrame({
                                "Date & Time": weather_df_processed.index,
                                "Temperature (°C)": weather_df_processed["Temp"],
                                "Humidity (%)": weather_df_processed.get("humidity", "N/A"), # Added .get for safety
                                "Wind Speed (km/h)": weather_df_processed["WindSpeed"],
                                "Pressure (hPa)": weather_df_processed["Press"],
                                "Predicted PM2.5 (μg/m³)": pm25_predictions
                            })
                            
                            # Add AQI category
                            detailed_df["AQI Category"] = [pm25_to_aqi_category(pm25)[0] for pm25 in pm25_predictions]
                            
                            # Display the detailed data
                            st.dataframe(detailed_df, hide_index=True)
                    else:
                        st.error("Failed to generate PM2.5 predictions.")
                else:
                    st.error("Failed to process weather data.")
            else:
                st.error("Failed to fetch weather data.")

# Footer
st.markdown("---")
