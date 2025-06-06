# Feature Requirements for Random Forest Model

## Model Structure
The model is a scikit-learn Pipeline with two main steps:
1. `preprocessor` - A ColumnTransformer that handles both numeric and categorical features
2. `regressor` - A RandomForestRegressor with max_depth=10

## Required Features
Based on the model inspection, the following features must be provided:

### Numeric Features (processed with imputation and scaling)
- `DewP` - Dew point temperature
- `Temp` - Temperature
- `Press` - Pressure
- `WindSpeed` - Wind speed
- `HoursOfSnow` - Hours of snow
- `HoursOfRain` - Hours of rain
- `WindSpeed_Winsorized` - Winsorized wind speed (outliers capped)
- `HoursOfRain_rolling` - Rolling hours of rain
- `HoursOfSnow_rolling` - Rolling hours of snow
- `WinDir_U` - Wind direction U component
- `WinDir_V` - Wind direction V component
- `day_of_week` - Day of week (0-6)
- `day_of_year` - Day of year (1-366)
- `is_weekend` - Boolean flag for weekend
- `pm2.5_lag1` - PM2.5 value from 1 hour ago
- `pm2.5_lag2` - PM2.5 value from 2 hours ago
- `pm2.5_lag3` - PM2.5 value from 3 hours ago
- `pm2.5_lag6` - PM2.5 value from 6 hours ago
- `pm2.5_lag12` - PM2.5 value from 12 hours ago
- `pm2.5_lag24` - PM2.5 value from 24 hours ago
- `pm2.5_roll24_mean` - 24-hour rolling mean of PM2.5
- `pm2.5_roll24_std` - 24-hour rolling standard deviation of PM2.5
- `month` - Month (1-12)
- `hour` - Hour (0-23)

### Categorical Features (processed with imputation and one-hot encoding)
- `Extreme_PM2.5` - Flag for extreme PM2.5 values
- `time_of_day` - Period of day (morning, afternoon, evening, night)
- `Season` - Season (spring, summer, fall, winter)
- `Extreme_Event_VMD_shift1` - Flag for extreme events with shift

## Feature Engineering Required
The app needs to implement:

1. **Time-based features**:
   - `time_of_day` - Categorize hours into periods
   - `is_weekend` - Calculate from day of week
   - `day_of_week` - Extract from datetime
   - `day_of_year` - Extract from datetime
   - `Season` - Determine from month

2. **Wind direction components**:
   - `WinDir_U` - U component of wind direction
   - `WinDir_V` - V component of wind direction

3. **Weather-related features**:
   - `HoursOfSnow` - Must be derived or simulated
   - `HoursOfRain` - Must be derived or simulated
   - `HoursOfSnow_rolling` - Rolling calculation
   - `HoursOfRain_rolling` - Rolling calculation

4. **PM2.5 statistics**:
   - Lag features (1, 2, 3, 6, 12, 24 hours)
   - Rolling statistics (mean, std over 24 hours)
   - `Extreme_PM2.5` - Flag for extreme values
   - `Extreme_Event_VMD_shift1` - Shifted extreme event flag

5. **Winsorization**:
   - `WindSpeed_Winsorized` - Cap outliers in wind speed
