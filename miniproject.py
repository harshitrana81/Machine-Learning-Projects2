import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(page_title="Electricity Forecast", page_icon="⚡")

# --- 1. LOAD THE MODEL (Optimized) ---
@st.cache_resource
def load_model():
    # Ensure this filename matches exactly what you uploaded to GitHub
    return joblib.load('electricity_rf_model.joblib')

model = load_model()

# --- 2. DATA PREPARATION ---
st.title("⚡ Smart Grid: Electricity Demand Predictor")
st.write("Predicting real-time energy consumption using Random Forest.")

# Sidebar for inputs
st.sidebar.header("User Input Parameters")
input_date = st.sidebar.date_input("Select Date", datetime.now())

# Streamlit's time_input automatically shows AM/PM based on your browser/locale
input_time = st.sidebar.time_input("Select Time", datetime.now())

# Combine date and time
# This automatically converts 2:00 PM to hour 14 for the model
user_datetime = datetime.combine(input_date, input_time)

# --- 3. THE PREDICTION LOGIC ---
# Using the last 3 actual values from your dataset (example values)
history = [14500.0, 15200.0, 14800.0] 
initial_ma3 = sum(history) / 3

# Feature list must match the order and names used in training
features = ['hour', 'day_of_week', 'month', 'lag_1', 'ma_3']

first_row = pd.DataFrame({
    'hour': [user_datetime.hour],          # Hour will be 0-23
    'day_of_week': [user_datetime.weekday()], 
    'month': [user_datetime.month],
    'lag_1': [history[-1]], 
    'ma_3': [initial_ma3]   
})

if st.button("Predict Current Demand"):
    current_prediction = model.predict(first_row[features])[0]
    
    # Displaying time in 12-hour format with AM/PM for the user
    display_time = user_datetime.strftime('%I:%M %p') 
    
    st.metric(label=f"Predicted Demand for {display_time}", 
              value=f"{current_prediction:.2f} MW")

    st.divider()

    # --- 4. 24-HOUR FORECAST ---
    st.subheader("📅 24-Hour Forecast (Recursive)")
    
    forecast_results = []
    temp_history = history.copy()
    temp_history.append(current_prediction)
    loop_time = user_datetime

    for i in range(24):
        # pd.Timedelta handles the transition across midnight/new days perfectly
        loop_time += pd.Timedelta(hours=1)
        current_ma3 = sum(temp_history[-3:]) / 3

        row = pd.DataFrame({
            'hour': [loop_time.hour],
            'day_of_week': [loop_time.weekday()],
            'month': [loop_time.month],
            'lag_1': [temp_history[-1]],
            'ma_3': [current_ma3]  
        })

        pred = model.predict(row[features])[0]
        forecast_results.append({
            'Time': loop_time.strftime('%Y-%m-%d %I:%M %p'), # Clear AM/PM format
            'Predicted_MW': pred
        })
        temp_history.append(pred)

    forecast_df = pd.DataFrame(forecast_results)
    
    # Visualizing the trend
    st.line_chart(forecast_df.set_index('Time'))
    st.dataframe(forecast_df, use_container_width=True)
