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
    return joblib.load('electricity_rf_model(1).joblib')

model = load_model()

# --- 2. DATA PREPARATION ---
# Note: In a real app, you'd load your latest CSV here to get the 'history'
# For this example, we will use placeholder 'history' data
st.title("⚡ Smart Grid: Electricity Demand Predictor")
st.write("Predicting real-time energy consumption using Random Forest.")

# Sidebar for inputs
st.sidebar.header("User Input Parameters")
input_date = st.sidebar.date_input("Select Date", datetime.now())
input_time = st.sidebar.time_input("Select Time", datetime.now())

# Combine date and time
user_datetime = datetime.combine(input_date, input_time)

# --- 3. THE PREDICTION LOGIC ---
# In your Colab, history was df['DUQ_MW'].tail(3).tolist()
# We'll simulate this with sample values (Update these with your real data)
history = [14500.0, 15200.0, 14800.0] 

initial_ma3 = sum(history) / 3

# Define the features exactly as they were during model training
features = ['hour', 'day_of_week', 'month', 'lag_1', 'ma_3']

first_row = pd.DataFrame({
    'hour': [user_datetime.hour],
    'day_of_week': [user_datetime.weekday()], # Monday=0, Sunday=6
    'month': [user_datetime.month],
    'lag_1': [history[-1]], 
    'ma_3': [initial_ma3]   
})

if st.button("Predict Current Demand"):
    current_prediction = model.predict(first_row[features])[0]
    
    st.metric(label=f"Predicted Demand for {user_datetime.strftime('%H:%M')}", 
              value=f"{current_prediction:.2f} MW")

    st.divider()

    # --- 4. 24-HOUR FORECAST ---
    st.subheader("📅 24-Hour Forecast (Recursive)")
    
    forecast_results = []
    temp_history = history.copy()
    temp_history.append(current_prediction)
    loop_time = user_datetime

    for i in range(24):
        loop_time += pd.Timedelta(hours=1)
        current_ma3 = sum(temp_history[-3:]) / 3

        row = pd.DataFrame({
            'hour': [loop_time.hour],
            'day_of_week': [loop_time.dayofweek],
            'month': [loop_time.month],
            'lag_1': [temp_history[-1]],
            'ma_3': [current_ma3]  
        })

        pred = model.predict(row[features])[0]
        forecast_results.append({'Time': loop_time, 'Predicted_MW': pred})
        temp_history.append(pred)

    forecast_df = pd.DataFrame(forecast_results)
    
    # Display results as a Chart and Table
    st.line_chart(forecast_df.set_index('Time'))
    st.dataframe(forecast_df, use_container_width=True)
