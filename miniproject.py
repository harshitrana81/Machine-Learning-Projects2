import streamlit as st
import pandas as pd
import joblib
import datetime

# --- SET UP ---
st.set_page_config(page_title="Electricity Forecast AI", layout="centered")
st.title("⚡ Electricity Consumption Predictor")
st.write("Predicting real-time energy demand using Random Forest.")

# --- LOAD MODEL ---
@st.cache_resource # This ensures the model only loads ONCE for speed
def load_model():
    return joblib.load('electricity_rf_model.pkl')

model = load_model()

# --- INPUT SECTION ---
st.subheader("1. Enter Current Data")
col1, col2 = st.columns(2)

with col1:
    input_date = st.date_input("Select Date", datetime.date.today())
    input_time = st.time_input("Select Time", datetime.time(12, 0))
    target_dt = pd.to_datetime(f"{input_date} {input_time}")

with col2:
    # In a real app, 'last_val' would come from your hardware/API
    last_val = st.number_input("Last Hour Consumption (MW)", value=15000.0)
    # For MA_3, we'll simulate the history or let user input it
    ma3_val = st.number_input("3-Hour Moving Average (MW)", value=14800.0)

# --- PREDICTION ---
if st.button("Predict & Forecast 24 Hours"):
    # Create features for the first prediction
    first_row = pd.DataFrame({
        'hour': [target_dt.hour],
        'day_of_week': [target_dt.dayofweek],
        'month': [target_dt.month],
        'lag_1': [last_val],
        'ma_3': [ma3_val]
    })
    
    current_pred = model.predict(first_row)[0]
    st.success(f"Immediate Prediction for {target_dt.strftime('%H:%M')}: **{current_pred:.2f} MW**")
    
    # --- 24 HOUR FORECAST LOOP ---
    history = [ma3_val, last_val, current_pred] # Sliding window
    forecast_results = []
    loop_time = target_dt
    
    for i in range(24):
        loop_time += pd.Timedelta(hours=1)
        current_ma = sum(history[-3:]) / 3
        
        row = pd.DataFrame({
            'hour': [loop_time.hour],
            'day_of_week': [loop_time.dayofweek],
            'month': [loop_time.month],
            'lag_1': [history[-1]],
            'ma_3': [current_ma]
        })
        
        pred = model.predict(row)[0]
        forecast_results.append({"Time": loop_time, "Predicted MW": pred})
        history.append(pred)
        
    # Display Results
    forecast_df = pd.DataFrame(forecast_results)
    st.subheader("📅 24-Hour Forecast Chart")
    st.line_chart(forecast_df.set_index('Time'))
    st.write(forecast_df)
