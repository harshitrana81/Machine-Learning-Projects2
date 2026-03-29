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
   # --- 4. 24-HOUR FORECAST VISUALIZATION ---
    st.divider()
    st.subheader("📊 24-Hour Energy Demand Forecast")
    
    forecast_results = []
    temp_history = history.copy()
    temp_history.append(current_prediction)
    loop_time = user_datetime

    for i in range(24):
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
            'Time': loop_time, 
            'Demand_MW': pred
        })
        temp_history.append(pred)

    forecast_df = pd.DataFrame(forecast_results)

    # Better Visualization: Area Chart for "Power" look
    st.area_chart(forecast_df.set_index('Time'), color="#FF4B4B")

    # Adding "Insight" Metrics
    col1, col2 = st.columns(2)
    with col1:
        peak_val = forecast_df['Demand_MW'].max()
        peak_time = forecast_df.loc[forecast_df['Demand_MW'].idxmax(), 'Time'].strftime('%I:%M %p')
        st.metric("Peak Forecasted Demand", f"{peak_val:.2f} MW", help="Highest expected load in next 24h")
    
    with col2:
        st.write(f"**Peak expected at:** {peak_time}")
        st.write("**Trend:** " + ("📈 Increasing" if forecast_df['Demand_MW'].iloc[-1] > current_prediction else "📉 Decreasing"))

    # Expandable Table for raw data
    with st.expander("View Detailed Hourly Forecast Data"):
        # Format time for display in the table
        display_df = forecast_df.copy()
        display_df['Time'] = display_df['Time'].dt.strftime('%Y-%m-%d %I:%M %p')
        st.dataframe(display_df, use_container_width=True)
