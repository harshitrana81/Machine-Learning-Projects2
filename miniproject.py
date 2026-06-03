import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# Use wide layout to make side-by-side visualization look professional
st.set_page_config(page_title="Electricity Forecast", page_icon="⚡", layout="wide")

# --- 1. LOAD THE MODEL (Optimized) ---
@st.cache_resource
def load_model():
    return joblib.load('electricity_rf_model.joblib')

model = load_model()

# --- 2. HEADER ---
st.title("⚡ Smart Grid: Electricity Demand Predictor")
st.write("Predicting real-time energy consumption using Random Forest.")
st.divider()

# --- 3. LAYOUT: SIDE-BY-SIDE ---
# col1 is for inputs/metric (left), col2 is for the graph/table (right)
col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.header("User Input")
    input_date = st.date_input("Select Date", datetime.now())
    input_time = st.time_input("Select Time", datetime.now())
    user_datetime = datetime.combine(input_date, input_time)

    # Prediction Logic Data
    history = [14500.0, 15200.0, 14800.0] 
    initial_ma3 = sum(history) / 3
    features = ['hour', 'day_of_week', 'month', 'lag_1', 'ma_3']

    if st.button("Generate Forecast", use_container_width=True):
        # 1-Hour Prediction
        first_row = pd.DataFrame({
            'hour': [user_datetime.hour],
            'day_of_week': [user_datetime.weekday()],
            'month': [user_datetime.month],
            'lag_1': [history[-1]], 
            'ma_3': [initial_ma3]   
        })
        current_prediction = model.predict(first_row[features])[0]
        
        # Display Metric on the left
        display_time = user_datetime.strftime('%I:%M %p') 
        st.metric(label=f"Predicted Demand ({display_time})", 
                  value=f"{current_prediction:.2f} MW")
        
        # --- 4. 24-HOUR FORECAST CALCULATION ---
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
            forecast_results.append({'Time': loop_time, 'Demand_MW': pred})
            temp_history.append(pred)

        # Store in session state to show in the right column
        st.session_state['forecast_df'] = pd.DataFrame(forecast_results)
        st.session_state['current_prediction'] = current_prediction

with col2:
    st.header("📊 Forecast Visualization")
    
    if 'forecast_df' in st.session_state:
        df = st.session_state['forecast_df']
        curr_pred = st.session_state['current_prediction']

        # A. Curved Line Plot (Right Side)
        st.line_chart(df.set_index('Time'), color="#29b5e8")

        # B. Insights underneath the plot
        peak_val = df['Demand_MW'].max()
        peak_time = df.loc[df['Demand_MW'].idxmax(), 'Time'].strftime('%I:%M %p')
        
        st.info(f"💡 **Peak Forecast:** {peak_val:.2f} MW at **{peak_time}**")
        
        # C. Expandable Data Table (Right Side)
        with st.expander("View Hourly Data Table"):
            display_df = df.copy()
            display_df['Time'] = display_df['Time'].dt.strftime('%Y-%m-%d %I:%M %p')
            st.dataframe(display_df, use_container_width=True)
    else:
        st.info("Please click 'Generate Forecast' on the left to view the visualization.")
