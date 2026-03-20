import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import datetime

# --- Page Config ---
st.set_page_config(page_title="Electricity Consumption Predictor", layout="wide")

st.title("⚡ Electricity Consumption Forecast (DUQ MW)")
st.write("This app predicts hourly electricity consumption using a Random Forest Regressor.")

# --- 1. Load Data ---
@st.cache_data
def load_data():
    # Make sure DUQ_hourly.csv is in your GitHub repo!
    df = pd.read_csv("DUQ_hourly.csv", parse_dates=['Datetime'])
    df.set_index('Datetime', inplace=True)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    
    # Feature Engineering
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['lag_1'] = df['DUQ_MW'].shift(1)
    df.dropna(inplace=True)
    return df

try:
    df = load_data()
    
    # --- 2. Train Model ---
    X = df[['hour', 'day_of_week', 'month', 'lag_1']]
    y = df['DUQ_MW']
    
    train_size = int(len(df) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # --- 3. Sidebar Inputs ---
    st.sidebar.header("User Input Parameters")
    input_date = st.sidebar.date_input("Select Date", datetime.date(2026, 3, 21))
    input_time = st.sidebar.time_input("Select Time", datetime.time(12, 0))
    
    user_datetime = pd.to_datetime(f"{input_date} {input_time}")
    last_actual_value = df.iloc[-1]['DUQ_MW']

    # --- 4. Prediction Logic ---
    if st.button("Predict Consumption"):
        # Single Prediction
        input_row = pd.DataFrame({
            'hour': [user_datetime.hour],
            'day_of_week': [user_datetime.dayofweek],
            'month': [user_datetime.month],
            'lag_1': [last_actual_value]
        })
        
        prediction = model.predict(input_row)[0]
        st.metric(label=f"Predicted Load for {user_datetime}", value=f"{prediction:.2f} MW")

        # --- 5. 24-Hour Forecast ---
        st.subheader("📅 24-Hour Future Forecast")
        forecast_results = []
        last_val = prediction
        loop_time = user_datetime

        for _ in range(24):
            loop_time += pd.Timedelta(hours=1)
            row = pd.DataFrame({
                'hour': [loop_time.hour],
                'day_of_week': [loop_time.dayofweek],
                'month': [loop_time.month],
                'lag_1': [last_val]
            })
            pred = model.predict(row)[0]
            forecast_results.append({'Time': loop_time, 'Predicted_MW': pred})
            last_val = pred
        
        forecast_df = pd.DataFrame(forecast_results)
        
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(forecast_df)
        with col2:
            fig, ax = plt.subplots()
            ax.plot(forecast_df['Time'], forecast_df['Predicted_MW'], marker='o', color='orange')
            plt.xticks(rotation=45)
            ax.set_ylabel("MW")
            st.pyplot(fig)

except FileNotFoundError:
    st.error("Error: 'DUQ_hourly.csv' not found. Please upload it to your GitHub repository.")
