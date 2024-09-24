import numpy as np
import tensorflow as tf
from tensorflow import keras
import psycopg2
import time

# Load the trained LSTM models
model_temp = keras.models.load_model('model_suhu.h5')
model_hum = keras.models.load_model('model_hum.h5')
window = 10

# Load the MinMaxScaler object
# scaler_temp = load('standard_temp.joblib')
# scaler_hum = load('standard_hum.joblib')

# Initialize the data buffer to store the last 10 data points
data_buffer = []
last_fetched_data = None

def fetch_data_from_db():
    # Connect to your postgres DB
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="1",
        host="localhost",
        port = "5432"
    )
    cursor = conn.cursor()
    
    # Execute a query to fetch the latest data points
    cursor.execute("SELECT temp, hum FROM datadht_client1")
    
    # Retrieve the results
    results = cursor.fetchall()
    
    cursor.close()
    conn.close()
    
    return results  # reverse to get the correct order

def send_prediction_to_db(pred_temp, pred_hum):
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="1",
        host="localhost",
        port = "5432"
    )
    cursor = conn.cursor()
    
    # Insert the prediction data into a different table
    query = "INSERT INTO datadhtpred_client1 (pred_suhu, pred_kelembapan) VALUES (%s, %s)"
    cursor.execute(query, (float(pred_temp), float(pred_hum)))
    
    conn.commit()
    cursor.close()
    conn.close()

while True:
    # Fetch data from the database
    db_data = fetch_data_from_db()
    
    if db_data != last_fetched_data:
        last_fetched_data = db_data
        data_buffer.extend(db_data)
        
        # Ensure we only keep the last 10 data points
        if len(data_buffer) > window:
            data_buffer = data_buffer[-window:]
        
        # If the buffer has at least 10 data points, make a prediction
        if len(data_buffer) >= window:
            # Scale the data using the MinMaxScaler
            # scaled_temp_data = scaler_temp.transform(np.array(data_buffer[-window:])[:, 0].reshape(-1, 1))
            # scaled_hum_data = scaler_hum.transform(np.array(data_buffer[-window:])[:, 1].reshape(-1, 1))
            
            temp_data = np.array(data_buffer[-window:])[:, 0].reshape(-1, 1)
            hum_data = np.array(data_buffer[-window:])[:, 1].reshape(-1, 1)
            
            temp_data = temp_data.reshape((1, window, 1))  # reshape for LSTM input
            hum_data = hum_data.reshape((1, window, 1))  # reshape for LSTM input

            # Make a prediction
            pred_temp = model_temp.predict(temp_data)
            pred_hum = model_hum.predict(hum_data)
            print(f"Prediction: {pred_temp[0][0]:.2f}°C, {pred_hum[0][0]:.2f}%")

            send_prediction_to_db(pred_temp[0][0], pred_hum[0][0])
    else:
        print("No new data available, waiting for updates...")

    # Wait for 2 seconds before fetching the next set of data points
    time.sleep(1)
