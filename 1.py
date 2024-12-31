import numpy as np
import pandas as pd
import mindspore as ms
from mindspore import nn, Tensor
from mindspore.train.callback import LossMonitor
import matplotlib.pyplot as plt
import os
from datetime import datetime
from holidays import UnitedStates
import requests
# Load dataset
def load_data(file_path):
    try:
        data = pd.read_parquet(file_path)
        print("Dataset loaded successfully.")
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

# Preprocess dataset
def preprocess_data(data):
    try:
        # Convert datetime columns
        data['pickup_hour'] = pd.to_datetime(data['tpep_pickup_datetime']).dt.floor('H')
        data['dropoff_hour'] = pd.to_datetime(data['tpep_dropoff_datetime']).dt.floor('H')

        # Aggregate by pickup_hour and PULocationID
        hourly_location_counts = data.groupby(['pickup_hour', 'PULocationID'])['total_amount'].sum().reset_index(name='total_amount')

        # Prepare features (X) and target labels (y)
        X = hourly_location_counts[['pickup_hour', 'PULocationID']]
        y = hourly_location_counts['total_amount']

        # Add additional features (e.g., holiday)
        X['is_holiday'] = X['pickup_hour'].apply(is_holiday)

        # Temporarily disable weather feature
        X['weather'] = X['pickup_hour'].apply(get_weather)  # Commented out weather feature

        # Encode categorical variables (e.g., PULocationID)
        X = pd.get_dummies(X, columns=['PULocationID'], drop_first=True)

        # No normalization for 'is_holiday' since it's binary (0 or 1)
        # Normalize numerical features only if needed (but avoid it for 'is_holiday')
        # numerical_columns = ['is_holiday']  # Only normalizing 'is_holiday' for now
        # for col in numerical_columns:
        #     X[col] = (X[col] - X[col].mean()) / X[col].std()

        # Convert to NumPy arrays
        X = X.to_numpy()
        y = y.to_numpy()

        # Reshape for RNN input: (batch_size, seq_len, feature_size)
        X = X[:, np.newaxis, :]  # Adding sequence length dimension
        y = y[:, np.newaxis]     # Shape (batch_size, 1)

        print(f"Preprocessed data shapes - X: {X.shape}, y: {y.shape}")
        return X, y
    except Exception as e:
        print(f"Error in data preprocessing: {e}")
        return None, None

def is_holiday(timestamp):
    # Use the holidays package to check if a date is a holiday in the US
    us_holidays = UnitedStates(years=timestamp.year)
    return 1 if timestamp.date() in us_holidays else 0

def get_weather(timestamp):
    # Fetch weather data for New York City (example implementation)
    api_key = '7df065055ea73f1871a886451a26b60b'  # Replace with your actual API key
    url = f"https://api.openweathermap.org/data/2.5/weather?q=New%20York&appid={api_key}&units=metric"
    try:
        response = requests.get(url)
        data = response.json()
        if 'weather' in data:
            main_weather = data['weather'][0]['main'].lower()
            if 'rain' in main_weather:
                return 1  # Rainy
            elif 'snow' in main_weather:
                return 2  # Snowy
            else:
                return 0  # Clear
        return 0
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        return 0

# Main function
def main():
    file_path = './yellow_tripdata_2024-07.parquet'
    data = load_data(file_path)
    if data is None:
        return

    X, y = preprocess_data(data)
    if X is None or y is None or len(X) == 0:
        print("Data preprocessing failed or no valid data available.")
        return
    print("Process succeed")
    # Split dataset
    split_idx = int(0.8 * len(X))
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]
    
     # Save datasets
    save_path = './saved_data'
    os.makedirs(save_path, exist_ok=True)

 # Save as CSV
    # Flatten the data before saving
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    y_train_flat = y_train.reshape(y_train.shape[0], -1)
    y_test_flat = y_test.reshape(y_test.shape[0], -1)
    pd.DataFrame(X_train_flat).to_csv(os.path.join(save_path, 'X_train.csv'), index=False)
    pd.DataFrame(y_train_flat).to_csv(os.path.join(save_path, 'y_train.csv'), index=False)
    pd.DataFrame(X_test_flat).to_csv(os.path.join(save_path, 'X_test.csv'), index=False)
    pd.DataFrame(y_test_flat).to_csv(os.path.join(save_path, 'y_test.csv'), index=False)

    print(f"Data saved successfully in directory: {save_path}")
if __name__ == "__main__":
    main()
