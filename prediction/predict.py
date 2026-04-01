import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from supabase import create_client, Client
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings
warnings.filterwarnings('ignore')

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in .env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def get_sensor_data(limit=500):
    response = (
        supabase.table("sensor_readings")
        .select("*")
        .order("recorded_at", desc=False)
        .limit(limit)
        .execute()
    )
    return response.data


def build_time_indexed_frame(rows):
    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df["recorded_at"] = pd.to_datetime(df["recorded_at"])
    df = df.sort_values("recorded_at").reset_index(drop=True)
    return df


def predict_next_values(series, steps=5):
    if len(series) < 5:
        return None

    df = pd.DataFrame({"y": series})
    df["x"] = range(len(df))

    X = df[["x"]]
    y = df["y"]

    model = LinearRegression()
    model.fit(X, y)

    future_x = pd.DataFrame({"x": range(len(df), len(df) + steps)})
    preds = model.predict(future_x)

    return [round(float(v), 2) for v in preds.tolist()]


def predict_moving_average(series, steps=5, window=3):
    """Simple Moving Average prediction"""
    if len(series) < window:
        return None

    # Calculate moving average
    ma = pd.Series(series).rolling(window=window).mean()

    # Use the last valid moving average value for prediction
    last_ma = ma.dropna().iloc[-1]

    # For simplicity, assume constant trend
    preds = [round(float(last_ma), 2) for _ in range(steps)]

    return preds


def predict_arima(series, steps=5):
    """ARIMA prediction"""
    if len(series) < 10:  # Need more data for ARIMA
        return None

    try:
        # Fit ARIMA model (p=1, d=1, q=1) - can be tuned
        model = ARIMA(series, order=(1, 1, 1))
        model_fit = model.fit()

        # Make predictions
        forecast = model_fit.forecast(steps=steps)

        return [round(float(v), 2) for v in forecast]
    except:
        return None


def predict_lstm(series, steps=5, epochs=10, batch_size=32):
    """LSTM Neural Network prediction"""
    if len(series) < 15:  # Reduced requirement
        return None

    try:
        # Prepare data
        data = np.array(series).reshape(-1, 1)

        # Scale data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        # Create sequences
        sequence_length = min(5, len(scaled_data) - 1)  # Shorter sequences
        X, y = [], []

        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i, 0])
            y.append(scaled_data[i, 0])

        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        # Simpler LSTM model
        model = Sequential([
            LSTM(32, input_shape=(X.shape[1], 1)),  # Simpler model
            Dense(16),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train model
        model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

        # Make predictions
        predictions = []
        current_sequence = scaled_data[-sequence_length:].reshape(1, sequence_length, 1)

        for _ in range(steps):
            pred = model.predict(current_sequence, verbose=0)
            predictions.append(pred[0][0])

            # Update sequence for next prediction
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, 0] = pred[0][0]

        # Inverse transform predictions
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        predictions = predictions.flatten()

        return [round(float(v), 2) for v in predictions]

    except Exception as e:
        print(f"LSTM prediction failed: {e}")
        return None


def predict_exponential_smoothing(series, steps=5):
    """Exponential Smoothing prediction"""
    if len(series) < 5:
        return None

    try:
        # Fit Exponential Smoothing model
        model = ExponentialSmoothing(series, seasonal=None, trend='add')
        model_fit = model.fit()

        # Make predictions
        forecast = model_fit.forecast(steps)

        return [round(float(v), 2) for v in forecast]
    except:
        return None


def estimate_future_timestamps(df, steps=5):
    if len(df) < 2:
        return []

    diffs = df["recorded_at"].diff().dropna()

    if diffs.empty:
        return []

    avg_diff = diffs.mean()
    last_time = df["recorded_at"].iloc[-1]

    future_times = []
    for i in range(1, steps + 1):
        future_times.append(last_time + i * avg_diff)

    return future_times


def main():
    print("Fetching sensor data...")
    rows = get_sensor_data(limit=50)  # Reduced limit for faster testing

    if not rows:
        print("No data found in sensor_readings yet.")
        return

    print(f"Found {len(rows)} readings")
    df = build_time_indexed_frame(rows)

    if df.empty:
        print("No usable data found.")
        return

    print("\nLatest 10 readings:")
    print(df.tail(10)[["recorded_at", "device_id", "temperature_c", "humidity", "air_quality"]])

    steps = 5
    future_times = estimate_future_timestamps(df, steps)

    if not future_times:
        print("\nNot enough data yet to make predictions.")
        print("Collect at least 5 readings first.")
        return

    # Get predictions from all models
    models = {
        "Linear Regression": predict_next_values,
        "Moving Average": predict_moving_average,
        "Exponential Smoothing": predict_exponential_smoothing,
        "ARIMA": predict_arima,
        "LSTM": predict_lstm
    }

    predictions = {}

    for model_name, predict_func in models.items():
        print(f"\n--- Running {model_name} ---")

        temp_preds = predict_func(df["temperature_c"].tolist(), steps)
        hum_preds = predict_func(df["humidity"].tolist(), steps)
        air_preds = predict_func(df["air_quality"].tolist(), steps)

        if temp_preds and hum_preds and air_preds:
            prediction_df = pd.DataFrame({
                "predicted_time": future_times,
                "pred_temperature_c": temp_preds,
                "pred_humidity": hum_preds,
                "pred_air_quality": air_preds
            })

            print(f"Next {steps} {model_name.lower()} predictions:")
            print(prediction_df)

            predictions[model_name] = {
                "temperature": temp_preds,
                "humidity": hum_preds,
                "air_quality": air_preds,
                "timestamps": future_times
            }
        else:
            print(f"Not enough data for {model_name} predictions (need more readings)")

    # Save predictions to JSON for frontend integration
    if predictions:
        import json
        with open('predictions.json', 'w') as f:
            # Convert timestamps to strings for JSON serialization
            json_predictions = {}
            for model, preds in predictions.items():
                json_predictions[model] = {
                    "temperature": preds["temperature"],
                    "humidity": preds["humidity"],
                    "air_quality": preds["air_quality"],
                    "timestamps": [str(t) for t in preds["timestamps"]]
                }
            json.dump(json_predictions, f, indent=2)
        print("\nPredictions saved to predictions.json for frontend integration")


if __name__ == "__main__":
    main()