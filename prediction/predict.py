import os
from dotenv import load_dotenv
import pandas as pd
from supabase import create_client, Client
from sklearn.linear_model import LinearRegression

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

    # Get predictions only from linear regression
    model_name = "Linear Regression"
    predictions = {}

    print(f"\n--- Running {model_name} ---")

    temp_preds = predict_next_values(df["temperature_c"].tolist(), steps)
    hum_preds = predict_next_values(df["humidity"].tolist(), steps)
    air_preds = predict_next_values(df["air_quality"].tolist(), steps)

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