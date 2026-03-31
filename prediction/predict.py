import os
from dotenv import load_dotenv
import pandas as pd
from supabase import create_client, Client
from sklearn.linear_model import LinearRegression

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

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

    return preds.tolist()

def main():
    rows = get_sensor_data()

    if not rows:
      print("No data found.")
      return

    df = pd.DataFrame(rows)
    df["recorded_at"] = pd.to_datetime(df["recorded_at"])

    temp_preds = predict_next_values(df["temperature_c"].tolist(), 5)
    hum_preds = predict_next_values(df["humidity"].tolist(), 5)
    air_preds = predict_next_values(df["air_quality"].tolist(), 5)

    print("\nLatest readings:")
    print(df.tail(5)[["recorded_at", "temperature_c", "humidity", "air_quality"]])

    print("\nPredicted next 5 temperature values:")
    print(temp_preds)

    print("\nPredicted next 5 humidity values:")
    print(hum_preds)

    print("\nPredicted next 5 air quality values:")
    print(air_preds)

if __name__ == "__main__":
    main()