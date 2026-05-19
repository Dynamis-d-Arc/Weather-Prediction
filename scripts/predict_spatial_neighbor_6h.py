import json
import subprocess
import urllib.parse
import urllib.request
from datetime import timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import pandas as pd

PSQL = r"C:\Program Files\PostgreSQL\18\bin\psql.exe"
DB = ["-h", "localhost", "-p", "5432", "-U", "postgres", "-d", "postgres"]
HOURLY = (
    "temperature_2m,rain,relative_humidity_2m,dew_point_2m,"
    "pressure_msl,surface_pressure,cloud_cover"
)
GRID_COLUMNS = 21


def psql(args):
    return subprocess.run(
        [PSQL, *DB, *args],
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
    ).stdout


def fetch_forecast():
    grid_out = psql(
        [
            "-At",
            "-F",
            ",",
            "-c",
            "SELECT grid_number, longitude, latitude FROM bangkok_grid_3km ORDER BY grid_number;",
        ]
    )
    grids = [
        (int(grid_number), float(lon), float(lat))
        for grid_number, lon, lat in (
            line.split(",") for line in grid_out.splitlines() if line.strip()
        )
    ]
    rows = []
    for offset in range(0, len(grids), 50):
        batch = grids[offset : offset + 50]
        params = {
            "latitude": ",".join(f"{lat:.8f}" for _, _, lat in batch),
            "longitude": ",".join(f"{lon:.8f}" for _, lon, _ in batch),
            "hourly": HOURLY,
            "models": "ecmwf_ifs",
            "timezone": "Asia/Bangkok",
            "forecast_days": 2,
        }
        url = "https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode(
            params, safe=","
        )
        request = urllib.request.Request(
            url, headers={"User-Agent": "codex-spatial-rain-predict/1.0"}
        )
        with urllib.request.urlopen(request, timeout=180) as response:
            payload = json.load(response)
        if isinstance(payload, dict):
            payload = [payload]

        for (grid_number, lon, lat), item in zip(batch, payload):
            hourly = item["hourly"]
            times = pd.to_datetime(hourly["time"])
            for i, weather_time in enumerate(times):
                rows.append(
                    {
                        "grid_number": grid_number,
                        "longitude": lon,
                        "latitude": lat,
                        "weather_time": weather_time.to_pydatetime(),
                        "temperature_2m": hourly["temperature_2m"][i],
                        "rain": hourly["rain"][i],
                        "relative_humidity_2m": hourly["relative_humidity_2m"][i],
                        "dew_point_2m": hourly["dew_point_2m"][i],
                        "pressure_msl": hourly["pressure_msl"][i],
                        "surface_pressure": hourly["surface_pressure"][i],
                        "cloud_cover": hourly["cloud_cover"][i],
                    }
                )
    return pd.DataFrame(rows)


def add_lag_features(df):
    df = df.sort_values(["grid_number", "weather_time"]).copy()
    grouped = df.groupby("grid_number", sort=False)
    df["hour_of_day"] = df["weather_time"].dt.hour
    df["month"] = df["weather_time"].dt.month
    df["day_of_week"] = (df["weather_time"].dt.weekday + 1) % 7
    df["is_monsoon_season"] = (
        ((df["weather_time"].dt.month == 5) & (df["weather_time"].dt.day >= 15))
        | df["weather_time"].dt.month.isin([6, 7, 8, 9])
        | ((df["weather_time"].dt.month == 10) & (df["weather_time"].dt.day <= 15))
    ).astype(int)

    for hours in [1, 3, 6]:
        suffix = f"{hours}h_ago"
        df[f"temperature_2m_{suffix}"] = grouped["temperature_2m"].shift(hours)
        df[f"humidity_{suffix}"] = grouped["relative_humidity_2m"].shift(hours)
        df[f"dew_point_{suffix}"] = grouped["dew_point_2m"].shift(hours)
        df[f"pressure_msl_{suffix}"] = grouped["pressure_msl"].shift(hours)
        df[f"surface_pressure_{suffix}"] = grouped["surface_pressure"].shift(hours)
        df[f"cloud_cover_{suffix}"] = grouped["cloud_cover"].shift(hours)

    df["rain_1h_ago"] = grouped["rain"].shift(1)
    df["rain_last_3h"] = (
        grouped["rain"].rolling(3, min_periods=1).sum().shift(1).reset_index(level=0, drop=True)
    )
    df["rain_last_6h"] = (
        grouped["rain"].rolling(6, min_periods=1).sum().shift(1).reset_index(level=0, drop=True)
    )
    df["pressure_change_1h"] = df["pressure_msl"] - df["pressure_msl_1h_ago"]
    df["pressure_change_3h"] = df["pressure_msl"] - df["pressure_msl_3h_ago"]
    df["pressure_change_6h"] = df["pressure_msl"] - df["pressure_msl_6h_ago"]
    df["surface_pressure_change_1h"] = df["surface_pressure"] - df["surface_pressure_1h_ago"]
    df["surface_pressure_change_3h"] = df["surface_pressure"] - df["surface_pressure_3h_ago"]
    df["surface_pressure_change_6h"] = df["surface_pressure"] - df["surface_pressure_6h_ago"]
    return df


def neighbor_pairs(grid_numbers):
    grid_set = set(grid_numbers)
    pairs = []
    for grid_number in grid_numbers:
        row = (grid_number - 1) // GRID_COLUMNS
        col = (grid_number - 1) % GRID_COLUMNS
        for drow in [-1, 0, 1]:
            for dcol in [-1, 0, 1]:
                if drow == 0 and dcol == 0:
                    continue
                neighbor = (row + drow) * GRID_COLUMNS + (col + dcol) + 1
                if neighbor in grid_set:
                    pairs.append((grid_number, neighbor))
    return pd.DataFrame(pairs, columns=["grid_number", "neighbor_grid_number"])


def add_spatial_neighbor_features(df):
    pairs = neighbor_pairs(sorted(df["grid_number"].unique()))
    neighbor_source = df[
        [
            "grid_number",
            "weather_time",
            "rain_1h_ago",
            "rain_last_3h",
            "humidity_1h_ago",
            "cloud_cover_1h_ago",
            "pressure_change_3h",
        ]
    ].rename(columns={"grid_number": "neighbor_grid_number"})
    joined = pairs.merge(neighbor_source, on="neighbor_grid_number", how="left")
    agg = (
        joined.groupby(["grid_number", "weather_time"], as_index=False)
        .agg(
            neighbor_count=("neighbor_grid_number", "count"),
            neighbor_rain_1h_ago_avg=("rain_1h_ago", "mean"),
            neighbor_rain_last_3h_avg=("rain_last_3h", "mean"),
            neighbor_humidity_1h_ago_avg=("humidity_1h_ago", "mean"),
            neighbor_cloud_cover_1h_ago_avg=("cloud_cover_1h_ago", "mean"),
            neighbor_pressure_change_3h_avg=("pressure_change_3h", "mean"),
        )
    )
    return df.merge(agg, on=["grid_number", "weather_time"], how="left")


def main():
    artifact = joblib.load("models/rainfall_probability_spatial_neighbor_model.joblib")
    model = artifact["model"]
    feature_cols = artifact["feature_cols"]

    forecast = add_spatial_neighbor_features(add_lag_features(fetch_forecast()))
    now = pd.Timestamp.now(tz=ZoneInfo("Asia/Bangkok")).floor("h").tz_localize(None)
    prediction_input_hours = [now + timedelta(hours=i) for i in range(6)]
    predictions = forecast[forecast["weather_time"].isin(prediction_input_hours)].copy()
    predictions["predicted_rain_hour"] = predictions["weather_time"] + pd.Timedelta(hours=1)
    predictions["rain_probability_next_hour"] = model.predict_proba(
        predictions[feature_cols]
    )[:, 1]

    Path("predictions").mkdir(exist_ok=True)
    output_path = Path("predictions") / "rain_probability_spatial_next_6h.csv"
    predictions.to_csv(output_path, index=False)

    hourly = (
        predictions.groupby("predicted_rain_hour")["rain_probability_next_hour"]
        .agg(["mean", "max"])
        .reset_index()
    )
    top = predictions.sort_values("rain_probability_next_hour", ascending=False).head(15)
    print("SPATIAL_NEXT_6_HOURS_SUMMARY")
    print(f"rows={len(predictions):,}")
    print(f"overall_mean={predictions['rain_probability_next_hour'].mean():.4f}")
    print(f"overall_max={predictions['rain_probability_next_hour'].max():.4f}")
    print(f"grid_hours_over_50pct={(predictions['rain_probability_next_hour'] >= 0.5).sum()}")
    print(f"csv={output_path.resolve()}")
    print("HOURLY_SUMMARY")
    print(hourly.to_string(index=False))
    print("TOP_15")
    print(
        top[
            [
                "predicted_rain_hour",
                "grid_number",
                "longitude",
                "latitude",
                "rain_probability_next_hour",
                "neighbor_rain_1h_ago_avg",
                "neighbor_rain_last_3h_avg",
                "neighbor_cloud_cover_1h_ago_avg",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
