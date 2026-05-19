import argparse
import json
from pathlib import Path

import pandas as pd


def build_layer(prediction_csv):
    df = pd.read_csv(prediction_csv, parse_dates=["predicted_rain_hour"])
    required = {
        "grid_number",
        "longitude",
        "latitude",
        "predicted_rain_hour",
        "rain_probability_next_hour",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Prediction CSV is missing columns: {missing}")

    features = []
    for row in df.itertuples(index=False):
        features.append(
            {
                "grid_number": int(row.grid_number),
                "longitude": float(row.longitude),
                "latitude": float(row.latitude),
                "predicted_rain_hour": row.predicted_rain_hour.isoformat(),
                "probability": round(float(row.rain_probability_next_hour), 4),
            }
        )

    hours = [value.isoformat() for value in sorted(df["predicted_rain_hour"].unique())]
    return {
        "metadata": {
            "source_csv": str(prediction_csv),
            "aggregation": "No 3x3 aggregation; one heat point per original grid",
            "hours": hours,
        },
        "points": features,
    }


def main():
    parser = argparse.ArgumentParser(description="Build heatmap data from rain predictions.")
    parser.add_argument(
        "--prediction-csv",
        default="predictions/rain_probability_next_6h_2026-05-18.csv",
        help="Prediction CSV produced by the model.",
    )
    parser.add_argument(
        "--output-js",
        default="data/weather_heatmap_points.js",
        help="Output JavaScript file for the standalone map.",
    )
    args = parser.parse_args()

    prediction_csv = Path(args.prediction_csv)
    output_js = Path(args.output_js)
    output_js.parent.mkdir(parents=True, exist_ok=True)

    layer = build_layer(prediction_csv)
    output_js.write_text(
        "window.WEATHER_HEATMAP_POINTS = "
        + json.dumps(layer, ensure_ascii=False, indent=2)
        + ";\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(layer['points'])} heatmap points to {output_js}")
    print(f"Hours: {', '.join(layer['metadata']['hours'])}")


if __name__ == "__main__":
    main()
