import json
import os
import subprocess
import time
from pathlib import Path

import joblib
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline

DB_PASSWORD = os.getenv("PGPASSWORD")
if not DB_PASSWORD:
    raise RuntimeError("Set the PGPASSWORD environment variable before training.")
SAMPLE_FRACTION = 0.50
MODEL_PATH = Path("models/rainfall_probability_spatial_neighbor_model.joblib")

# The first part is the same as the lag model. The final block adds spatial
# neighbor features from adjacent 3km grid cells.
FEATURE_COLS = [
    "grid_number",
    "longitude",
    "latitude",
    "hour_of_day",
    "month",
    "day_of_week",
    "is_monsoon_season",
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "pressure_msl",
    "surface_pressure",
    "cloud_cover",
    "temperature_2m_1h_ago",
    "humidity_1h_ago",
    "dew_point_1h_ago",
    "pressure_msl_1h_ago",
    "surface_pressure_1h_ago",
    "cloud_cover_1h_ago",
    "rain_1h_ago",
    "temperature_2m_3h_ago",
    "humidity_3h_ago",
    "dew_point_3h_ago",
    "pressure_msl_3h_ago",
    "surface_pressure_3h_ago",
    "cloud_cover_3h_ago",
    "rain_last_3h",
    "temperature_2m_6h_ago",
    "humidity_6h_ago",
    "dew_point_6h_ago",
    "pressure_msl_6h_ago",
    "surface_pressure_6h_ago",
    "cloud_cover_6h_ago",
    "rain_last_6h",
    "pressure_change_1h",
    "pressure_change_3h",
    "pressure_change_6h",
    "surface_pressure_change_1h",
    "surface_pressure_change_3h",
    "surface_pressure_change_6h",
    "neighbor_count",
    "neighbor_rain_1h_ago_avg",
    "neighbor_rain_last_3h_avg",
    "neighbor_humidity_1h_ago_avg",
    "neighbor_cloud_cover_1h_ago_avg",
    "neighbor_pressure_change_3h_avg",
]

PSQL = r"C:\Program Files\PostgreSQL\18\bin\psql.exe"
DB_ARGS = ["-h", "localhost", "-p", "5432", "-U", "postgres", "-d", "postgres"]


def psql(args):
    return subprocess.run(
        [PSQL, *DB_ARGS, *args],
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
    ).stdout


def ensure_spatial_neighbor_table():
    print("Creating weather_features_spatial_neighbor table...", flush=True)
    sql = """
    DROP TABLE IF EXISTS weather_features_spatial_neighbor;

    CREATE TABLE weather_features_spatial_neighbor AS
    WITH grid_positions AS (
        SELECT
            grid_number,
            ((grid_number - 1) / 21)::int AS grid_row,
            ((grid_number - 1) % 21)::int AS grid_col
        FROM bangkok_grid_3km
    ),
    neighbor_pairs AS (
        SELECT
            center.grid_number AS grid_number,
            neighbor.grid_number AS neighbor_grid_number
        FROM grid_positions center
        JOIN grid_positions neighbor
          ON ABS(center.grid_row - neighbor.grid_row) <= 1
         AND ABS(center.grid_col - neighbor.grid_col) <= 1
         AND center.grid_number <> neighbor.grid_number
    ),
    neighbor_features AS (
        SELECT
            wf.grid_number,
            wf.weather_time,
            COUNT(nf.grid_number) AS neighbor_count,
            AVG(nf.rain_1h_ago) AS neighbor_rain_1h_ago_avg,
            AVG(nf.rain_last_3h) AS neighbor_rain_last_3h_avg,
            AVG(nf.humidity_1h_ago) AS neighbor_humidity_1h_ago_avg,
            AVG(nf.cloud_cover_1h_ago) AS neighbor_cloud_cover_1h_ago_avg,
            AVG(nf.pressure_change_3h) AS neighbor_pressure_change_3h_avg
        FROM weather_features wf
        JOIN neighbor_pairs np
          ON np.grid_number = wf.grid_number
        JOIN weather_features nf
          ON nf.grid_number = np.neighbor_grid_number
         AND nf.weather_time = wf.weather_time
        GROUP BY wf.grid_number, wf.weather_time
    )
    SELECT
        wf.*,
        EXTRACT(HOUR FROM wf.weather_time)::int AS hour_of_day,
        EXTRACT(MONTH FROM wf.weather_time)::int AS month,
        EXTRACT(DOW FROM wf.weather_time)::int AS day_of_week,
        nf.neighbor_count,
        nf.neighbor_rain_1h_ago_avg,
        nf.neighbor_rain_last_3h_avg,
        nf.neighbor_humidity_1h_ago_avg,
        nf.neighbor_cloud_cover_1h_ago_avg,
        nf.neighbor_pressure_change_3h_avg
    FROM weather_features wf
    LEFT JOIN neighbor_features nf
      ON nf.grid_number = wf.grid_number
     AND nf.weather_time = wf.weather_time;

    ALTER TABLE weather_features_spatial_neighbor
      ADD PRIMARY KEY (grid_number, weather_time);

    CREATE INDEX weather_features_spatial_neighbor_time_idx
      ON weather_features_spatial_neighbor (weather_time);

    CREATE INDEX weather_features_spatial_neighbor_target_idx
      ON weather_features_spatial_neighbor (will_rain_next_hour);

    ANALYZE weather_features_spatial_neighbor;

    SELECT COUNT(*) AS rows,
           COUNT(DISTINCT grid_number) AS grids,
           MIN(weather_time) AS min_time,
           MAX(weather_time) AS max_time
    FROM weather_features_spatial_neighbor;
    """
    print(psql(["-v", "ON_ERROR_STOP=1", "-c", sql]), flush=True)


def train_model():
    engine = create_engine(
        f"postgresql+psycopg2://postgres:{DB_PASSWORD}@localhost:5432/postgres"
    )
    query = text(
        """
        SELECT *
        FROM weather_features_spatial_neighbor
        WHERE random() < :sample_fraction
          AND will_rain_next_hour IS NOT NULL
        ORDER BY weather_time;
        """
    )
    started = time.time()
    print("Loading 50% sample from weather_features_spatial_neighbor...", flush=True)
    df = pd.read_sql(query, engine, params={"sample_fraction": SAMPLE_FRACTION})
    print(f"Loaded {len(df):,} rows in {(time.time() - started) / 60:.1f} min", flush=True)

    split_time = df["weather_time"].quantile(0.80)
    train_df = df[df["weather_time"] <= split_time].copy()
    test_df = df[df["weather_time"] > split_time].copy()

    x_train = train_df[FEATURE_COLS]
    y_train = train_df["will_rain_next_hour"]
    x_test = test_df[FEATURE_COLS]
    y_test = test_df["will_rain_next_hour"]

    clf = Pipeline(
        [
            (
                "preprocess",
                ColumnTransformer(
                    [("numeric", SimpleImputer(strategy="median"), FEATURE_COLS)]
                ),
            ),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.06,
                    max_iter=180,
                    max_leaf_nodes=31,
                    l2_regularization=0.1,
                    random_state=42,
                    verbose=1,
                ),
            ),
        ]
    )

    print(f"Training rows={len(x_train):,}; test rows={len(x_test):,}", flush=True)
    started = time.time()
    clf.fit(x_train, y_train)
    print(f"Training completed in {(time.time() - started) / 60:.1f} min", flush=True)

    probabilities = clf.predict_proba(x_test)[:, 1]
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "brier_score": float(brier_score_loss(y_test, probabilities)),
        "sample_fraction": SAMPLE_FRACTION,
        "sample_rows": int(len(df)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "test_positive_rate": float(y_test.mean()),
        "features": FEATURE_COLS,
        "spatial_neighbor_definition": "8 surrounding grid cells from 3x3 window around each center grid",
    }

    Path("models").mkdir(exist_ok=True)
    joblib.dump(
        {
            "model": clf,
            "feature_cols": FEATURE_COLS,
            "target_col": "will_rain_next_hour",
            "metrics": metrics,
            "training_strategy": "50pct_sample_with_spatial_neighbor_features",
        },
        MODEL_PATH,
    )
    print(f"Saved model: {MODEL_PATH.resolve()}", flush=True)
    print(json.dumps(metrics, indent=2), flush=True)


def main():
    ensure_spatial_neighbor_table()
    train_model()


if __name__ == "__main__":
    main()
