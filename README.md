# Bangkok Rainfall Probability Model

This project predicts next-hour rainfall probability over a Bangkok grid and visualizes the result as an OpenStreetMap heatmap.

The current workflow uses:

```text
Open-Meteo API
PostgreSQL
weather_features
weather_features_spatial_neighbor
HistGradientBoostingClassifier
Leaflet / OpenStreetMap heatmap page
```

## Current Model

The main model is the spatial-neighbor model:

```text
models/rainfall_probability_spatial_neighbor_model.joblib
```

Training script:

```text
scripts/train_spatial_neighbor_model.py
```

Prediction script:

```text
scripts/predict_spatial_neighbor_6h.py
```

The model uses a 50% random sample of `weather_features_spatial_neighbor`.

Latest run:

```text
Rows sampled: 3,489,565
ROC AUC: 0.871
Average precision: 0.321
Brier score: 0.0408
```

## Why This Model

The model uses `HistGradientBoostingClassifier` from scikit-learn.

It was chosen because it works well for tabular weather data:

```text
temperature
humidity
dew point
pressure
cloud cover
rain history
location
time
season
neighbor-grid features
```

It can learn non-linear relationships, handles large numeric datasets reasonably well, and outputs probabilities using `predict_proba()`.

## Database Tables

Raw weather table:

```text
weather_data_1y
```

Grid table:

```text
bangkok_grid_3km
```

ML feature table:

```text
weather_features
```

Spatial ML feature table:

```text
weather_features_spatial_neighbor
```

## Feature Engineering

The base feature table includes lag features such as:

```text
rain_1h_ago
rain_last_3h
rain_last_6h
humidity_1h_ago
cloud_cover_1h_ago
pressure_change_3h
```

It also includes:

```text
is_monsoon_season
```

Monsoon rule:

```text
May 15 to October 15 = 1
Otherwise = 0
```

The spatial-neighbor table adds averages from nearby grid cells:

```text
neighbor_count
neighbor_rain_1h_ago_avg
neighbor_rain_last_3h_avg
neighbor_humidity_1h_ago_avg
neighbor_cloud_cover_1h_ago_avg
neighbor_pressure_change_3h_avg
```

Neighbor definition:

```text
The 8 surrounding cells in a 3x3 window around each grid.
Edge grids use fewer neighbors.
```

## Target

The model predicts:

```text
will_rain_next_hour
```

This target is derived from the next hour's `rain` value:

```text
1 = next hour rain > 0
0 = next hour rain = 0
```

This is not independent rain-gauge truth. It is derived from the Open-Meteo rain data stored in PostgreSQL.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Set the local PostgreSQL password before running training/import scripts:

```powershell
$env:PGPASSWORD="your_password"
```

The scripts assume local PostgreSQL:

```text
host: localhost
port: 5432
database: postgres
user: postgres
```

## Import Current Weather Data

Import the current Bangkok date from Open-Meteo into PostgreSQL:

```bash
python scripts\import_weekend_weather.py
```

Import a specific date:

```bash
python scripts\import_weekend_weather.py --start-date 2026-05-19
```

The importer checks overlaps and uses `ON CONFLICT DO NOTHING`.

## Rebuild Features

After importing new raw rows, rebuild the feature table:

```bash
python scripts\rebuild_weather_features.py
```

Then rebuild and retrain the spatial model:

```bash
python scripts\train_spatial_neighbor_model.py
```

## Predict Next 6 Hours

Run:

```bash
python scripts\predict_spatial_neighbor_6h.py
```

Output:

```text
predictions/rain_probability_spatial_next_6h.csv
```

## Build Heatmap Data

Convert prediction CSV into browser heatmap data:

```bash
python scripts\build_heatmap_layer.py --prediction-csv predictions\rain_probability_spatial_next_6h.csv --output-js data\weather_heatmap_points.js
```

## View Map

Start a local server:

```bash
python -m http.server 8765 --bind 127.0.0.1
```

Open:

```text
http://127.0.0.1:8765/osm_weather_map.html
```

The map uses a white-to-dark-blue heatmap:

```text
White / pale blue = lower rain probability
Dark blue = higher rain probability
```

## Main Files

```text
scripts/import_weekend_weather.py
scripts/rebuild_weather_features.py
scripts/train_spatial_neighbor_model.py
scripts/predict_spatial_neighbor_6h.py
scripts/build_heatmap_layer.py
models/rainfall_probability_spatial_neighbor_model.joblib
data/weather_heatmap_points.js
osm_weather_map.html
```

