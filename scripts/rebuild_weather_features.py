import subprocess

PSQL = r"C:\Program Files\PostgreSQL\18\bin\psql.exe"
DB = ["-h", "localhost", "-p", "5432", "-U", "postgres", "-d", "postgres"]


def psql(sql):
    return subprocess.run(
        [PSQL, *DB, "-v", "ON_ERROR_STOP=1", "-c", sql],
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
    ).stdout


SQL = """
DROP TABLE IF EXISTS weather_features;

CREATE TABLE weather_features AS
WITH ordered AS (
    SELECT
        grid_number,
        longitude,
        latitude,
        weather_time,
        temperature_2m,
        rain,
        relative_humidity_2m,
        dew_point_2m,
        pressure_msl,
        surface_pressure,
        cloud_cover,
        CASE
            WHEN (
                (EXTRACT(MONTH FROM weather_time) = 5 AND EXTRACT(DAY FROM weather_time) >= 15)
                OR EXTRACT(MONTH FROM weather_time) IN (6, 7, 8, 9)
                OR (EXTRACT(MONTH FROM weather_time) = 10 AND EXTRACT(DAY FROM weather_time) <= 15)
            )
            THEN 1
            ELSE 0
        END AS is_monsoon_season,
        LAG(temperature_2m, 1) OVER w AS temperature_2m_1h_ago,
        LAG(relative_humidity_2m, 1) OVER w AS humidity_1h_ago,
        LAG(dew_point_2m, 1) OVER w AS dew_point_1h_ago,
        LAG(pressure_msl, 1) OVER w AS pressure_msl_1h_ago,
        LAG(surface_pressure, 1) OVER w AS surface_pressure_1h_ago,
        LAG(cloud_cover, 1) OVER w AS cloud_cover_1h_ago,
        LAG(rain, 1) OVER w AS rain_1h_ago,
        LAG(temperature_2m, 3) OVER w AS temperature_2m_3h_ago,
        LAG(relative_humidity_2m, 3) OVER w AS humidity_3h_ago,
        LAG(dew_point_2m, 3) OVER w AS dew_point_3h_ago,
        LAG(pressure_msl, 3) OVER w AS pressure_msl_3h_ago,
        LAG(surface_pressure, 3) OVER w AS surface_pressure_3h_ago,
        LAG(cloud_cover, 3) OVER w AS cloud_cover_3h_ago,
        SUM(rain) OVER (
            PARTITION BY grid_number
            ORDER BY weather_time
            ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
        ) AS rain_last_3h,
        LAG(temperature_2m, 6) OVER w AS temperature_2m_6h_ago,
        LAG(relative_humidity_2m, 6) OVER w AS humidity_6h_ago,
        LAG(dew_point_2m, 6) OVER w AS dew_point_6h_ago,
        LAG(pressure_msl, 6) OVER w AS pressure_msl_6h_ago,
        LAG(surface_pressure, 6) OVER w AS surface_pressure_6h_ago,
        LAG(cloud_cover, 6) OVER w AS cloud_cover_6h_ago,
        SUM(rain) OVER (
            PARTITION BY grid_number
            ORDER BY weather_time
            ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING
        ) AS rain_last_6h,
        LEAD(rain, 1) OVER w AS rain_next_hour
    FROM weather_data_1y
    WINDOW w AS (PARTITION BY grid_number ORDER BY weather_time)
)
SELECT
    grid_number,
    longitude,
    latitude,
    weather_time,
    temperature_2m,
    rain,
    relative_humidity_2m,
    dew_point_2m,
    pressure_msl,
    surface_pressure,
    cloud_cover,
    is_monsoon_season,
    temperature_2m_1h_ago,
    humidity_1h_ago,
    dew_point_1h_ago,
    pressure_msl_1h_ago,
    surface_pressure_1h_ago,
    cloud_cover_1h_ago,
    rain_1h_ago,
    temperature_2m_3h_ago,
    humidity_3h_ago,
    dew_point_3h_ago,
    pressure_msl_3h_ago,
    surface_pressure_3h_ago,
    cloud_cover_3h_ago,
    rain_last_3h,
    temperature_2m_6h_ago,
    humidity_6h_ago,
    dew_point_6h_ago,
    pressure_msl_6h_ago,
    surface_pressure_6h_ago,
    cloud_cover_6h_ago,
    rain_last_6h,
    pressure_msl - pressure_msl_1h_ago AS pressure_change_1h,
    pressure_msl - pressure_msl_3h_ago AS pressure_change_3h,
    pressure_msl - pressure_msl_6h_ago AS pressure_change_6h,
    surface_pressure - surface_pressure_1h_ago AS surface_pressure_change_1h,
    surface_pressure - surface_pressure_3h_ago AS surface_pressure_change_3h,
    surface_pressure - surface_pressure_6h_ago AS surface_pressure_change_6h,
    CASE
        WHEN rain_next_hour IS NULL THEN NULL
        WHEN rain_next_hour > 0 THEN 1
        ELSE 0
    END AS will_rain_next_hour
FROM ordered;

ALTER TABLE weather_features
  ADD PRIMARY KEY (grid_number, weather_time);

CREATE INDEX weather_features_time_idx
  ON weather_features (weather_time);

CREATE INDEX weather_features_target_idx
  ON weather_features (will_rain_next_hour);

CREATE INDEX weather_features_monsoon_idx
  ON weather_features (is_monsoon_season);

ANALYZE weather_features;

SELECT COUNT(*) AS rows,
       COUNT(DISTINCT grid_number) AS grids,
       MIN(weather_time) AS min_time,
       MAX(weather_time) AS max_time,
       COUNT(*) FILTER (WHERE will_rain_next_hour IS NULL) AS null_targets
FROM weather_features;
"""


if __name__ == "__main__":
    print(psql(SQL))
