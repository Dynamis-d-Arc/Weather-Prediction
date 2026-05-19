import csv
import io
import json
import argparse
import subprocess
import time
import urllib.parse
import urllib.request
from datetime import date
from urllib.error import HTTPError
from zoneinfo import ZoneInfo

PSQL = r"C:\Program Files\PostgreSQL\18\bin\psql.exe"
DB = ["-h", "localhost", "-p", "5432", "-U", "postgres", "-d", "postgres"]

HOURLY = (
    "temperature_2m,rain,relative_humidity_2m,dew_point_2m,"
    "pressure_msl,surface_pressure,cloud_cover"
)
FIELDS = HOURLY.split(",")
BATCH_SIZE = 50
STAGING = "weather_data_stage_import"
TIMEZONE = "Asia/Bangkok"


def psql(args, input_text=None):
    return subprocess.run(
        [PSQL, *DB, *args],
        input=input_text,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
    ).stdout


def fetch_batch(batch, start_date, end_date):
    params = {
        "latitude": ",".join(f"{lat:.8f}" for _, _, lat in batch),
        "longitude": ",".join(f"{lon:.8f}" for _, lon, _ in batch),
        "start_date": start_date,
        "end_date": end_date,
        "hourly": HOURLY,
        "models": "ecmwf_ifs",
        "timezone": TIMEZONE,
    }
    url = "https://historical-forecast-api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode(
        params, safe=","
    )
    last_error = None
    for attempt in range(1, 6):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "codex-weather-import/1.1"}
            )
            with urllib.request.urlopen(request, timeout=180) as response:
                payload = json.load(response)
            if isinstance(payload, dict) and payload.get("error"):
                raise RuntimeError(payload.get("reason") or str(payload))
            if isinstance(payload, dict):
                payload = [payload]
            if len(payload) != len(batch):
                raise RuntimeError(f"Expected {len(batch)} locations, got {len(payload)}")
            return payload
        except Exception as exc:
            last_error = exc
            wait = 90 * attempt if isinstance(exc, HTTPError) and exc.code == 429 else 20 * attempt
            print(
                f"Retry {attempt}/5 for grids {batch[0][0]}-{batch[-1][0]}: {exc}; "
                f"sleeping {wait}s",
                flush=True,
            )
            time.sleep(wait)
    raise last_error


def parse_args():
    today = date.today().isoformat()
    parser = argparse.ArgumentParser(
        description="Import Open-Meteo historical forecast weather into PostgreSQL."
    )
    parser.add_argument(
        "--start-date",
        default=today,
        help="Start date in YYYY-MM-DD. Defaults to today's date.",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="End date in YYYY-MM-DD. Defaults to --start-date.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Fetch even if all expected rows already exist. Existing rows are still skipped by ON CONFLICT.",
    )
    return parser.parse_args()


def expected_rows(start_date, end_date, grid_count):
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    if end < start:
        raise ValueError("end-date cannot be earlier than start-date")
    days = (end - start).days + 1
    return days * 24 * grid_count


def existing_summary(start_date, end_date):
    return psql(
        [
            "-c",
            f"""
            SELECT weather_time::date AS day,
                   COUNT(*) AS rows,
                   COUNT(DISTINCT grid_number) AS grids
            FROM weather_data_1y
            WHERE weather_time::date BETWEEN DATE '{start_date}' AND DATE '{end_date}'
            GROUP BY weather_time::date
            ORDER BY day;
            """,
        ]
    )


def main():
    args = parse_args()
    start_date = args.start_date
    end_date = args.end_date or start_date

    print(f"Import date range: {start_date} to {end_date}", flush=True)

    psql(
        [
            "-v",
            "ON_ERROR_STOP=1",
            "-c",
            (
                f"DROP TABLE IF EXISTS {STAGING}; "
                f"CREATE UNLOGGED TABLE {STAGING} (LIKE weather_data_1y INCLUDING DEFAULTS);"
            ),
        ]
    )

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
    expected = expected_rows(start_date, end_date, len(grids))
    existing_count = int(
        psql(
            [
                "-At",
                "-c",
                (
                    "SELECT COUNT(*) FROM weather_data_1y "
                    f"WHERE weather_time::date BETWEEN DATE '{start_date}' AND DATE '{end_date}';"
                ),
            ]
        ).strip()
    )
    print("Existing rows for date range before import:", flush=True)
    print(existing_summary(start_date, end_date), flush=True)
    print(f"Existing rows: {existing_count:,}; expected full coverage: {expected:,}", flush=True)
    if existing_count >= expected and not args.force:
        print(
            "Date range already appears complete. Use --force to fetch again; "
            "overlaps will still be skipped.",
            flush=True,
        )
        psql(["-v", "ON_ERROR_STOP=1", "-c", f"DROP TABLE IF EXISTS {STAGING};"])
        return

    copy_sql = (
        f"COPY {STAGING} (grid_number,longitude,latitude,weather_time,"
        "temperature_2m,rain,relative_humidity_2m,dew_point_2m,pressure_msl,"
        "surface_pressure,cloud_cover) FROM STDIN WITH (FORMAT csv)"
    )
    insert_sql = f"""
        INSERT INTO weather_data_1y (
            grid_number, longitude, latitude, weather_time,
            temperature_2m, rain, relative_humidity_2m, dew_point_2m,
            pressure_msl, surface_pressure, cloud_cover, model, timezone, fetched_at
        )
        SELECT
            grid_number, longitude, latitude, weather_time,
            temperature_2m, rain, relative_humidity_2m, dew_point_2m,
            pressure_msl, surface_pressure, cloud_cover, model, timezone, fetched_at
        FROM {STAGING}
        ON CONFLICT (grid_number, weather_time) DO NOTHING;
    """

    staged_rows = 0
    for offset in range(0, len(grids), BATCH_SIZE):
        batch = grids[offset : offset + BATCH_SIZE]
        payloads = fetch_batch(batch, start_date, end_date)
        rows = []
        for (grid_number, lon, lat), payload in zip(batch, payloads):
            hourly = payload.get("hourly") or {}
            missing = [field for field in ["time", *FIELDS] if field not in hourly]
            if missing:
                raise RuntimeError(f"Grid {grid_number} missing fields {missing}")
            for i, weather_time in enumerate(hourly["time"]):
                weather_day = weather_time[:10]
                if start_date <= weather_day <= end_date:
                    rows.append(
                        [
                            grid_number,
                            lon,
                            lat,
                            weather_time,
                            *[
                                hourly[field][i] if hourly[field][i] is not None else ""
                                for field in FIELDS
                            ],
                        ]
                    )

        buffer = io.StringIO()
        csv.writer(buffer, lineterminator="\n").writerows(rows)
        psql(["-v", "ON_ERROR_STOP=1", "-c", copy_sql], buffer.getvalue())
        staged_rows += len(rows)
        print(
            f"Fetched grids {batch[0][0]}-{batch[-1][0]}, "
            f"staged {len(rows)} rows, total staged {staged_rows}",
            flush=True,
        )
        time.sleep(3)

    psql(["-v", "ON_ERROR_STOP=1", "-c", insert_sql])
    summary = psql(
        [
            "-c",
            f"""
            SELECT weather_time::date AS day,
                   COUNT(*) AS rows,
                   COUNT(DISTINCT grid_number) AS grids
            FROM weather_data_1y
            WHERE weather_time::date BETWEEN DATE '{start_date}' AND DATE '{end_date}'
            GROUP BY weather_time::date
            ORDER BY day;

            SELECT COUNT(*) AS total_rows,
                   MIN(weather_time) AS min_time,
                   MAX(weather_time) AS max_time
            FROM weather_data_1y;
            """,
        ]
    )
    psql(["-v", "ON_ERROR_STOP=1", "-c", f"DROP TABLE IF EXISTS {STAGING};"])
    print(summary)


if __name__ == "__main__":
    main()
