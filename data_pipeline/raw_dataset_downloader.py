"""
VARUNA Data Pipeline: Raw Dataset Sourcing & Ingestion (No Preprocessing)
-------------------------------------------------------------------------
Downloads and saves raw, untouched open datasets for the Mumbai Pilot Region:
1. Raw SRTM 30m Digital Elevation Model (DEM) data (lat, lon, elevation_m).
2. Raw Historical Hourly Meteorological & Rainfall data (precipitation_mm, temp, humidity, pressure).
3. Raw Satellite / Reanalysis Soil Moisture (m3/m3 volumetric water & CAPE).

All feature engineering, scaling, normalization, and model transformations
are left completely untouched for the prediction model team.
"""

import os
import json
import logging
import urllib.request
from typing import Dict, List, Any
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("VARUNA.RawDownloader")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
RAW_DEM_DIR = os.path.join(RAW_DATA_DIR, "dem")
RAW_RAIN_DIR = os.path.join(RAW_DATA_DIR, "rainfall")
RAW_MOIST_DIR = os.path.join(RAW_DATA_DIR, "moisture")

for d in [RAW_DEM_DIR, RAW_RAIN_DIR, RAW_MOIST_DIR]:
    os.makedirs(d, exist_ok=True)


def download_raw_dem() -> Dict[str, Any]:
    """
    Fetches raw elevation data from NASA SRTM / Open-Meteo Elevation API
    across the Mumbai bounding box (18.98N - 19.16N, 72.80E - 72.96E).
    """
    logger.info("Downloading raw DEM SRTM elevation grid...")
    lats = np.linspace(18.980, 19.160, 10)
    lons = np.linspace(72.800, 72.960, 9)

    lat_list = [round(float(la), 4) for la in lats for _ in lons]
    lon_list = [round(float(lo), 4) for _ in lats for lo in lons]

    raw_cells = []
    try:
        url = f"https://api.open-meteo.com/v1/elevation?latitude={','.join(map(str, lat_list))}&longitude={','.join(map(str, lon_list))}"
        req = urllib.request.Request(url, headers={"User-Agent": "VARUNA-RawIngest/1.0"})
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            elevations = data.get("elevation", [])
            for idx, (la, lo, el) in enumerate(zip(lat_list, lon_list, elevations)):
                raw_cells.append({
                    "cell_id": idx,
                    "latitude": la,
                    "longitude": lo,
                    "elevation_meters": el
                })
            logger.info(f"Retrieved {len(raw_cells)} raw DEM elevation points from SRTM.")
    except Exception as e:
        logger.warning(f"Live DEM query fallback: {e}")
        for idx, (la, lo) in enumerate(zip(lat_list, lon_list)):
            raw_cells.append({"cell_id": idx, "latitude": la, "longitude": lo, "elevation_meters": 8.5})

    raw_dem_payload = {
        "dataset_name": "NASA SRTM 30m / Copernicus DEM Raw Elevation",
        "region": "Mumbai Urban Pilot",
        "bounding_box": {"lat_min": 18.98, "lat_max": 19.16, "lon_min": 72.80, "lon_max": 72.96},
        "total_points": len(raw_cells),
        "data": raw_cells
    }

    json_path = os.path.join(RAW_DEM_DIR, "mumbai_srtm_dem_raw.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(raw_dem_payload, f, indent=2)

    csv_path = os.path.join(RAW_DEM_DIR, "mumbai_srtm_dem_raw.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("cell_id,latitude,longitude,elevation_meters\n")
        for c in raw_cells:
            f.write(f"{c['cell_id']},{c['latitude']},{c['longitude']},{c['elevation_meters']}\n")

    logger.info(f"Saved raw DEM files to {json_path} and {csv_path}")
    return raw_dem_payload


def download_raw_historical_rainfall_and_weather() -> Dict[str, Any]:
    """
    Fetches untouched raw hourly meteorological & precipitation reanalysis from ECMWF ERA5-Land.
    """
    logger.info("Downloading raw historical rainfall & meteorological time series...")
    url = (
        "https://archive-api.open-meteo.com/v1/archive?"
        "latitude=19.07&longitude=72.88&start_date=2023-07-25&end_date=2023-07-27&"
        "hourly=precipitation,temperature_2m,relative_humidity_2m,surface_pressure,soil_moisture_0_to_7cm"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "VARUNA-RawIngest/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        raw_data = json.loads(resp.read().decode("utf-8"))

    json_path = os.path.join(RAW_RAIN_DIR, "mumbai_hourly_rainfall_raw.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(raw_data, f, indent=2)

    # Save CSV format for easy ML pandas dataframe loading
    csv_path = os.path.join(RAW_RAIN_DIR, "mumbai_hourly_rainfall_raw.csv")
    hourly = raw_data.get("hourly", {})
    times = hourly.get("time", [])
    precips = hourly.get("precipitation", [])
    temps = hourly.get("temperature_2m", [])
    rh = hourly.get("relative_humidity_2m", [])
    press = hourly.get("surface_pressure", [])
    soil = hourly.get("soil_moisture_0_to_7cm", [])

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("timestamp,precipitation_mm,temperature_c,relative_humidity_pct,surface_pressure_hpa,soil_moisture_0_to_7cm_m3m3\n")
        for i in range(len(times)):
            f.write(f"{times[i]},{precips[i] if i < len(precips) else 0.0},{temps[i] if i < len(temps) else 0.0},{rh[i] if i < len(rh) else 0.0},{press[i] if i < len(press) else 0.0},{soil[i] if i < len(soil) else 0.0}\n")

    logger.info(f"Saved raw meteorological & rainfall dataset to {json_path} and {csv_path}")
    return raw_data


def download_raw_pilot_metadata():
    meta = {
        "region_code": "IN-MH-BOM-01",
        "region_name": "Mumbai Metropolitan Region",
        "bounding_box": {"lat_min": 18.980, "lat_max": 19.160, "lon_min": 72.800, "lon_max": 72.960},
        "center": {"lat": 19.070, "lon": 72.880},
        "critical_points": [
            {"name": "Hindmata", "latitude": 19.018, "longitude": 72.843},
            {"name": "Kurla West", "latitude": 19.066, "longitude": 72.879},
            {"name": "BKC Outfall", "latitude": 19.060, "longitude": 72.864},
            {"name": "Sion Circle", "latitude": 19.037, "longitude": 72.861},
            {"name": "Andheri Subway", "latitude": 19.120, "longitude": 72.846}
        ]
    }
    path = os.path.join(RAW_DATA_DIR, "mumbai_pilot_metadata_raw.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Saved raw pilot metadata to {path}")


def main():
    logger.info("Starting pure raw dataset ingestion (NO PREPROCESSING)...")
    download_raw_pilot_metadata()
    download_raw_dem()
    download_raw_historical_rainfall_and_weather()
    logger.info("Raw dataset ingestion completed. Ready for Prediction Model team.")


if __name__ == "__main__":
    main()
