import os
import json
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, status
from app.core.config import settings

router = APIRouter()

RAW_DIR = os.path.join(os.path.dirname(settings.BASE_DIR), "data", "raw")


@router.get("/dem", summary="Get Raw Unprocessed DEM Topography Points")
def get_raw_dem() -> Dict[str, Any]:
    """Returns raw SRTM elevation points (cell_id, latitude, longitude, elevation_meters) without any derived slopes or indices."""
    path = os.path.join(RAW_DIR, "dem", "mumbai_srtm_dem_raw.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Raw DEM dataset not found.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@router.get("/rainfall", summary="Get Raw Unprocessed Historical Hourly Rainfall & Weather")
def get_raw_rainfall() -> Dict[str, Any]:
    """Returns raw ECMWF ERA5-Land hourly precipitation, temperature, humidity, and soil moisture."""
    path = os.path.join(RAW_DIR, "rainfall", "mumbai_hourly_rainfall_raw.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Raw rainfall dataset not found.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@router.get("/catalog", summary="Get Raw Datasets File Catalog for Model Team")
def get_raw_catalog() -> Dict[str, Any]:
    """Lists raw dataset file paths available for pandas or numpy ingestion."""
    return {
        "status": "RAW_UNPROCESSED_READY",
        "description": "Untouched raw open datasets for feature engineering and prediction model team.",
        "files": {
            "dem_elevation_csv": os.path.join(RAW_DIR, "dem", "mumbai_srtm_dem_raw.csv"),
            "dem_elevation_json": os.path.join(RAW_DIR, "dem", "mumbai_srtm_dem_raw.json"),
            "rainfall_weather_csv": os.path.join(RAW_DIR, "rainfall", "mumbai_hourly_rainfall_raw.csv"),
            "rainfall_weather_json": os.path.join(RAW_DIR, "rainfall", "mumbai_hourly_rainfall_raw.json"),
            "pilot_metadata_json": os.path.join(RAW_DIR, "mumbai_pilot_metadata_raw.json")
        },
        "python_snippet": "import pandas as pd\ndf_dem = pd.read_csv(r'VARUNA/data/raw/dem/mumbai_srtm_dem_raw.csv')\ndf_rain = pd.read_csv(r'VARUNA/data/raw/rainfall/mumbai_hourly_rainfall_raw.csv')"
    }
