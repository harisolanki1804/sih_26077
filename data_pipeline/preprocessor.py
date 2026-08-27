"""
VARUNA Data Pipeline: Preprocessor & Feature Grid Generator
------------------------------------------------------------
Transforms raw rasters and time-series into a clean, normalized, model-ready
spatio-temporal feature matrix with complete Kinematics, Instability, Moisture,
Topography, and Supervised Training Target Labels.
"""

import os
import json
import logging
from typing import Dict, List, Any
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("VARUNA.Preprocessor")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DEM_PATH = os.path.join(DATA_DIR, "dem", "mumbai_srtm_dem.json")
RAINFALL_PATH = os.path.join(DATA_DIR, "rainfall", "mumbai_historical_deluge.json")
OUTPUT_PATH = os.path.join(DATA_DIR, "feature_grid_timeseries.json")


def load_raw_datasets():
    with open(DEM_PATH, "r", encoding="utf-8") as f:
        dem_data = json.load(f)
    with open(RAINFALL_PATH, "r", encoding="utf-8") as f:
        rainfall_data = json.load(f)
    return dem_data, rainfall_data


def process_features() -> Dict[str, Any]:
    dem_data, rainfall_data = load_raw_datasets()
    cells_dem = {c["cell_index"]: c for c in dem_data["cells"]}
    
    logger.info(f"Processing {len(rainfall_data['timesteps'])} timesteps across {len(cells_dem)} grid cells...")

    processed_timesteps: List[Dict[str, Any]] = []

    for ts in rainfall_data["timesteps"]:
        t_id = ts["timestep_id"]
        t_iso = ts["timestamp"]
        tide_locked = ts["is_high_tide_locked"]
        tide_height = ts["tide_height_m"]

        features_at_t = []

        for cell_raw in ts["cells"]:
            c_idx = cell_raw["cell_index"]
            dem_meta = cells_dem[c_idx]

            rain_1h = cell_raw["rainfall_1h_mm"]
            rain_3h = cell_raw["rainfall_3h_mm"]
            rain_6h = cell_raw["rainfall_6h_mm"]
            rain_24h = cell_raw["rainfall_24h_mm"]
            soil_moist_pct = cell_raw["soil_moisture_pct"]
            cape = cell_raw["cape_instability_jkg"]
            ctt = cell_raw.get("cloud_top_temp_celsius", -45.0)
            ctt_drop = cell_raw.get("ctt_drop_rate_c_hr", 0.0)
            ws_kmh = cell_raw.get("wind_speed_10m_kmh", 25.0)
            wd_deg = cell_raw.get("wind_direction_10m_deg", 240.0)
            wu = cell_raw.get("wind_u_ms", -5.0)
            wv = cell_raw.get("wind_v_ms", -3.0)
            wg_kmh = cell_raw.get("wind_gusts_kmh", 35.0)

            elev = cell_raw["elevation_m"]
            slope = cell_raw["slope_deg"]
            runoff_c = cell_raw["runoff_coefficient"]
            dist_outfall = cell_raw["drainage_outfall_dist_m"]

            # Derived Indicators
            # 1. Soil saturation factor [0.0 - 1.0]
            sat_factor = min(1.0, max(0.0, (soil_moist_pct - 30.0) / 70.0))

            # 2. Topographical Depression Index
            retention_index = max(0.0, round((15.0 - min(15.0, elev)) / 15.0 * (1.0 / (max(0.3, slope))), 3))

            # 3. Dynamic Effective Surface Runoff (mm/hr)
            effective_runoff_mm_hr = round(runoff_c * rain_1h * (0.8 + 0.6 * sat_factor), 2)

            # 4. Tidal Outfall Resistance Factor
            tidal_backwater_factor = 1.0
            if tide_locked and dist_outfall < 6000.0:
                tidal_backwater_factor = round(1.0 + (tide_height - 3.5) * 0.7 * (1.0 - (dist_outfall / 6000.0)), 2)

            features_at_t.append({
                "cell_index": c_idx,
                "lat": cell_raw["lat"],
                "lon": cell_raw["lon"],
                # Topographical Layer
                "elevation_m": elev,
                "slope_deg": slope,
                "runoff_coefficient": runoff_c,
                "drainage_outfall_dist_m": dist_outfall,
                "retention_index": retention_index,
                "is_depression_bowl": dem_meta["is_depression_bowl"],
                # Moisture & Precipitation Layer
                "rainfall_1h_mm": rain_1h,
                "rainfall_3h_mm": rain_3h,
                "rainfall_6h_mm": rain_6h,
                "rainfall_24h_mm": rain_24h,
                "soil_moisture_pct": soil_moist_pct,
                "soil_saturation_factor": round(sat_factor, 3),
                "effective_runoff_mm_hr": effective_runoff_mm_hr,
                # Instability Layer
                "cape_instability_jkg": cape,
                "cloud_top_temp_celsius": ctt,
                "ctt_drop_rate_c_hr": ctt_drop,
                # Kinematics & Lift Layer
                "wind_speed_10m_kmh": ws_kmh,
                "wind_direction_10m_deg": wd_deg,
                "wind_u_ms": wu,
                "wind_v_ms": wv,
                "wind_gusts_kmh": wg_kmh,
                # Coastal Boundary State
                "tidal_backwater_factor": tidal_backwater_factor,
                # Supervised Model Target Labels (Ground Truth Proxies)
                "target_observed_flood_depth_cm": cell_raw.get("target_observed_flood_depth_cm", 0.0),
                "target_severity_class": cell_raw.get("target_severity_class", 0),
                "target_flash_flood_flag": cell_raw.get("target_flash_flood_flag", 0),
                "target_cloudburst_flag": cell_raw.get("target_cloudburst_flag", 0),
                "target_waterlogging_flag": cell_raw.get("target_waterlogging_flag", 0)
            })

        processed_timesteps.append({
            "timestep_id": t_id,
            "timestamp": t_iso,
            "phase": ts["phase"],
            "tide_height_m": tide_height,
            "is_high_tide_locked": tide_locked,
            "avg_rainfall_1h_mm": ts["regional_avg_rainfall_1h_mm"],
            "max_rainfall_1h_mm": ts["max_cell_rainfall_1h_mm"],
            "avg_cloud_top_temp_celsius": ts.get("regional_avg_ctt_celsius", -45.0),
            "features": features_at_t
        })

    model_ready_package = {
        "region_code": rainfall_data["region_code"],
        "event_code": rainfall_data["event_code"],
        "provenance_note": rainfall_data.get("provenance_note", "Calibrated high-density multi-hazard case study."),
        "total_timesteps": len(processed_timesteps),
        "total_grid_cells": len(cells_dem),
        "feature_layers": [
            "Topography (DEM 30m, slope, depression index)",
            "Moisture (Precipitation 1h/3h/6h/24h, soil moisture saturation)",
            "Atmospheric Instability (CAPE, CTT, CTT drop-rate)",
            "Kinematics & Lift (Wind 10m speed/direction, U/V vectors, gusts)",
            "Coastal (Tide surge height, tidal lock flag)"
        ],
        "target_labels": [
            "target_observed_flood_depth_cm (Regression Target)",
            "target_severity_class [0: LOW, 1: MEDIUM, 2: HIGH, 3: CRITICAL] (Multiclass Target)",
            "target_flash_flood_flag (Binary Target)",
            "target_cloudburst_flag (Binary Target)",
            "target_waterlogging_flag (Binary Target)"
        ],
        "timesteps": processed_timesteps
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(model_ready_package, f, indent=2)

    logger.info(f"Successfully created model-ready feature grid time-series at {OUTPUT_PATH}")
    return model_ready_package


if __name__ == "__main__":
    process_features()
