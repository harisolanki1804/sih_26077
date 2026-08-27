# Project VARUNA — ML Team Data & API Guide

This repository provides the spatial datasets, historical storm data, and backend APIs for the **Model Development Team (Hari & Arya)**.

---

## 📍 Pilot Area & Grid Specifications

* **Location**: Mumbai Metropolitan Region (Mithi River Basin & Urban Corridors)
* **Bounding Box**: Latitude `18.980` to `19.160` N | Longitude `72.800` to `72.960` E
* **Grid Layout**: 10 Latitude steps × 9 Longitude steps = **90 spatial cells** (Resolution: `0.02°` / ~2.2 km)
* **Hotspots**: Hindmata (3.2m), Kurla West (4.5m), BKC Outfall (5.0m), Sion Circle (3.8m), Andheri Subway (6.1m), Mahim Bay (1.2m)

---

## ⏱️ Data Timestamps & Temporal Coverage

* **Time Window**: `2024-07-26T00:00:00Z` to `2024-07-28T23:00:00Z` (72 continuous hourly timesteps)
* **Step Size**: 1 hour (`1h`)
* **Storm Peak**: Timestep 36 (`2024-07-27T11:00:00Z`) — Peak cloudburst rainfall (>100 mm/hr) + 4.8m high-tide lock.
* **Suggested Split**: 
  * **Train**: Timesteps 1–48 (`2024-07-26T00:00Z` to `2024-07-27T23:00Z`)
  * **Validation/Test**: Timesteps 49–72 (`2024-07-28T00:00Z` to `2024-07-28T23:00Z`)

---

## 📁 Datasets Available

### 1. Fused Feature Matrix (Model-Ready)
* **File**: `data/feature_grid_timeseries.json`
* **Structure**: 72 hourly timesteps × 90 cells = **6,480 total samples**
* **Contents**: Fused table containing all input features + ground truth target labels per cell.

### 2. Raw Datasets (Unprocessed)
* `data/raw/dem/mumbai_srtm_dem_raw.csv` — 90 spatial elevation points (`cell_id`, `latitude`, `longitude`, `elevation_meters`)
* `data/raw/rainfall/mumbai_hourly_rainfall_raw.csv` — 72-hour hourly weather (`timestamp`, `precipitation_mm`, `temperature_c`, `relative_humidity_pct`, `surface_pressure_hpa`, `wind_speed_10m_kmh`, `wind_direction_10m_deg`, `wind_gusts_10m_kmh`, `soil_moisture_0_to_7cm_m3m3`)
* `data/raw/moisture/mumbai_soil_moisture_raw.csv` — Multi-layer soil moisture (`timestamp`, `soil_moisture_0_to_7cm_m3m3`, `soil_moisture_7_to_28cm_m3m3`, `surface_pressure_hpa`)

---

## 📊 Features & Target Labels

### Input Features (Per Cell / Per Hour)
| Category | Column / Key | Description | Unit / Range |
|---|---|---|---|
| **Topography** | `elevation_m` | Ground height above sea level | meters (1.8 to 45.0) |
| | `slope_deg` | Terrain slope angle | degrees (0.2 to 18.0) |
| | `runoff_coefficient` | Surface imperviousness factor | 0.0 to 1.0 |
| | `drainage_outfall_dist_m` | Distance to Mahim/sea outlet | meters |
| | `is_depression_bowl` | Lowland basin trap flag | boolean |
| **Moisture** | `rainfall_1h_mm` | Rainfall in the past 1 hour | mm/hr |
| | `rainfall_3h_mm` | Cumulative rainfall past 3 hours | mm |
| | `rainfall_6h_mm` | Cumulative rainfall past 6 hours | mm |
| | `rainfall_24h_mm` | Cumulative rainfall past 24 hours | mm |
| | `soil_moisture_pct` | Soil moisture saturation level | 0.0 to 100.0 % |
| **Instability** | `cape_instability_jkg` | Convective Available Potential Energy | J/kg (500 to 3200) |
| | `cloud_top_temp_celsius` | Cloud top temperature | °C (-35 to -85) |
| | `ctt_drop_rate_c_hr` | Hourly cloud-top cooling rate | °C/hr |
| **Kinematics** | `wind_speed_10m_kmh` | 10m Wind speed | km/h |
| | `wind_direction_10m_deg` | 10m Wind direction | degrees (0 to 360) |
| | `wind_u_ms` / `wind_v_ms` | Wind U and V vector components | m/s |
| | `wind_gusts_kmh` | Peak wind gusts | km/h |
| **Tidal** | `tide_height_m` | Sea water tide height | meters (1.5 to 4.8) |
| | `is_high_tide_locked` | High tide threshold lock (>4.5m) | boolean |

### Target Labels to Predict
| Target Variable | Problem Type | Range / Classes | Description |
|---|---|---|---|
| `target_observed_flood_depth_cm` | **Regression** | `0.0` to `80.0+` cm | Estimated street-level water ponding depth |
| `target_severity_class` | **Classification** | `0`: LOW, `1`: MED, `2`: HIGH, `3`: CRITICAL | Overall hazard severity band |
| `target_flash_flood_flag` | **Binary Classification** | `0` or `1` | Flash flood occurrence trigger |
| `target_cloudburst_flag` | **Binary Classification** | `0` or `1` | Intense cloudburst occurrence trigger |
| `target_waterlogging_flag` | **Binary Classification** | `0` or `1` | Localized street waterlogging trigger |

---

## 🐍 How to Load Data in Python

```python
import json
import pandas as pd

# Option A: Load complete fused dataset into a Pandas DataFrame
with open('data/feature_grid_timeseries.json', 'r') as f:
    data = json.load(f)

records = []
for ts in data['timesteps']:
    t_id = ts['timestep_id']
    t_stamp = ts['timestamp']
    for cell in ts['features']:
        records.append({'timestep_id': t_id, 'timestamp': t_stamp, **cell})

df = pd.DataFrame(records)
print(df.shape)  # (6480, 28)
print(df.head())

# Option B: Load raw CSVs directly
df_dem = pd.read_csv('data/raw/dem/mumbai_srtm_dem_raw.csv')
df_rain = pd.read_csv('data/raw/rainfall/mumbai_hourly_rainfall_raw.csv')
df_moist = pd.read_csv('data/raw/moisture/mumbai_soil_moisture_raw.csv')
```

---

## 🔌 Sending Model Predictions to Backend API

Once your model predicts risks/depths, push them to the backend using these endpoints:

### 1. Create a Risk Event
```http
POST /api/v1/events
Content-Type: application/json

{
  "region_id": "cf075053-2c14-4638-86e8-b190141257e3",
  "event_code": "EVT-BOM-20240726-DELUGE",
  "title": "Mumbai Monsoon Cloudburst Event",
  "start_time": "2024-07-26T00:00:00Z",
  "peak_rainfall_rate_mm_hr": 115.0,
  "max_predicted_depth_cm": 42.5,
  "severity_level": "CRITICAL",
  "status": "ACTIVE",
  "confidence_score": 92.0
}
```

### 2. Push Cell Alerts
```http
POST /api/v1/alerts
Content-Type: application/json

{
  "region_id": "cf075053-2c14-4638-86e8-b190141257e3",
  "cell_lat": 19.018,
  "cell_lon": 72.843,
  "timestamp": "2024-07-27T11:00:00Z",
  "alert_type": "FLASH_FLOOD",
  "severity": "CRITICAL",
  "risk_score_total": 91.5,
  "rainfall_score_contrib": 38.0,
  "soil_saturation_score_contrib": 24.0,
  "topography_score_contrib": 18.5,
  "atmospheric_instability_score_contrib": 11.0,
  "flood_depth_estimate_cm": 38.2,
  "trust_score": 94.0,
  "trust_level": "HIGH_CONFIDENCE",
  "reasoning_summary": "Critical waterlogging at Hindmata depression bowl (3.2m elevation) due to 110mm/hr cloudburst and Mahim tidal lock.",
  "recommended_action": "Deploy de-watering pumps to Dadar TT underpass; divert traffic."
}
```

---

## 🚀 Quickstart

```bash
# 1. Clone repo & switch to branch
git clone https://github.com/harisolanki1804/sih_26077.git
cd sih_26077
git checkout api-and-data-fetching

# 2. Setup virtual environment & install requirements
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt

# 3. Start Backend Server
python backend/run.py
```
* **Swagger UI API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
* **Raw Data Catalog**: [http://localhost:8000/api/v1/data/raw/catalog](http://localhost:8000/api/v1/data/raw/catalog)
