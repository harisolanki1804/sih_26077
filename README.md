<div align="center">

# 🌊 Project VARUNA
### Hyper-Local AI Early Warning & Multi-Hazard Decision Platform for Urban Flash Floods

[![Python Version](https://img.shields.io/badge/Python-3.12%2B-blue.svg?style=for-the-badge&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-2.0%2B-D71F00.svg?style=for-the-badge&logo=sqlalchemy)](https://www.sqlalchemy.org)
[![Database](https://img.shields.io/badge/Database-SQLite%20%7C%20PostgreSQL-336791.svg?style=for-the-badge&logo=postgresql)](https://postgresql.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

</div>

---

## 📖 Executive Summary & Data Fusion Concept

**Project VARUNA** addresses urban flash-flood and severe convective cloudburst nowcasting by solving the **Multi-Modal Data Fusion** challenge:

1. **The Core Challenge**: Satellite imagery (infrared/microwave matrices), atmospheric reanalysis (vertical pressure & temperature profiles), and topography (DEM terrain rasters) exist on completely different grids, spatial resolutions, and time intervals.
2. **Spatio-Temporal Alignment (Phase 2 Output)**: We align all modalities onto **one shared geographic grid** ($10 \times 9 = 90$ micro-cells at $0.02^\circ$ resolution, $\sim 2.2\text{km}$) across **72 synchronized hourly timesteps**.
3. **Learned Fusion Layer**: Rather than naive feature concatenation, the model team uses a learned weighting layer that determines the relative contribution of each pillar:
   $$\text{Fused State} = f\Big(w_{\text{moisture}} \cdot \mathbf{X}_{\text{moist}} + w_{\text{lift}} \cdot \mathbf{X}_{\text{kinematics}} + w_{\text{instability}} \cdot \mathbf{X}_{\text{cape}} + w_{\text{topo}} \cdot \mathbf{X}_{\text{dem}}\Big)$$
   *e.g., moisture and CAPE dominate thunderstorm nowcasting, while DEM elevation and tidal backwater dominate street flood depth.*

---

## 📍 Pilot Study: Mumbai Urban Flood Corridor

* **Region Code**: `IN-MH-BOM-01`
* **Spatial Bounding Box**: Latitude $[18.980^\circ\text{N}, 19.160^\circ\text{N}]$, Longitude $[72.800^\circ\text{E}, 72.960^\circ\text{E}]$
* **Grid Resolution**: $0.02^\circ$ ($10 \times 9 = 90$ synchronized spatial cells)
* **Critical Monitored Hotspots**:
  * **Hindmata / Dadar TT Circle** ($3.2\text{m}$ MSL — tidal depression bowl)
  * **Kurla West & LBS Marg** ($4.5\text{m}$ MSL — Mithi River confluence & railway nexus)
  * **Bandra-Kurla Complex (BKC) Outfall** ($5.0\text{m}$ MSL — Vakola nullah discharge)
  * **Sion Circle & Gandhi Market** ($3.8\text{m}$ MSL — arterial highway underpass)
  * **Andheri Subway** ($6.1\text{m}$ MSL — rapid flash-flood subway trap)
  * **Mahim Bay Tidal Outfall** ($1.2\text{m}$ MSL — tidal lockout during spring surge)

---

## ⏱️ Dataset Timestamps & Temporal Details

The datasets are **100% synchronized** across a continuous **72-hour extreme monsoon deluge window**:

### 1. Synchronized Time Window
* **Start**: `2024-07-26T00:00:00Z` (00:00 UTC / 05:30 IST)
* **End**: `2024-07-28T23:00:00Z` (23:00 UTC / 04:30 IST next day)
* **Step Size**: **1-hour ($1\text{h}$)** continuous increments ($T_1 \to T_{72}$)
* **Format**: ISO 8601 UTC (`YYYY-MM-DDTHH:MM`)

### 2. Storm Phase Timeline

| Timestep Range | Timestamp (UTC) | Meteorological Storm Phase | Regional Avg Rain Rate | High Tide Lock |
|---|---|---|---|---|
| **$T_1 - T_{18}$** | `2024-07-26T00:00Z` – `17:00Z` | **Pre-Event Baseline**: Overcast skies, light drizzle | $0.5 - 4.0\text{ mm/hr}$ | Normal ($2.1\text{m}$) |
| **$T_{19} - T_{30}$** | `2024-07-26T18:00Z` – `2024-07-27T05:00Z` | **Squall Line Inflow**: Rapid moisture convergence & wind pickup | $15.0 - 35.0\text{ mm/hr}$ | Rising ($3.6\text{m}$) |
| **$T_{31} - T_{42}$** | `2024-07-27T06:00Z` – `17:00Z` | **Severe Cloudburst Peak**: Critical waterlogging (Peak at $T_{36}$) | **$60.0 - 125.0\text{ mm/hr}$** | **LOCKED ($4.8\text{m}$ surge)** |
| **$T_{43} - T_{54}$** | `2024-07-27T18:00Z` – `2024-07-28T05:00Z` | **Sustained Downpour**: High surface runoff saturation | $25.0 - 45.0\text{ mm/hr}$ | Receding |
| **$T_{55} - T_{72}$** | `2024-07-28T06:00Z` – `23:00Z` | **Recession & Drainage**: Sump de-watering phase | $1.0 - 15.0\text{ mm/hr}$ | Cleared ($1.8\text{m}$) |

---

## 🔬 Feature Matrix & Ground-Truth Target Labels

The model-ready dataset [`data/feature_grid_timeseries.json`](data/feature_grid_timeseries.json) provides a complete feature vector + supervised labels for every cell at every hour:

### 📥 Input Features (All 4 Hazard Pillars)
1. **Topography**: `elevation_m`, `slope_deg`, `runoff_coefficient`, `drainage_outfall_dist_m`, `retention_index`, `is_depression_bowl`.
2. **Moisture**: `rainfall_1h_mm`, `rainfall_3h_mm`, `rainfall_6h_mm`, `rainfall_24h_mm`, `soil_moisture_pct`, `soil_saturation_factor`, `effective_runoff_mm_hr`.
3. **Atmospheric Instability**: `cape_instability_jkg`, `cloud_top_temp_celsius` (per cell), `ctt_drop_rate_c_hr`.
4. **Kinematics & Lift**: `wind_speed_10m_kmh`, `wind_direction_10m_deg`, `wind_u_ms`, `wind_v_ms`, `wind_gusts_kmh`.
5. **Coastal Boundary**: `tide_height_m`, `tidal_backwater_factor`, `is_high_tide_locked`.

### 🎯 Supervised Target Labels (Ground Truth Proxies)
* `target_observed_flood_depth_cm` — Continuous inundation depth in centimeters (Regression Target).
* `target_severity_class` — Hazard band: `0=LOW`, `1=MEDIUM`, `2=HIGH`, `3=CRITICAL` (Multiclass Classification).
* `target_flash_flood_flag` — Binary flash-flood occurrence flag (`0` or `1`).
* `target_cloudburst_flag` — Binary cloudburst event flag (`0` or `1`).
* `target_waterlogging_flag` — Binary street-level ponding flag (`0` or `1`).

> **💡 Train / Test Recommendation**: For ML model training, use Timesteps $T_1 - T_{48}$ (first 48 hours) as the Training set, and held-out Timesteps $T_{49} - T_{72}$ (last 24 hours) as the Validation/Test set.

---

## 🗃️ Data Provenance & Calibration Transparency

* **DEM Elevation**: Sourced from real **NASA SRTM 30m (1-arc-sec)** & Copernicus GLO-30 via Open Elevation.
* **Meteorological Series**: Sourced from real **ECMWF ERA5-Land Reanalysis (9km hourly)**.
* **Calibrated Deluge Dynamics**: The storm progression represents a calibrated high-density cloudburst scenario modeled on real Mumbai heavy-monsoon physics (Mithi River basin drainage capacity, Mahim Bay spring tidal lock, and low-lying underpass depressions).

---

## 📁 Repository Structure

```
VARUNA/
├── data/
│   ├── raw/                               # Pure untouched raw datasets for ML model team
│   │   ├── dem/
│   │   │   ├── mumbai_srtm_dem_raw.csv    # Raw SRTM elevation points (cell_id, lat, lon, elevation_m)
│   │   │   └── mumbai_srtm_dem_raw.json
│   │   ├── rainfall/
│   │   │   ├── mumbai_hourly_rainfall_raw.csv # Raw hourly precipitation, temp, humidity, pressure, wind
│   │   │   └── mumbai_hourly_rainfall_raw.json
│   │   ├── moisture/
│   │   │   ├── mumbai_soil_moisture_raw.csv   # Raw volumetric soil moisture (0-7cm & 7-28cm layers)
│   │   │   └── mumbai_soil_moisture_raw.json
│   │   └── mumbai_pilot_metadata_raw.json # Bounding box and hotspot coordinates
│   ├── dem/                               # Processed SRTM elevation raster
│   ├── rainfall/                          # Deluge time-series
│   ├── moisture_satellite/                # Satellite soil moisture proxy
│   └── feature_grid_timeseries.json       # 4D Fused model-ready feature grid + labels
├── data_pipeline/
│   ├── raw_dataset_downloader.py          # Pure raw dataset ingestion (no preprocessing)
│   ├── dataset_downloader.py              # Sourced & calibrated case study generator
│   ├── preprocessor.py                    # Multi-modal fusion & target label generator
│   └── seed_db.py                         # Schema migration & initial seed script
├── backend/
│   ├── app/
│   │   ├── main.py                        # FastAPI application entry point
│   │   ├── core/                          # Config and SQLite/PostgreSQL connection
│   │   ├── models/                        # SQLAlchemy ORM (Region, Feature, RiskEvent, Alert)
│   │   ├── schemas/                       # Pydantic validation schemas
│   │   └── api/v1/endpoints/
│   │       ├── raw_data.py                # Raw datasets endpoints for Prediction Model team
│   │       ├── regions.py                 # Region spatial metadata
│   │       ├── features.py                # Hydro-meteorological feature feeds
│   │       ├── events.py                  # Hazard events
│   │       ├── alerts.py                  # Explainable alerts and trust scoring
│   │       └── replay.py                  # Historical deluge replay controls
│   ├── test_api.py                        # Integration test suite
│   ├── run.py                             # Server launcher
│   └── requirements.txt
└── README.md
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Environment Setup
```bash
# Clone the repository
git clone https://github.com/harisolanki1804/sih_26077.git
cd sih_26077
git checkout api-and-data-fetching

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt
```

### 2. Download Raw Datasets & Initialize Database
```bash
# Ingest raw NASA SRTM & ECMWF datasets (100% free, no API keys needed)
python data_pipeline/raw_dataset_downloader.py

# Generate fused model-ready feature grid and labels
python data_pipeline/preprocessor.py

# Initialize SQLite/PostgreSQL schema and seed pilot region
python data_pipeline/seed_db.py
```

### 3. Run FastAPI Backend Server
```bash
python backend/run.py
```
* **Interactive OpenAPI Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
* **Health Check & Diagnostics**: [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health)

### 4. Run Automated Tests
```bash
python backend/test_api.py
```

---

## 🌐 REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/health` | System health check and database diagnostics |
| `GET` | `/api/v1/data/raw/catalog` | Catalog of raw CSV/JSON files for the ML team |
| `GET` | `/api/v1/data/raw/dem` | Raw DEM elevation points |
| `GET` | `/api/v1/data/raw/rainfall` | Raw hourly rainfall & wind kinematics time series |
| `GET` | `/api/v1/regions` | List registered pilot regions |
| `GET` | `/api/v1/regions/{code}/hotspots` | Critical urban flood hotspots |
| `GET` | `/api/v1/features/latest` | Latest 90-cell spatial grid snapshot |
| `GET` | `/api/v1/events` | List tracked hazard events |
| `GET` | `/api/v1/alerts` | Query active/historical alerts with severity filters |
| `GET` | `/api/v1/alerts/{id}/explain` | **Explainability & Trust Breakdown** (Decomposition & Playbook) |
| `POST` | `/api/v1/alerts/{id}/acknowledge` | Emergency operator alert acknowledgement |
| `GET` | `/api/v1/replay/status` | Current storm simulation status |
| `POST` | `/api/v1/replay/step` | Advance storm simulation 1 timestep or jump to peak |
| `POST` | `/api/v1/replay/reset` | Reset simulation to pre-storm baseline |

---

## 👥 Team Roles & Responsibilities

| Role / Owner | Scope of Ownership |
|---|---|
| **Rudra** | Dataset Sourcing & Ingestion Pipeline; Multi-Source Grid Alignment; FastAPI Backend + Database (SQLite/PostgreSQL); Events & Alerts API; Replay Engine; System Architecture |
| **Hari & Arya (Model Team)** | Feature Engineering, Learned Fusion Layer, ML Training, Prediction Models (Flash Flood / Cloudburst Classifier), Inundation Routing |
| **Srushti (Frontend Team)** | React / Leaflet Command Dashboard, Live Alert Feed, Map Overlay, Operator Reasoning Panel |
| **Himanshu & Shubham** | PPT / Documentation & Pitch Strategy |

---

## 📄 License
Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.
