# VARUNA — AI-Driven Hyperlocal Flood Digital Twin

## What is VARUNA?

VARUNA is an AI-powered early warning system that predicts **where** flash floods will hit, **how deep** the water will be on each street, and **how confident** the prediction is — all **2-6 hours in advance**.

### Core Intelligence
**Spatiotemporal Deep Learning + Physics-Informed AI + Multi-Source Data Fusion + Hyperlocal Risk Mapping + Explainable Forecasting**

### Key Features
- **9 AI/ML Modules** — all trained with PyTorch (214K+ parameters)
- **Real satellite data** — INSAT-3D via MOSDAC API
- **72-hour Digital Twin replay** — rewind any storm, replay step-by-step
- **What-If Simulator** — "what if rainfall doubles?"
- **Works offline** — runs on a laptop, no internet needed during floods

---

## Quick Setup (20 minutes)

### 1. Clone & Install

```bash
git clone <repo-url>
cd sih_26077

# Create virtual environment
python -m venv .venv
.venv/Scripts/activate  # Windows
# source .venv/bin/activate  # Mac/Linux

# Install backend dependencies
cd backend
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install shap

# Install frontend dependencies
cd ../frontend
npm install
```

### 2. Configure MOSDAC Credentials

Create/edit `backend/.env`:
```
MOSDAC_USERNAME=your_username
MOSDAC_PASSWORD=your_password
```

**How to get MOSDAC credentials:**
1. Go to https://mosdac.gov.in
2. Click **Register** → create free account
3. Verify email → login
4. Use those credentials in `.env`

### 3. Start the System

**Terminal 1 — Backend:**
```bash
cd backend
.venv/Scripts/python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

**Terminal 2 — Frontend:**
```bash
cd frontend
npm run dev
```

**Open:** http://localhost:5173

---

## How to Test Each Module

### Module 1: Storm Cell Detection & Tracking
- **What it does:** Detects convective cloud clusters and tracks their motion
- **How to test:** Click "Mumbai Replay" → click any cell on the map
- **Look for:** `inference_mode: trained_neural_network` in API response
- **API:** `GET /api/v1/ai/storm-cells/36`

### Module 2: Risk Heatmap Generation
- **What it does:** Classifies every pixel into risk zones (thunderstorm/cloudburst/flash flood)
- **How to test:** Map shows colored cells — red = critical, orange = high, yellow = medium, green = low
- **API:** `GET /api/v1/ai/risk-heatmap/36`

### Module 3: Spatiotemporal Nowcasting
- **What it does:** Forecasts rainfall 2-6 hours ahead using ConvLSTM + Transformer
- **How to test:** Check "What's Coming" panel — bars should vary across timesteps
- **API:** `GET /api/v1/ai/nowcast/36`

### Module 4: Multi-Hazard Prediction
- **What it does:** Simultaneously predicts thunderstorm, cloudburst, and flash flood risk
- **How to test:** Check Overview tab for three hazard probabilities
- **API:** `GET /api/v1/ai/multi-hazard/36`

### Module 5: Cross-Attention Fusion
- **What it does:** Aligns satellite, reanalysis, and DEM data at each grid point
- **How to test:** Check fused features in API response
- **API:** `GET /api/v1/ai/fused-features/36`

### Module 6: Urban Flood Depth Estimation
- **What it does:** Predicts exact water depth (cm) per street segment using GNN
- **How to test:** Click cells with ⚠ markers — shows overflow nodes
- **API:** `GET /api/v1/ai/flood-depth/36`

### Module 7: Forecast Trust Scoring
- **What it does:** Flags when current pattern resembles past high-error cases
- **How to test:** Check Trust tab — shows High/Medium/Low confidence with conformal prediction
- **API:** `GET /api/v1/ai/trust-score/36`

### Module 8: Explainable AI (XAI)
- **What it does:** Shows top meteorological drivers behind every alert
- **How to test:** Check XAI tab — shows rainfall, CAPE, CTT, soil moisture as top drivers
- **API:** `GET /api/v1/ai/xai/36`

### Module 9: Crowd-Report Validation (NLP)
- **What it does:** Classifies citizen flood reports as confirming/denying a predicted flood zone
- **How to test:** Type "heavy rain flooding help" — should classify as CONFIRMS_FLOOD_ZONE
- **API:** `POST /api/v1/ai/crowd-report`

### Replay Engine (All Modules Together)
- **▶ Play** — auto-advances through 72 timesteps (1.5s each)
- **◀/▶** — step-by-step control
- **🔥 Peak** — jump to timestep 36 (maximum storm)
- Each step runs the full 9-module AI pipeline

---

## Training AI Models

All models are pre-trained. To retrain:

```bash
cd backend
.venv/Scripts/python.exe -c "
from app.services.ai.trainer import VARUNATrainer
trainer = VARUNATrainer()
trainer.train_all(epochs=100, lr=0.001)
"
```

**Training time:** ~2 minutes on CPU

**What gets trained:**
- Multi-Hazard Predictor (145K params)
- Flood Depth Estimator (10K params)
- Trust Scorer (24K params)
- XAI Attention Layer (9K params)
- Storm Cell Detector (3.5K params)
- Crowd NLP Classifier (1.3K params)
- Spatiotemporal Nowcaster (3.4M params)

---

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/v1/health` | System health check |
| `GET /api/v1/ai/inference/{ts}` | Full 9-module pipeline |
| `GET /api/v1/ai/storm-cells/{ts}` | Storm cell detection |
| `GET /api/v1/ai/risk-heatmap/{ts}` | Risk heatmap |
| `GET /api/v1/ai/nowcast/{ts}` | 6-hour nowcast |
| `GET /api/v1/ai/multi-hazard/{ts}` | Multi-hazard prediction |
| `GET /api/v1/ai/fused-features/{ts}` | Cross-source fusion |
| `GET /api/v1/ai/flood-depth/{ts}` | Flood depth estimation |
| `GET /api/v1/ai/trust-score/{ts}` | Trust scoring |
| `GET /api/v1/ai/xai/{ts}` | XAI explanation |
| `POST /api/v1/ai/crowd-report` | NLP classification |
| `GET /api/v1/ai/model-status` | Model training status |
| `GET /api/v1/replay/status` | Replay engine state |
| `POST /api/v1/replay/step?step_to={ts}` | Jump to timestep |
| `GET /api/v1/alerts` | Active alerts |
| `GET /api/v1/realtime/india` | Live India grid |
| `GET /api/v1/innovations/satellite/latest` | INSAT-3D satellite data |
| `GET /api/v1/innovations/physics/risk/{ts}` | Physics-informed risk |
| `GET /api/v1/innovations/evacuation/routes/{cell}` | Evacuation routes |
| `POST /api/v1/innovations/scenario/simulate` | What-If simulator |
| `POST /api/v1/innovations/chatbot/ask` | NLP chatbot |

---

## Project Structure

```
sih_26077/
├── backend/
│   ├── app/
│   │   ├── api/v1/endpoints/     # 47 REST API endpoints
│   │   ├── services/
│   │   │   ├── ai/               # 9 AI/ML modules
│   │   │   │   ├── model_architectures.py  # PyTorch models
│   │   │   │   ├── inference_engine.py      # Unified inference
│   │   │   │   ├── trainer.py               # Training pipeline
│   │   │   │   └── data_loader.py           # Data preparation
│   │   │   ├── satellite/        # MOSDAC INSAT-3D fetcher
│   │   │   ├── data/             # IMDAA reanalysis
│   │   │   ├── physics/          # PINN risk model
│   │   │   ├── realtime/         # Open-Meteo live data
│   │   │   └── scenario/         # What-If engine
│   │   └── schemas/              # Pydantic models
│   ├── models/checkpoints/       # 7 trained PyTorch models
│   └── evaluation/               # Evaluation metrics
├── frontend/
│   ├── src/
│   │   ├── App.jsx               # Main layout
│   │   ├── components/
│   │   │   ├── GeoMap.jsx        # Interactive map
│   │   │   ├── MapView.jsx       # Map container
│   │   │   ├── LeftPanel.jsx     # KPI cards
│   │   │   └── IntelligencePanel.jsx  # Right panel
│   │   └── utils/                # API client, helpers
│   └── package.json
├── data/
│   ├── feature_grid_timeseries.json  # 72-hour Mumbai dataset
│   ├── dem/                      # SRTM elevation data
│   └── raw/                      # Raw weather data
└── RUN_GUIDE.md                  # This file
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ECONNREFUSED ::1:8000` | Start backend first: `cd backend && uvicorn app.main:app --port 8000` |
| `MOSDAC auth failed` | Check username/password in `backend/.env` |
| `Open-Meteo 400 error` | Already fixed — uses valid parameters only |
| `Open-Meteo 429 error` | Rate limited — wait 5 minutes |
| `Model not loading` | Run training command above |
| `Frontend blank page` | Check browser console for errors |
| `Map not showing` | Ensure Leaflet CSS is loaded in `index.html` |

---

## Data Sources

| Source | Type | Purpose |
|--------|------|---------|
| Open-Meteo | Free API (no key) | Real-time weather |
| MOSDAC/INSAT-3D | Free (credentials) | Satellite data |
| SRTM DEM | Static JSON | Elevation, slope |
| IMDAA | Free (academic) | Reanalysis data |

---

## Tech Stack Summary

**AI/ML:** Python, PyTorch, ConvLSTM, Transformer, SHAP  
**Backend:** FastAPI, SQLAlchemy, SQLite  
**Frontend:** React.js, Leaflet, Vite  
**Data:** Open-Meteo, MOSDAC/INSAT-3D, SRTM DEM  
**Physics:** SCS-CN, Manning's Equation, Shallow Water Eq  
