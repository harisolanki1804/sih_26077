# VARUNA — AI-Driven Hyperlocal Weather, Flash-Flood & Confidence-Aware Early Warning System

**Smart India Hackathon | Problem Statement: SIH26077**
*(AI-Driven Hyper-Local Early Warning System for Severe Weather Nowcasting)*

## Overview

VARUNA is an AI-powered early warning system that predicts severe thunderstorms, cloudbursts, and flash floods 2–6 hours in advance, then translates that prediction into street-level flood depth for urban areas — with an explainable confidence score attached to every alert.

Traditional weather models are too slow and too coarse to catch fast-developing local storms, and rainfall forecasts alone don't tell anyone which street will actually flood. VARUNA fuses satellite data, atmospheric reanalysis data, and terrain data into one pipeline to close that gap.

## Key Features

- **Multi-hazard nowcasting** — predicts thunderstorm, cloudburst, and flash-flood risk from a single fused model
- **Street-level flood depth estimation** — combines rainfall prediction with elevation/terrain data
- **Explainable Risk Engine** — every alert shows *why* it fired (top contributing factors), not just a score
- **Forecast Trust Score** — a confidence label (High/Medium/Low) showing how much to trust each prediction
- **Live command dashboard** — map view, real-time alert feed, and a detailed breakdown panel for operators

## Tech Stack

| Layer | Technology |
|---|---|
| AI / Modeling | Python, scikit-learn / XGBoost, SHAP (explainability) |
| Backend | FastAPI, PostgreSQL |
| Frontend | React, Leaflet |
| Data Sources | INSAT satellite data, IMDAA reanalysis, DEM (SRTM/CartoDEM) |

## Team

| Role | Members |
|---|---|
| Main Model Development | Hari, Arya |
| Data & APIs | Rudra |
| Frontend | Srushti |
| PPT / Documentation | Himanshu, Shubham |

## Project Status

🚧 In development for Smart India Hackathon.

## Getting Started

```bash
# Clone the repo
git clone <repo-url>
cd varuna

# Backend
cd backend && pip install -r requirements.txt && uvicorn app.main:app --reload

# Frontend
cd frontend && npm install && npm run dev

# Model
cd model && pip install -r requirements.txt && python train.py
```

## License

TBD
