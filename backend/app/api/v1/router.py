from fastapi import APIRouter
from app.api.v1.endpoints import health, regions, features, events, alerts, replay, raw_data

api_router = APIRouter()

api_router.include_router(health.router, tags=["Health & Diagnostics"])
api_router.include_router(raw_data.router, prefix="/data/raw", tags=["Raw Datasets for Model Team"])
api_router.include_router(regions.router, prefix="/regions", tags=["Regions & Pilot Areas"])
api_router.include_router(features.router, prefix="/features", tags=["Feature Grids & Hydro Data"])
api_router.include_router(events.router, prefix="/events", tags=["Hazard Events"])
api_router.include_router(alerts.router, prefix="/alerts", tags=["Explainable Alerts & Trust"])
api_router.include_router(replay.router, prefix="/replay", tags=["Historical Deluge Replay Engine"])
