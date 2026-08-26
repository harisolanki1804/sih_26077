"""
VARUNA Database Seed Script
----------------------------
Populates SQLite/PostgreSQL database with:
1. Pilot Region (Mumbai Urban Flood Corridor)
2. Historical Deluge Risk Event
3. Initial Feature Grid & Baseline Alert State
"""

import os
import sys
import json
import logging
from datetime import datetime

# Set backend path
BACKEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend")
sys.path.insert(0, BACKEND_DIR)

from app.core.database import SessionLocal, init_db
from app.core.config import settings
from app.models.region import Region
from app.models.risk_event import RiskEvent, SeverityLevel, EventStatus
from app.models.feature import Feature
from app.models.alert import Alert
from app.services.replay_service import replay_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("VARUNA.Seed")


def seed():
    logger.info("Initializing database tables...")
    init_db()
    db = SessionLocal()

    try:
        # 1. Load Metadata & Seed Pilot Region
        meta_path = os.path.join(settings.DATA_DIR, "pilot_mumbai_metadata.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        region = db.query(Region).filter(Region.code == meta["region_code"]).first()
        if not region:
            logger.info(f"Seeding Pilot Region: {meta['name']} ({meta['region_code']})...")
            region = Region(
                code=meta["region_code"],
                name=meta["name"],
                description=meta["description"],
                bbox_lat_min=meta["bbox"]["lat_min"],
                bbox_lat_max=meta["bbox"]["lat_max"],
                bbox_lon_min=meta["bbox"]["lon_min"],
                bbox_lon_max=meta["bbox"]["lon_max"],
                center_lat=meta["center"]["lat"],
                center_lon=meta["center"]["lon"],
                area_sqkm=meta["area_sqkm"],
                avg_drainage_capacity_mm_hr=meta["avg_drainage_capacity_mm_hr"],
                high_tide_threshold_m=meta["high_tide_threshold_m"],
                metadata_json=meta
            )
            db.add(region)
            db.commit()
            db.refresh(region)
            logger.info(f"Pilot region seeded with ID: {region.id}")
        else:
            logger.info(f"Pilot Region {region.code} already exists.")

        # 2. Seed Baseline Risk Event
        event_code = settings.DEFAULT_EVENT_CODE
        event = db.query(RiskEvent).filter(RiskEvent.event_code == event_code).first()
        if not event:
            logger.info(f"Seeding Baseline Risk Event: {event_code}...")
            event = RiskEvent(
                region_id=region.id,
                event_code=event_code,
                title="Historical Mumbai Monsoon Cloudburst Case Study",
                description="72-hour extreme precipitation deluge replay across Mithi River catchment and central subways.",
                start_time=datetime.fromisoformat("2024-07-26T00:00:00+00:00"),
                severity_level=SeverityLevel.LOW.value,
                status=EventStatus.MONITORING.value,
                confidence_score=92.0
            )
            db.add(event)
            db.commit()
            logger.info("Risk Event seeded successfully.")

        # 3. Seed Initial Feature Grid Snapshots from feature_grid_timeseries.json
        features_path = os.path.join(settings.DATA_DIR, "feature_grid_timeseries.json")
        if os.path.exists(features_path):
            with open(features_path, "r", encoding="utf-8") as f:
                feat_data = json.load(f)
            
            existing_feats = db.query(Feature).filter(Feature.region_id == region.id).count()
            if existing_feats == 0:
                logger.info("Seeding initial timestep features to database...")
                initial_ts = feat_data["timesteps"][0]
                ts_dt = datetime.fromisoformat(initial_ts["timestamp"].replace("Z", "+00:00"))
                
                feat_records = []
                for cell in initial_ts["features"]:
                    feat_records.append(
                        Feature(
                            region_id=region.id,
                            timestamp=ts_dt,
                            cell_index=cell["cell_index"],
                            cell_lat=cell["lat"],
                            cell_lon=cell["lon"],
                            rainfall_1h_mm=cell["rainfall_1h_mm"],
                            rainfall_3h_mm=cell["rainfall_3h_mm"],
                            rainfall_6h_mm=cell["rainfall_6h_mm"],
                            rainfall_24h_mm=cell["rainfall_24h_mm"],
                            soil_moisture_pct=cell["soil_moisture_pct"],
                            soil_saturation_factor=cell["soil_saturation_factor"],
                            cape_instability_jkg=cell["cape_instability_jkg"],
                            elevation_m=cell["elevation_m"],
                            slope_deg=cell["slope_deg"],
                            runoff_coefficient=cell["runoff_coefficient"],
                            effective_runoff_mm_hr=cell["effective_runoff_mm_hr"],
                            drainage_outfall_dist_m=cell["drainage_outfall_dist_m"],
                            is_depression_bowl=cell["is_depression_bowl"],
                            tide_height_m=initial_ts.get("tide_height_m", 2.5),
                            is_high_tide_locked=initial_ts.get("is_high_tide_locked", False)
                        )
                    )
                db.bulk_save_objects(feat_records)
                db.commit()
                logger.info(f"Seeded {len(feat_records)} initial feature grid cells.")

        # 4. Advance replay by a couple of steps or initialize state
        logger.info("Initializing replay simulation state at timestep 1...")
        replay_engine.reset(db)
        logger.info("Database seeding completed successfully.")

    except Exception as e:
        logger.error(f"Error during seeding: {e}")
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    seed()
