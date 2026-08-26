import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, DateTime, ForeignKey, Integer, JSON, Boolean, Index
from sqlalchemy.orm import relationship
from app.core.database import Base


class Feature(Base):
    __tablename__ = "features"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    region_id = Column(String(36), ForeignKey("regions.id", ondelete="CASCADE"), index=True, nullable=False)
    timestamp = Column(DateTime, index=True, nullable=False)
    cell_index = Column(Integer, nullable=False)
    cell_lat = Column(Float, nullable=False)
    cell_lon = Column(Float, nullable=False)

    # Meteorological / Hydrological Features
    rainfall_1h_mm = Column(Float, default=0.0)
    rainfall_3h_mm = Column(Float, default=0.0)
    rainfall_6h_mm = Column(Float, default=0.0)
    rainfall_24h_mm = Column(Float, default=0.0)
    
    # Soil & Atmospheric Proxies
    soil_moisture_pct = Column(Float, default=50.0)
    soil_saturation_factor = Column(Float, default=0.5)
    cape_instability_jkg = Column(Float, default=800.0)
    
    # Topographical / Terrain Features
    elevation_m = Column(Float, nullable=False)
    slope_deg = Column(Float, default=1.0)
    runoff_coefficient = Column(Float, default=0.8)
    effective_runoff_mm_hr = Column(Float, default=0.0)
    drainage_outfall_dist_m = Column(Float, default=3000.0)
    is_depression_bowl = Column(Boolean, default=False)
    
    # Tidal Condition
    tide_height_m = Column(Float, default=2.5)
    is_high_tide_locked = Column(Boolean, default=False)

    raw_payload_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    region = relationship("Region", back_populates="features")

    __table_args__ = (
        Index("ix_features_region_time", "region_id", "timestamp"),
        Index("ix_features_cell_time", "cell_lat", "cell_lon", "timestamp"),
    )
