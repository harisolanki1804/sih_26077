from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, ConfigDict


class FeatureBase(BaseModel):
    cell_index: int
    cell_lat: float
    cell_lon: float
    rainfall_1h_mm: float
    rainfall_3h_mm: float
    rainfall_6h_mm: float
    rainfall_24h_mm: float
    soil_moisture_pct: float
    soil_saturation_factor: float
    cape_instability_jkg: float
    elevation_m: float
    slope_deg: float
    runoff_coefficient: float
    effective_runoff_mm_hr: float
    drainage_outfall_dist_m: float
    is_depression_bowl: bool
    tide_height_m: float
    is_high_tide_locked: bool
    raw_payload_json: Optional[Dict[str, Any]] = None


class FeatureCreate(FeatureBase):
    region_id: str
    timestamp: datetime


class FeatureRead(FeatureBase):
    id: str
    region_id: str
    timestamp: datetime
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class FeatureGridSnapshot(BaseModel):
    region_id: str
    region_code: str
    timestamp: datetime
    timestep_id: Optional[int] = None
    total_cells: int
    avg_rainfall_1h_mm: float
    max_rainfall_1h_mm: float
    tide_height_m: float
    is_high_tide_locked: bool
    cells: List[FeatureRead]
