import os
import json
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.config import settings
from app.models.region import Region
from app.schemas.region import RegionRead, RegionCreate, HotspotInfo

router = APIRouter()


@router.get("", response_model=List[RegionRead], summary="List All Monitored Pilot Regions")
def list_regions(db: Session = Depends(get_db)):
    """Returns all geographic regions registered in VARUNA."""
    return db.query(Region).all()


@router.get("/{region_id_or_code}", response_model=RegionRead, summary="Get Pilot Region Details")
def get_region(region_id_or_code: str, db: Session = Depends(get_db)):
    """Fetches single region by UUID or region code (e.g. IN-MH-BOM-01)."""
    region = (
        db.query(Region)
        .filter((Region.id == region_id_or_code) | (Region.code == region_id_or_code))
        .first()
    )
    if not region:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Region not found.")
    return region


@router.get("/{region_id_or_code}/hotspots", response_model=List[HotspotInfo], summary="Get Regional Flood Hotspots")
def get_region_hotspots(region_id_or_code: str, db: Session = Depends(get_db)):
    """Returns critical urban infrastructure hotspots (underpasses, basins, transit hubs) for the pilot region."""
    meta_path = os.path.join(settings.DATA_DIR, "pilot_mumbai_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return [HotspotInfo(**h) for h in data.get("critical_hotspots", [])]
    return []


@router.post("", response_model=RegionRead, status_code=status.HTTP_201_CREATED, summary="Create New Pilot Region")
def create_region(payload: RegionCreate, db: Session = Depends(get_db)):
    existing = db.query(Region).filter(Region.code == payload.code).first()
    if existing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Region with this code already exists.")
    region = Region(**payload.model_dump())
    db.add(region)
    db.commit()
    db.refresh(region)
    return region
