"""
VARUNA Multi-Hazard Risk Model & DEM Inundation Depth Heuristic
----------------------------------------------------------------
Provides:
1. Additive multi-hazard risk score calculation (0 - 100) decomposed into transparent factors:
   - Rainfall Intensity & Accumulation (0 - 40 pts)
   - Soil Saturation & Antecedent Moisture (0 - 25 pts)
   - Topography & DEM Depression Vulnerability (0 - 20 pts)
   - Atmospheric Convective Instability / CAPE (0 - 15 pts)
2. Simplified DEM-based inundation depth heuristic (in centimeters).
"""

import math
from typing import Dict, Any, Tuple


class RiskModelService:
    @staticmethod
    def calculate_cell_risk(
        rain_1h: float,
        rain_3h: float,
        rain_24h: float,
        soil_moist_pct: float,
        elevation_m: float,
        slope_deg: float,
        cape_jkg: float,
        drainage_capacity_mm_hr: float = 25.0,
        is_tide_locked: bool = False,
        tide_height_m: float = 2.5,
        drainage_outfall_dist_m: float = 3000.0,
        is_depression: bool = False
    ) -> Dict[str, Any]:
        """
        Computes additive explainable risk score and flood depth estimate for a single grid cell.
        """

        # 1. Rainfall Score Contribution (Max 40.0)
        # 1h rain intensity up to 75mm/hr gives up to 25 pts
        score_1h = min(25.0, (rain_1h / 75.0) * 25.0)
        # 3h and 24h cumulative loading gives up to 15 pts
        score_accum = min(15.0, (rain_3h / 150.0) * 10.0 + (rain_24h / 300.0) * 5.0)
        rainfall_score = round(min(40.0, score_1h + score_accum), 2)

        # 2. Soil Saturation Contribution (Max 25.0)
        # Saturation above 50% ramps up runoff generation exponentially
        if soil_moist_pct <= 40.0:
            soil_score = (soil_moist_pct / 40.0) * 5.0
        elif soil_moist_pct <= 75.0:
            soil_score = 5.0 + ((soil_moist_pct - 40.0) / 35.0) * 10.0
        else:
            soil_score = 15.0 + ((soil_moist_pct - 75.0) / 25.0) * 10.0
        soil_score = round(min(25.0, max(0.0, soil_score)), 2)

        # 3. Topography & DEM Depression Contribution (Max 20.0)
        # Low elevations (< 5.0m) and flat slopes (< 1.0 deg) in urban bowls
        elev_penalty = max(0.0, (12.0 - min(12.0, elevation_m)) / 12.0) * 10.0
        slope_penalty = max(0.0, (3.0 - min(3.0, slope_deg)) / 3.0) * 5.0
        depression_penalty = 5.0 if is_depression else 0.0
        topo_score = round(min(20.0, elev_penalty + slope_penalty + depression_penalty), 2)

        # 4. Atmospheric Instability (CAPE) Contribution (Max 15.0)
        # CAPE > 2500 J/kg indicates violent updrafts & cloudburst potential
        if cape_jkg < 1000:
            instab_score = (cape_jkg / 1000.0) * 3.0
        elif cape_jkg < 2500:
            instab_score = 3.0 + ((cape_jkg - 1000.0) / 1500.0) * 7.0
        else:
            instab_score = 10.0 + min(5.0, ((cape_jkg - 2500.0) / 1500.0) * 5.0)
        instab_score = round(min(15.0, max(0.0, instab_score)), 2)

        # Total Additive Risk Score
        total_risk = round(min(100.0, rainfall_score + soil_score + topo_score + instab_score), 1)

        # Severity Classification
        if total_risk >= 80.0:
            severity = "CRITICAL"
        elif total_risk >= 60.0:
            severity = "HIGH"
        elif total_risk >= 35.0:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        # 5. DEM-based Flood Depth Heuristic (cm)
        # Net un-drained rainfall volume
        net_excess_rain = max(0.0, rain_1h - (drainage_capacity_mm_hr * 0.7))
        
        # Depression retention multiplier
        retention_mult = 1.0 + (max(0.0, 7.0 - elevation_m) / 7.0) * (2.0 if is_depression else 1.2)

        # Tidal lock backwater multiplier
        tidal_mult = 1.0
        if is_tide_locked and drainage_outfall_dist_m < 8000.0:
            tidal_mult = 1.0 + (tide_height_m - 4.2) * 0.8 * (1.0 - (drainage_outfall_dist_m / 8000.0))

        # Soil saturation runoff transfer
        sat_trans = 0.6 + 0.4 * (soil_moist_pct / 100.0)

        # Inundation depth in centimeters
        if net_excess_rain > 0.5:
            # 1 mm excess rain = approx 0.1 cm baseline depth
            depth_cm = round((net_excess_rain * 0.15) * retention_mult * tidal_mult * sat_trans + (rain_3h * 0.05), 1)
        else:
            depth_cm = 0.0

        return {
            "total_risk_score": total_risk,
            "severity": severity,
            "rainfall_score_contrib": rainfall_score,
            "soil_saturation_score_contrib": soil_score,
            "topography_score_contrib": topo_score,
            "atmospheric_instability_score_contrib": instab_score,
            "flood_depth_estimate_cm": depth_cm,
            "is_cloudburst": bool(rain_1h >= 65.0 or (rain_1h >= 45.0 and cape_jkg >= 2400)),
            "is_waterlogging": bool(depth_cm >= 15.0)
        }


risk_model_service = RiskModelService()
