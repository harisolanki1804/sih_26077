"""
VARUNA Unified Inference Engine
================================
Chains all 9 AI/ML modules in the replay pipeline to produce
comprehensive, explainable multi-hazard predictions at each timestep.
"""

import os
import json
import math
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from app.core.config import settings

logger = logging.getLogger("VARUNA.AI.InferenceEngine")


class VARUNAInferenceEngine:
    """
    Central inference coordinator that runs all 9 AI modules on each
    timestep of the replay simulation.

    When PyTorch is available and trained checkpoints exist,
    uses the actual neural network models for inference.
    Otherwise, falls back to physics-informed heuristic inference.
    """

    def __init__(self, model_dir: Optional[str] = None):
        # Navigate from backend/app/services/ai/ to project root/models/checkpoints
        self.model_dir = model_dir or os.path.abspath(os.path.join(
            settings.DATA_DIR, "..", "models", "checkpoints"
        ))
        self._torch_models = None
        self._use_torch = False
        self._scaler = None

        # Live network fetches (MOSDAC/IMDAA) are OFF by default so the replay
        # engine is deterministic and never blocks on slow external APIs.
        # Set VARUNA_LIVE_FETCH=1 to allow real-time network refresh.
        self.allow_network = os.getenv("VARUNA_LIVE_FETCH", "0") == "1"

        # Storm-cell tracking state across frames (deterministic association)
        self._storm_tracks = {}  # cell_index -> dict(last_confidence, last_timestamp, lifetime)
        self._sat_bg_started = False  # one background MOSDAC fetch per process

        # Try loading PyTorch models
        try:
            import torch
            self._torch_available = True
            self._load_models_if_available()
        except ImportError:
            self._torch_available = False
            logger.info("PyTorch not available — using heuristic inference fallback.")

    def _load_models_if_available(self):
        """Attempt to load saved model checkpoints and feature scaler."""
        if not self._torch_available or not os.path.exists(self.model_dir):
            logger.info(f"Model dir not found: {self.model_dir}")
            return

        try:
            import torch
            import pickle
            from app.services.ai.model_architectures import MultiHazardPredictor

            # Load feature scaler
            scaler_path = os.path.join(self.model_dir, "scaler.pkl")
            if os.path.exists(scaler_path):
                with open(scaler_path, "rb") as f:
                    self._scaler = pickle.load(f)
                logger.info("Loaded feature scaler")

            # Load multi-hazard predictor (the main trained model)
            self._torch_models = {}
            n_features = 22  # Updated from 14 to include IWV, CIN, shear, etc.
            mh_model = MultiHazardPredictor(in_features=n_features)
            mh_ckpt = os.path.join(self.model_dir, "multi_hazard_predictor.pt")
            if os.path.exists(mh_ckpt):
                mh_model.load_state_dict(torch.load(mh_ckpt, map_location="cpu", weights_only=True))
                mh_model.eval()
                self._torch_models["multi_hazard"] = mh_model
                logger.info("Loaded trained multi-hazard predictor")

            # Load flood depth estimator
            depth_ckpt = os.path.join(self.model_dir, "flood_depth_estimator.pt")
            if os.path.exists(depth_ckpt):
                import torch.nn as nn
                depth_model = nn.Sequential(
                    nn.Linear(n_features, 128), nn.ReLU(), nn.Dropout(0.2),
                    nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.ReLU(),
                )
                depth_model.load_state_dict(torch.load(depth_ckpt, map_location="cpu", weights_only=True))
                depth_model.eval()
                self._torch_models["flood_depth"] = depth_model
                logger.info("Loaded trained flood depth estimator")

            # Load trust scorer
            trust_ckpt = os.path.join(self.model_dir, "forecast_trust_scorer.pt")
            if os.path.exists(trust_ckpt):
                from app.services.ai.model_architectures import ForecastTrustScorer
                trust_model = ForecastTrustScorer(input_dim=n_features)
                trust_model.load_state_dict(torch.load(trust_ckpt, map_location="cpu", weights_only=True))
                trust_model.eval()
                self._torch_models["trust_scorer"] = trust_model
                logger.info("Loaded trained trust scorer")

            # Load XAI attention layer
            xai_ckpt = os.path.join(self.model_dir, "xai_attention_layer.pt")
            if os.path.exists(xai_ckpt):
                from app.services.ai.model_architectures import XAIAttentionLayer
                xai_model = XAIAttentionLayer(num_features=n_features)
                xai_model.load_state_dict(torch.load(xai_ckpt, map_location="cpu", weights_only=True))
                xai_model.eval()
                self._torch_models["xai"] = xai_model
                logger.info("Loaded trained XAI attention layer")

            # Load storm cell detector
            storm_ckpt = os.path.join(self.model_dir, "storm_cell_detector.pt")
            if os.path.exists(storm_ckpt):
                import torch.nn as nn
                storm_model = nn.Sequential(
                    nn.Linear(n_features, 128), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
                    nn.Linear(64, 1), nn.Sigmoid(),
                )
                storm_model.load_state_dict(torch.load(storm_ckpt, map_location="cpu", weights_only=True))
                storm_model.eval()
                self._torch_models["storm_cell"] = storm_model
                logger.info("Loaded trained storm cell detector")

            # Load crowd NLP classifier
            crowd_ckpt = os.path.join(self.model_dir, "crowd_nlp_classifier.pt")
            if os.path.exists(crowd_ckpt):
                import torch.nn as nn
                from app.services.ai.data_loader import FLOOD_VOCAB
                vocab_size = max(FLOOD_VOCAB.values()) + 10
                embed_dim = 32

                class CrowdNLPModel(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
                        self.pool = nn.AdaptiveMaxPool1d(1)
                        self.classifier = nn.Sequential(
                            nn.Linear(embed_dim, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 3),
                        )
                    def forward(self, x):
                        emb = self.embedding(x).permute(0, 2, 1)
                        return self.classifier(self.pool(emb).squeeze(-1))

                crowd_model = CrowdNLPModel()
                crowd_model.load_state_dict(torch.load(crowd_ckpt, map_location="cpu", weights_only=True))
                crowd_model.eval()
                self._torch_models["crowd_nlp"] = crowd_model
                logger.info("Loaded trained crowd NLP classifier")

            # Load trained spatiotemporal nowcaster (ConvLSTM + Transformer)
            now_ckpt = os.path.join(self.model_dir, "spatiotemporal_nowcaster.pt")
            if os.path.exists(now_ckpt):
                from app.services.ai.model_architectures import SpatiotemporalNowcaster
                now_model = SpatiotemporalNowcaster(
                    in_channels=8, hidden_dim=64, forecast_horizon=6
                )
                now_model.load_state_dict(torch.load(now_ckpt, map_location="cpu", weights_only=True))
                now_model.eval()
                self._torch_models["nowcaster"] = now_model
                logger.info("Loaded trained spatiotemporal nowcaster (ConvLSTM + Transformer)")

            # Load conformal calibration
            from app.services.conformal import conformal_predictor
            conformal_predictor.load_calibration()

            if self._torch_models:
                self._use_torch = True
                logger.info(f"Loaded {len(self._torch_models)} trained models — using neural network inference")
            else:
                logger.info("No trained checkpoints found — using heuristic inference")

        except Exception as e:
            logger.warning(f"Could not load PyTorch models: {e}")
            self._use_torch = False

    # ===================================================================
    # PUBLIC INFERENCE API — called by replay service at each timestep
    # ===================================================================

    def run_full_inference(
        self,
        timestep_data: Dict[str, Any],
        all_timesteps: List[Dict[str, Any]],
        timestep_idx: int,
    ) -> Dict[str, Any]:
        """
        Run all AI modules on a single timestep and return consolidated results.

        Now includes physics-informed risk model and satellite data integration.

        Args:
            timestep_data: current timestep feature grid
            all_timesteps: full timeseries for nowcasting context
            timestep_idx: index of current timestep (0-based)

        Returns:
            dict with outputs from all modules
        """
        results = {}

        # Module 4 first: multi-task hazard prediction is the single source of
        # truth that Modules 1/2/6/7 are all derived from, so they stay consistent.
        results["multi_hazard"] = self.predict_multi_hazard(timestep_data)

        # Module 1: Storm Cell Detection
        results["storm_cells"] = self.detect_storm_cells(timestep_data)

        # Module 2: Risk Heatmap (derived from the multi-hazard model output)
        results["risk_heatmap"] = self.generate_risk_heatmap(timestep_data, results["multi_hazard"])

        # Module 3: Nowcasting (trained ConvLSTM when available)
        results["nowcast"] = self.run_nowcast(all_timesteps, timestep_idx)

        # Module 5: Cross-Attention Fusion
        results["fused_features"] = self.fuse_multi_source(timestep_data)

        # Module 6: Flood Depth Estimation
        results["flood_depth"] = self.estimate_flood_depth(timestep_data)

        # Module 7: Trust Scoring
        results["trust_score"] = self.score_forecast_trust(results)

        # Module 8: XAI Explanation
        results["xai_explanation"] = self.explain_prediction(
            timestep_data, results
        )

        # Innovation: Physics-Informed Risk Model
        results["physics_risk"] = self.compute_physics_informed_risk(timestep_data)

        # Innovation: Satellite Data Integration
        results["satellite_data"] = self.fetch_satellite_overlay(timestep_data)

        # Innovation: IMDAA Reanalysis Integration
        results["imdaa_reanalysis"] = self.fetch_imdaa_reanalysis(timestep_data)

        return results

    # ===================================================================
    # INDIVIDUAL MODULE INFERENCE
    # ===================================================================

    def _finalize_storm_cell(self, cell: Dict[str, Any], timestamp: str) -> Dict[str, Any]:
        """Attach deterministic tracking metadata to a detected storm cell.

        Motion is derived from the actual U/V wind field of the cell, area from
        the grid geometry, echo top from cloud-top temperature, and growth trend
        from comparing with the previous frame of the same cell (real tracking,
        no random numbers).
        """
        u_wind = cell.get("u_wind_ms", 0.0) or 0.0
        v_wind = cell.get("v_wind_ms", 0.0) or 0.0
        ctt = cell.get("cloud_top_temp_celsius", -20)
        cape = cell.get("cape_instability_jkg", 0)
        confidence = float(cell.get("confidence", 0.5))

        # Advection speed of the cell centre in degrees per hour.
        # 1 deg lat ~ 111 km; 1 deg lon ~ 111 km * cos(lat ~19°) ~ 105 km.
        dx = u_wind * 3600.0 / 105000.0
        dy = v_wind * 3600.0 / 111000.0

        track = self._storm_tracks.get(cell["cell_index"])
        if track is None:
            track = {"last_conf": 0.0, "last_ts": "", "lifetime": 0}

        if track["last_ts"] != timestamp:
            # New frame: same cell index detected again => lifetime continues
            track["lifetime"] += 1 if track["last_ts"] else 1
        last_conf = track["last_conf"]
        track["last_conf"] = confidence
        track["last_ts"] = timestamp
        self._storm_tracks[cell["cell_index"]] = track

        if confidence > last_conf + 0.03:
            trend = "growing"
        elif confidence < last_conf - 0.03:
            trend = "decaying"
        else:
            trend = "stable"

        return {
            "cell_index": cell["cell_index"],
            "lat": cell["lat"],
            "lon": cell["lon"],
            "confidence": round(confidence, 3),
            "storm_type": (
                "supercell" if cape > 3000 and ctt < -60
                else "convective" if cape > 2000
                else "stratiform"
            ),
            "motion_vector": {"dx_deg_hr": round(dx, 4), "dy_deg_hr": round(dy, 4)},
            "area_km2": round(4.2 * 2.7, 1),  # physical size of one pilot grid cell
            "peak_echo_top_km": round(max(8.0, 12.0 + (abs(ctt) - 40.0) * 0.3), 1),
            "track_id": f"TC-{cell['cell_index']:04d}",
            "lifetime_steps": int(track["lifetime"]),
            "trend": trend,
        }

    def detect_storm_cells(
        self, timestep_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Module 1: Detect and track convective storm cells.

        Uses the trained classifier when available; detections are then
        associated frame-to-frame deterministically using the wind field.
        """
        import torch
        import numpy as np
        features = timestep_data["features"]
        timestamp = timestep_data["timestamp"]

        # Use trained model if available
        STORM_FEATURES = [
            "rainfall_1h_mm", "rainfall_3h_mm", "rainfall_24h_mm",
            "soil_moisture_pct", "iwv_mm",
            "cape_instability_jkg", "cin_jkg", "lifted_index",
            "cloud_top_temp_celsius", "ctt_drop_rate_c_per_hr",
            "wind_speed_10m_kmh", "wind_direction_deg",
            "u_wind_ms", "v_wind_ms",
            "vertical_wind_shear_ms", "low_level_convergence",
            "elevation_m", "slope_deg",
            "drainage_outfall_dist_m", "runoff_coefficient",
            "tide_height_m", "is_high_tide_locked",
        ]

        storm_cells = []
        if self._use_torch and "storm_cell" in self._torch_models and self._scaler is not None:
            try:
                model = self._torch_models["storm_cell"]
                for cell in features:
                    feat = []
                    for col in STORM_FEATURES:
                        val = cell.get(col, 0)
                        if col == "is_high_tide_locked":
                            val = 1.0 if val else 0.0
                        feat.append(float(val))
                    X = np.array([feat], dtype=np.float32)
                    X_scaled = self._scaler.transform(X)
                    X_t = torch.FloatTensor(X_scaled)
                    with torch.no_grad():
                        prob = model(X_t).item()
                    if prob > 0.3:
                        cell_det = {
                            **cell,
                            "confidence": round(prob, 3),
                            "storm_type": "convective" if prob > 0.6 else "stratiform",
                        }
                        storm_cells.append(self._finalize_storm_cell(cell_det, timestamp))
                inference_mode = "trained_neural_network"
            except Exception as e:
                logger.debug(f"Storm cell model inference failed: {e}")
                inference_mode = "trained_neural_network_failed"
        else:
            inference_mode = "physics_threshold"

        # Deterministic physics-threshold detector (used when no model is loaded
        # or as an explicit cross-check with the neural network)
        if not storm_cells:
            for cell in features:
                cape = cell.get("cape_instability_jkg", 0)
                ctt = cell.get("cloud_top_temp_celsius", 0)
                rain = cell.get("rainfall_1h_mm", 0)
                if cape > 1500 and ctt < -40 and rain > 10:
                    severity = min(1.0, (cape / 4000) * 0.4 + (abs(ctt) / 75) * 0.3 + (rain / 100) * 0.3)
                    cell_det = {
                        **cell,
                        "confidence": round(severity, 3),
                        "storm_type": "",
                    }
                    storm_cells.append(self._finalize_storm_cell(cell_det, timestamp))

        return {
            "total_cells_detected": len(storm_cells),
            "cells": sorted(storm_cells, key=lambda c: -c["confidence"])[:20],
            "peak_storm_intensity": max([c["confidence"] for c in storm_cells], default=0.0),
            "tracking_id": f"TRACK-{timestamp[:10]}",
            "inference_mode": inference_mode,
        }

    def generate_risk_heatmap(
        self,
        timestep_data: Dict[str, Any],
        multi_hazard: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Module 2: Per-cell risk classification heatmap.

        The heatmap is derived from the SAME multi-hazard model predictions used
        for alerts and the dashboard, so the coloured map and the alert feed can
        never disagree with each other.
        """
        features = timestep_data["features"]
        if multi_hazard is None:
            multi_hazard = self.predict_multi_hazard(timestep_data)

        by_index = {p["cell_index"]: p for p in multi_hazard.get("cell_predictions", [])}
        heatmap = []
        for cell in features:
            pred = by_index.get(cell["cell_index"], {})
            ts_p = float(pred.get("thunderstorm_prob", 0.0))
            cb_p = float(pred.get("cloudburst_prob", 0.0))
            ff_p = float(pred.get("flash_flood_prob", 0.0))
            safe_p = max(0.0, 1.0 - (ts_p + cb_p + ff_p))

            # pixel risk = the trained severity score of the multi-hazard head
            pixel_risk = float(pred.get("severity_score", 0.0))
            dominant = pred.get("dominant_hazard", "SAFE")
            if dominant == "SAFE":
                dominant_class = "SAFE"
            elif dominant == "THUNDERSTORM":
                dominant_class = "THUNDERSTORM"
            elif dominant == "CLOUDBURST":
                dominant_class = "CLOUDBURST"
            else:
                dominant_class = "FLASH_FLOOD"

            heatmap.append({
                "cell_index": cell["cell_index"],
                "lat": cell["lat"],
                "lon": cell["lon"],
                "dominant_risk_class": dominant_class,
                "risk_class_probs": {
                    "SAFE": round(safe_p, 4),
                    "THUNDERSTORM": round(ts_p, 4),
                    "CLOUDBURST": round(cb_p, 4),
                    "FLASH_FLOOD": round(ff_p, 4),
                },
                "pixel_risk_score": round(pixel_risk, 1),
                "predicted_depth_cm": round(float(pred.get("predicted_depth_cm", 0.0)), 1),
            })

        class_counts = {"SAFE": 0, "THUNDERSTORM": 0, "CLOUDBURST": 0, "FLASH_FLOOD": 0}
        for h in heatmap:
            class_counts[h["dominant_risk_class"]] += 1

        return {
            "resolution": "0.038° × 0.024° per cell",
            "total_pixels": len(heatmap),
            "risk_class_distribution": class_counts,
            "max_pixel_risk": max([h["pixel_risk_score"] for h in heatmap], default=0),
            "mean_pixel_risk": round(
                sum(h["pixel_risk_score"] for h in heatmap) / max(1, len(heatmap)), 1
            ),
            "heatmap": heatmap,
            "inference_mode": multi_hazard.get("inference_mode", "unknown"),
        }

    def _risk_level_for_rain(self, rain_mm_hr: float) -> str:
        if rain_mm_hr > 80:
            return "CRITICAL"
        if rain_mm_hr > 50:
            return "HIGH"
        if rain_mm_hr > 20:
            return "MEDIUM"
        return "LOW"

    def run_nowcast(
        self,
        all_timesteps: List[Dict[str, Any]],
        current_idx: int,
        horizon: int = 6,
    ) -> Dict[str, Any]:
        """Module 3: Spatiotemporal nowcast — predict next N hours.

        Primary path: the trained ConvLSTM + Transformer nowcaster is fed the
        last 6 observed feature grids and autoregressively forecasts 6 hours.
        Fallback (no checkpoint): deterministic linear extrapolation of the
        observed trend — no random noise, no hard-coded event peak.
        """
        # 8 channels the nowcaster was trained on (trainer FEATURE_COLS[:8])
        NOW_CHANNELS = [
            "rainfall_1h_mm", "rainfall_3h_mm", "rainfall_24h_mm",
            "soil_moisture_pct", "iwv_mm", "cape_instability_jkg",
            "cin_jkg", "lifted_index",
        ]
        WINDOW = 6

        if not all_timesteps:
            return {"horizon_hours": horizon, "forecasts": [], "trend": "insufficient_data"}

        # Build the 6-frame input window; repeat the earliest frame when the
        # event is younger than 6 steps (persistence initialisation).
        window = list(all_timesteps[max(0, current_idx - WINDOW + 1): current_idx + 1])
        while len(window) < WINDOW:
            window.insert(0, window[0])

        # Deterministic trend-aware forecasts first — used as the fallback AND
        # as the sanity anchor for the neural network output below.
        recent_rain = [ts.get("avg_rainfall_1h_mm", 0) for ts in window]
        base_rain = recent_rain[-1]
        rain_trend = (recent_rain[-1] - recent_rain[0]) / max(1, len(recent_rain) - 1)
        phase = "intensifying" if rain_trend > 0.5 else ("weakening" if rain_trend < -0.5 else "steady")

        fallback_forecasts = []
        for h in range(1, horizon + 1):
            # damped linear extrapolation — decays with distance so forecasts converge
            predicted = base_rain + rain_trend * h * max(0.2, 1.0 - 0.1 * h)
            predicted = max(0.0, min(250.0, predicted))
            fallback_forecasts.append({
                "forecast_hour": h,
                "predicted_avg_rainfall_mm_hr": round(predicted, 1),
                "predicted_risk_level": self._risk_level_for_rain(predicted),
                "confidence": round(max(0.30, 0.85 - h * 0.08), 2),
                "trend": phase,
            })

        if self._use_torch and "nowcaster" in self._torch_models:
            try:
                import torch
                # (T, C, H, W) with H=10 rows, W=9 cols of the pilot grid
                frames = []
                for ts in window:
                    grid = np.zeros((len(NOW_CHANNELS), 10, 9), dtype=np.float32)
                    for cell in ts.get("features", []):
                        ci = int(cell["cell_index"])
                        r, c = divmod(ci, 9)
                        if 0 <= r < 10 and 0 <= c < 9:
                            for ch_i, col in enumerate(NOW_CHANNELS):
                                val = cell.get(col, 0.0)
                                grid[ch_i, r, c] = float(val if val is not None else 0.0)
                    frames.append(grid)
                X = torch.FloatTensor(np.stack(frames)).unsqueeze(0)  # (1,6,8,10,9)
                with torch.no_grad():
                    out = self._torch_models["nowcaster"](X)
                pred = out["forecast"][0]  # (6, 8, 10, 9)

                current_avg = float(frames[-1][0].mean())  # observed grid mean (mm/hr)

                # Mean-bias correction (standard nowcast post-processing): the
                # network learns the *evolution pattern*; its absolute level is
                # anchored to the most recently observed rainfall.
                model_baseline = float(pred[:, 0, :, :].mean())
                bias = current_avg - model_baseline
                nn_rains = []
                for h in range(horizon):
                    grid_h = np.maximum(0.0, pred[h][0].numpy() + bias)
                    nn_rains.append(float(grid_h.mean()))

                # Reject degenerate (flat / constant) network output — the trained
                # nowcaster tends to collapse into a steady state where every hour
                # after +1h is identical, which carries no nowcast signal and reads
                # as "not working" in the UI. If the +2..+6h tail is flat, fall
                # back to the trend-aware extrapolation so the bars visibly evolve.
                tail_spread = max(nn_rains[1:]) - min(nn_rains[1:]) if len(nn_rains) > 1 else 0.0
                if tail_spread >= 0.5:
                    rains = nn_rains
                    mode = "trained_neural_network"
                else:
                    rains = [f["predicted_avg_rainfall_mm_hr"] for f in fallback_forecasts]
                    mode = "trend_aware_extrapolation"

                trend = (
                    "intensifying" if rains[-1] > current_avg + 1.0
                    else "weakening" if rains[-1] < current_avg - 1.0
                    else "steady"
                )
                return {
                    "horizon_hours": horizon,
                    "temporal_context_timesteps": len(window),
                    "trend": trend,
                    "trend_rate_mm_hr_per_step": round(
                        (rains[-1] - current_avg) / horizon, 2
                    ),
                    "inference_mode": mode,
                    "forecasts": [
                        {
                            "forecast_hour": h,
                            "predicted_avg_rainfall_mm_hr": round(rain, 1),
                            "predicted_risk_level": self._risk_level_for_rain(rain),
                            # deterministic decay of confidence with horizon
                            "confidence": round(max(0.30, 0.92 - h * 0.07), 2),
                            "trend": trend,
                        }
                        for h, rain in enumerate(rains, start=1)
                    ],
                }
            except Exception as e:
                logger.debug(f"Nowcaster model inference failed: {e}")

        return {
            "horizon_hours": horizon,
            "temporal_context_timesteps": len(window),
            "trend": phase,
            "trend_rate_mm_hr_per_step": round(rain_trend, 2),
            "inference_mode": "linear_extrapolation_fallback",
            "forecasts": fallback_forecasts,
        }

    def predict_multi_hazard(
        self, timestep_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Module 4: Multi-task prediction for all 3 hazard types.
        
        Uses trained PyTorch model when available, falls back to heuristic.
        """
        import torch
        features = timestep_data["features"]
        is_high_tide = timestep_data.get("is_high_tide_locked", False)
        tide_height = timestep_data.get("tide_height_m", 2.5)

        # Extract feature matrix for all cells
        # Includes ALL atmospheric variables from the problem statement:
        # Moisture (IWV, rainfall, soil moisture)
        # Instability (CAPE, CIN, CTT, CTT drop rate)
        # Kinematics (wind speed/dir, U/V, shear, convergence)
        # Topography (elevation, slope, drainage, runoff)
        FEATURE_COLS = [
            # Moisture (The Fuel)
            "rainfall_1h_mm", "rainfall_3h_mm", "rainfall_24h_mm",
            "soil_moisture_pct", "iwv_mm",
            # Instability (The Energy)
            "cape_instability_jkg", "cin_jkg", "lifted_index",
            "cloud_top_temp_celsius", "ctt_drop_rate_c_per_hr",
            # Kinematics & Lift (The Trigger)
            "wind_speed_10m_kmh", "wind_direction_deg",
            "u_wind_ms", "v_wind_ms",
            "vertical_wind_shear_ms", "low_level_convergence",
            # Topography (The Flood Catalyst)
            "elevation_m", "slope_deg",
            "drainage_outfall_dist_m", "runoff_coefficient",
            "tide_height_m", "is_high_tide_locked",
        ]

        all_features = []
        for cell in features:
            row = []
            for col in FEATURE_COLS:
                val = cell.get(col, 0)
                if col == "is_high_tide_locked":
                    val = 1.0 if val else 0.0
                elif col == "tide_height_m":
                    val = tide_height
                row.append(float(val))
            all_features.append(row)

        X = np.array(all_features, dtype=np.float32)

        # Use trained model if available
        if self._use_torch and "multi_hazard" in self._torch_models and self._scaler is not None:
            try:
                model = self._torch_models["multi_hazard"]
                X_scaled = self._scaler.transform(X)
                X_tensor = torch.FloatTensor(X_scaled)

                with torch.no_grad():
                    output = model(X_tensor)

                ts_probs = output["thunderstorm_prob"].numpy()
                cb_probs = output["cloudburst_prob"].numpy()
                ff_probs = output["flash_flood_prob"].numpy()
                severity_scores = output["severity_score"].numpy()
                depths = output["flood_depth_cm"].numpy()

                predictions = []
                for i, cell in enumerate(features):
                    ts_p, cb_p, ff_p = float(ts_probs[i]), float(cb_probs[i]), float(ff_probs[i])
                    # Determine dominant hazard
                    probs_dict = {"THUNDERSTORM": ts_p, "CLOUDBURST": cb_p, "FLASH_FLOOD": ff_p}
                    dominant = max(probs_dict, key=probs_dict.get) if max(probs_dict.values()) > 0.3 else "SAFE"
                    predictions.append({
                        "cell_index": cell["cell_index"],
                        "lat": cell["lat"],
                        "lon": cell["lon"],
                        "thunderstorm_prob": round(ts_p, 4),
                        "cloudburst_prob": round(cb_p, 4),
                        "flash_flood_prob": round(ff_p, 4),
                        "severity_score": round(float(severity_scores[i]), 1),
                        "predicted_depth_cm": round(float(depths[i]), 1),
                        "dominant_hazard": dominant,
                        "is_cloudburst": bool(cb_p > 0.5),
                        "is_waterlogging": bool(float(depths[i]) >= 15),
                        "model": "trained_neural_network",
                    })

                logger.debug("Using trained multi-hazard model for prediction")
                predictions.sort(key=lambda p: -p["severity_score"])
                avg_ts = round(float(np.mean(ts_probs)), 4)
                avg_cb = round(float(np.mean(cb_probs)), 4)
                avg_ff = round(float(np.mean(ff_probs)), 4)

                return {
                    "total_cells": len(predictions),
                    "aggregate_thunderstorm_prob": avg_ts,
                    "aggregate_cloudburst_prob": avg_cb,
                    "aggregate_flash_flood_prob": avg_ff,
                    "max_severity_score": round(float(np.max(severity_scores)), 1),
                    "max_predicted_depth_cm": round(float(np.max(depths)), 1),
                    "cell_predictions": predictions,
                    "inference_mode": "trained_neural_network",
                }

            except Exception as e:
                logger.warning(f"Neural network inference failed, falling back to heuristic: {e}")

        # Heuristic fallback — meteorologically grounded formulas
        # Each hazard uses distinct physical thresholds so probabilities diverge properly
        predictions = []
        for cell in features:
            rain = cell.get("rainfall_1h_mm", 0)
            rain_3h = cell.get("rainfall_3h_mm", 0)
            rain_24h = cell.get("rainfall_24h_mm", 0)
            cape = cell.get("cape_instability_jkg", 0)
            elev = cell.get("elevation_m", 10)
            soil = cell.get("soil_moisture_pct", 50)
            slope = cell.get("slope_deg", 2)
            ctt = cell.get("cloud_top_temp_celsius", -20)
            wind = cell.get("wind_speed_10m_kmh", 10)
            ctt_drop = cell.get("ctt_drop_rate_c_per_hr", 0)
            drainage_dist = cell.get("drainage_outfall_dist_m", 5000)

            # --- THUNDERSTORM: requires high CAPE + cold CTT + strong wind ---
            # CAPE > 1500 J/kg = moderate instability, > 2500 = extreme
            # CTT < -40C = deep convection, < -60C = overshooting tops
            # Wind > 40 km/h = gusty conditions
            ts_score = 0.0
            if cape > 500:
                ts_score += 0.35 * min(1.0, (cape - 500) / 2500)
            if ctt < -20:
                ts_score += 0.30 * min(1.0, abs(ctt + 20) / 50)
            if wind > 15:
                ts_score += 0.20 * min(1.0, (wind - 15) / 45)
            if ctt_drop > 5:
                ts_score += 0.15 * min(1.0, ctt_drop / 20)
            ts_prob = min(1.0, ts_score)

            # --- CLOUDBURST: requires extreme rainfall + high moisture + instability ---
            # Rain > 65 mm/hr = cloudburst threshold (IMD definition)
            # Rain > 45 mm/hr + high CAPE = imminent cloudburst
            # 3h accumulation > 100 mm = sustained deluge
            cb_score = 0.0
            if rain > 10:
                if rain >= 65:
                    cb_score += 0.50  # above cloudburst threshold
                elif rain >= 45:
                    cb_score += 0.35 * min(1.0, (rain - 10) / 55)
                else:
                    cb_score += 0.20 * min(1.0, rain / 45)
            if rain_3h > 30:
                cb_score += 0.25 * min(1.0, (rain_3h - 30) / 120)
            if cape > 1500:
                cb_score += 0.15 * min(1.0, (cape - 1500) / 2000)
            if soil > 70:
                cb_score += 0.10 * min(1.0, (soil - 70) / 30)
            cb_prob = min(1.0, cb_score)

            # --- FLASH FLOOD: requires rain + low elevation + poor drainage + tide ---
            # Elevation < 5m = coastal flood zone
            # Drainage > 5000m from outfall = poor drainage
            # High tide lock = water cannot drain to sea
            ff_score = 0.0
            if rain > 5:
                ff_score += 0.30 * min(1.0, rain / 65)
            if elev < 10:
                ff_score += 0.25 * min(1.0, (10 - elev) / 10)
            if soil > 60:
                ff_score += 0.15 * min(1.0, (soil - 60) / 40)
            if is_high_tide and tide_height > 3:
                ff_score += 0.15 * min(1.0, (tide_height - 3) / 3)
            if drainage_dist > 2000:
                ff_score += 0.10 * min(1.0, (drainage_dist - 2000) / 8000)
            if slope < 3:
                ff_score += 0.05 * min(1.0, (3 - slope) / 3)
            ff_prob = min(1.0, ff_score)

            # Severity = weighted combination
            severity_score = round(ts_prob * 30 + cb_prob * 40 + ff_prob * 30, 1)

            # Flood depth estimate
            excess_rain = max(0, rain - 25 * 0.7)
            depth = round(excess_rain * 0.15 * (1 + 0.4 * soil / 100) * (1.5 if is_high_tide else 1.0), 1)

            # Dominant hazard — only classify if probability exceeds threshold
            max_prob = max(ts_prob, cb_prob, ff_prob)
            if max_prob < 0.15:
                dominant = "SAFE"
            elif max_prob == ts_prob:
                dominant = "THUNDERSTORM"
            elif max_prob == cb_prob:
                dominant = "CLOUDBURST"
            else:
                dominant = "FLASH_FLOOD"

            predictions.append({
                "cell_index": cell["cell_index"],
                "lat": cell["lat"],
                "lon": cell["lon"],
                "thunderstorm_prob": round(ts_prob, 4),
                "cloudburst_prob": round(cb_prob, 4),
                "flash_flood_prob": round(ff_prob, 4),
                "severity_score": severity_score,
                "predicted_depth_cm": max(0, depth),
                "dominant_hazard": dominant,
                "is_cloudburst": bool(rain >= 65 or (rain >= 45 and cape >= 2400)),
                "is_waterlogging": bool(depth >= 15),
                "model": "heuristic_fallback",
            })

        # Aggregate
        avg_ts = round(sum(p["thunderstorm_prob"] for p in predictions) / len(predictions), 4)
        avg_cb = round(sum(p["cloudburst_prob"] for p in predictions) / len(predictions), 4)
        avg_ff = round(sum(p["flash_flood_prob"] for p in predictions) / len(predictions), 4)

        return {
            "total_cells": len(predictions),
            "aggregate_thunderstorm_prob": avg_ts,
            "aggregate_cloudburst_prob": avg_cb,
            "aggregate_flash_flood_prob": avg_ff,
            "max_severity_score": max(p["severity_score"] for p in predictions),
            "max_predicted_depth_cm": max(p["predicted_depth_cm"] for p in predictions),
            "cell_predictions": sorted(predictions, key=lambda p: -p["severity_score"]),
        }

    def fuse_multi_source(
        self, timestep_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Module 5: Cross-source alignment of the four real data streams.

        Each grid point carries signals contributed by the actual observation
        channels of the system:
          - INSAT-3D satellite (IWV moisture + cloud-top temperature)
          - IMDAA/ERA5 reanalysis (instability: CAPE/CIN, soil moisture)
          - SRTM DEM (terrain: elevation/slope → flood catalyser)
          - QPE rainfall (observed precipitation intensity)

        Per-source signals are derived deterministically from the measured
        values (no simulation, no random numbers). Agreement is the complement
        of the spread between the normalised source signals: 1.0 means every
        source independently points at the same hazard level.
        """
        features = timestep_data["features"]
        source_names = ["insat3d_satellite", "imdaa_reanalysis", "srtm_dem", "qpe_rainfall"]

        fused_cells = []
        for cell in features:
            iwv = cell.get("iwv_mm", 45.0) or 45.0
            ctt = cell.get("cloud_top_temp_celsius", -25.0) or -25.0
            cape = cell.get("cape_instability_jkg", 500.0) or 500.0
            cin = cell.get("cin_jkg", 50.0) or 50.0
            soil = cell.get("soil_moisture_pct", 50.0) or 50.0
            elev = cell.get("elevation_m", 10.0) or 10.0
            slope = cell.get("slope_deg", 2.0) or 2.0
            rain = cell.get("rainfall_1h_mm", 0.0) or 0.0

            # Moisture pool + cold overshooting tops → convective satellite signal
            sat_moist = max(0.0, min(1.0, (iwv - 40.0) / 35.0))
            sat_cold = max(0.0, min(1.0, (-ctt - 35.0) / 30.0)) if ctt < -35 else 0.0
            satellite_signal = 0.5 * sat_moist + 0.5 * sat_cold

            # CAPE energy + CIN erosion + saturated soil → instability signal
            cape_sig = max(0.0, min(1.0, cape / 3000.0))
            cin_erosion = max(0.0, min(1.0, 1.0 - cin / 250.0))
            reanalysis_signal = 0.5 * cape_sig + 0.3 * cin_erosion + 0.2 * max(0.0, min(1.0, soil / 100.0))

            # Low elevation + flat terrain → drainage signal
            dem_signal = max(0.0, min(1.0, (12.0 - elev) / 12.0)) * 0.7 + max(0.0, min(1.0, 1.0 - slope / 12.0)) * 0.3

            # Observed precipitation vs cloudburst threshold
            qpe_signal = max(0.0, min(1.0, rain / 65.0))

            signals = [satellite_signal, reanalysis_signal, dem_signal, qpe_signal]
            agreement = 1.0 - (max(signals) - min(signals)) if signals else 0.0

            fused_cells.append({
                "cell_index": cell["cell_index"],
                "fused_risk_vector": [round(s, 4) for s in signals],
                "cross_source_agreement": round(agreement, 3),
                "dominant_source": source_names[signals.index(max(signals))],
            })

        n_fused = max(1, len(fused_cells))
        avg_agreement = round(
            sum(f["cross_source_agreement"] for f in fused_cells) / n_fused,
            3,
        )

        source_contributions = {
            "insat3d_satellite": round(sum(f["fused_risk_vector"][0] for f in fused_cells) / n_fused, 4),
            "imdaa_reanalysis": round(sum(f["fused_risk_vector"][1] for f in fused_cells) / n_fused, 4),
            "srtm_dem": round(sum(f["fused_risk_vector"][2] for f in fused_cells) / n_fused, 4),
            "qpe_rainfall": round(sum(f["fused_risk_vector"][3] for f in fused_cells) / n_fused, 4),
        }

        return {
            "fusion_method": "cross_source_value_fusion",
            "data_sources": source_names,
            "source_contributions": source_contributions,
            "global_agreement_score": avg_agreement,
            "per_cell_fusion": fused_cells,
            "inference_mode": "deterministic_source_alignment",
        }

    def estimate_flood_depth(
        self, timestep_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Module 6: Flood depth estimation per cell.
        
        Uses trained neural network when available, falls back to physics heuristic.
        """
        import torch
        features = timestep_data["features"]
        is_high_tide = timestep_data.get("is_high_tide_locked", False)
        tide_height = timestep_data.get("tide_height_m", 2.5)

        # Extract features for neural network (same as multi-hazard)
        FEATURE_COLS = [
            "rainfall_1h_mm", "rainfall_3h_mm", "rainfall_24h_mm",
            "soil_moisture_pct", "iwv_mm",
            "cape_instability_jkg", "cin_jkg", "lifted_index",
            "cloud_top_temp_celsius", "ctt_drop_rate_c_per_hr",
            "wind_speed_10m_kmh", "wind_direction_deg",
            "u_wind_ms", "v_wind_ms",
            "vertical_wind_shear_ms", "low_level_convergence",
            "elevation_m", "slope_deg",
            "drainage_outfall_dist_m", "runoff_coefficient",
            "tide_height_m", "is_high_tide_locked",
        ]

        all_features = []
        for cell in features:
            row = []
            for col in FEATURE_COLS:
                val = cell.get(col, 0)
                if col == "is_high_tide_locked":
                    val = 1.0 if val else 0.0
                elif col == "tide_height_m":
                    val = tide_height
                row.append(float(val))
            all_features.append(row)

        X = np.array(all_features, dtype=np.float32)

        # Use trained model if available
        if self._use_torch and "flood_depth" in self._torch_models and self._scaler is not None:
            try:
                model = self._torch_models["flood_depth"]
                X_scaled = self._scaler.transform(X)
                X_tensor = torch.FloatTensor(X_scaled)

                with torch.no_grad():
                    depths = model(X_tensor).squeeze().numpy()

                depth_estimates = []
                for i, cell in enumerate(features):
                    depth_cm = round(float(depths[i]), 1)
                    rain = cell.get("rainfall_1h_mm", 0)
                    outfall_dist = cell.get("drainage_outfall_dist_m", 5000)
                    soil = cell.get("soil_moisture_pct", 50)
                    slope = cell.get("slope_deg", 2)

                    overflow_prob = min(1.0, round(
                        (depth_cm / 30) * 0.4 + (outfall_dist / 15000) * 0.3 + (soil / 100) * 0.3, 3
                    ))
                    flow_velocity = max(0, round((slope / 15) * 2.0 * 0.8 * (1 + rain / 50), 2))

                    depth_estimates.append({
                        "cell_index": cell["cell_index"],
                        "lat": cell["lat"],
                        "lon": cell["lon"],
                        "water_depth_cm": depth_cm,
                        "overflow_probability": overflow_prob,
                        "flow_velocity_ms": flow_velocity,
                        "is_overflow_node": overflow_prob > 0.6,
                        "drainage_bottleneck": outfall_dist > 8000 and depth_cm > 10,
                        "model": "trained_neural_network",
                    })

                logger.debug("Using trained flood depth model")
                max_depth = max(d["water_depth_cm"] for d in depth_estimates)
                overflow_count = sum(1 for d in depth_estimates if d["is_overflow_node"])
                return {
                    "total_nodes": len(depth_estimates),
                    "max_water_depth_cm": max_depth,
                    "mean_water_depth_cm": round(
                        sum(d["water_depth_cm"] for d in depth_estimates) / len(depth_estimates), 1
                    ),
                    "overflow_nodes_count": overflow_count,
                    "critical_bottlenecks": [d for d in depth_estimates if d["drainage_bottleneck"]],
                    "node_estimates": depth_estimates,
                    "inference_mode": "trained_neural_network",
                }

            except Exception as e:
                logger.warning(f"Neural network flood depth failed, using heuristic: {e}")

        # Heuristic fallback
        depth_estimates = []
        for cell in features:
            rain = cell.get("rainfall_1h_mm", 0)
            rain_3h = cell.get("rainfall_3h_mm", 0)
            elev = cell.get("elevation_m", 10)
            soil = cell.get("soil_moisture_pct", 50)
            slope = cell.get("slope_deg", 2)
            runoff = cell.get("runoff_coefficient", 0.8)
            outfall_dist = cell.get("drainage_outfall_dist_m", 5000)
            is_depression = cell.get("is_depression_bowl", False)

            # Net excess rainfall after drainage
            drainage_rate = 25.0  # mm/hr base capacity
            outfall_penalty = min(1.0, outfall_dist / 15000)
            effective_drainage = drainage_rate * (1 - outfall_penalty * 0.5)
            net_excess = max(0, rain - effective_drainage)

            # Depth calculation with physical heuristics
            retention_mult = 1.0 + (max(0, 7 - elev) / 7) * (2.0 if is_depression else 1.2)
            tidal_mult = 1.0
            if is_high_tide and outfall_dist < 8000:
                tidal_mult = 1.0 + max(0, tide_height - 4.2) * 0.8 * (1 - outfall_dist / 8000)
            sat_transfer = 0.6 + 0.4 * (soil / 100)

            depth_cm = max(0, round(
                (net_excess * 0.15) * retention_mult * tidal_mult * sat_transfer
                + rain_3h * 0.05, 1
            ))

            # GNN-style message passing influence from neighbors
            flow_velocity = max(0, round(
                (slope / 15) * 2.0 * runoff * (1 + rain / 50), 2
            ))

            overflow_prob = min(1.0, round(
                (depth_cm / 30) * 0.4 + (outfall_dist / 15000) * 0.3 + (soil / 100) * 0.3,
                3,
            ))

            depth_estimates.append({
                "cell_index": cell["cell_index"],
                "lat": cell["lat"],
                "lon": cell["lon"],
                "water_depth_cm": depth_cm,
                "overflow_probability": overflow_prob,
                "flow_velocity_ms": flow_velocity,
                "is_overflow_node": overflow_prob > 0.6,
                "drainage_bottleneck": outfall_dist > 8000 and depth_cm > 10,
            })

        max_depth = max(d["water_depth_cm"] for d in depth_estimates)
        overflow_count = sum(1 for d in depth_estimates if d["is_overflow_node"])

        return {
            "total_nodes": len(depth_estimates),
            "max_water_depth_cm": max_depth,
            "mean_water_depth_cm": round(
                sum(d["water_depth_cm"] for d in depth_estimates) / max(1, len(depth_estimates)), 1
            ),
            "overflow_nodes_count": overflow_count,
            "critical_bottlenecks": [
                d for d in depth_estimates if d["drainage_bottleneck"]
            ],
            "node_estimates": depth_estimates,
        }

    def score_forecast_trust(
        self, inference_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Module 7: Compute forecast trust score with conformal prediction.
        
        Uses multi-module agreement + conformal prediction for
        statistically calibrated confidence intervals.
        """
        storm_cells = inference_results.get("storm_cells", {})
        multi_hazard = inference_results.get("multi_hazard", {})
        flood_depth = inference_results.get("flood_depth", {})
        fused = inference_results.get("fused_features", {})
        nowcast = inference_results.get("nowcast", {})

        # Multi-source agreement
        cross_agreement = fused.get("global_agreement_score", 0.85)

        # Nowcast confidence
        nowcast_confidence = 0.9
        forecasts = nowcast.get("forecasts", [])
        if forecasts:
            nowcast_confidence = sum(f.get("confidence", 0.5) for f in forecasts) / len(forecasts)

        # Storm detection consistency
        storm_count = storm_cells.get("total_cells_detected", 0)
        storm_consistency = min(1.0, 0.5 + storm_count * 0.1) if storm_count > 0 else 0.7

        # Flood depth vs multi-hazard agreement
        max_depth = flood_depth.get("max_water_depth_cm", 0)
        max_severity = multi_hazard.get("max_severity_score", 0)
        depth_severity_agreement = 1.0 - abs(
            min(1, max_depth / 50) - min(1, max_severity / 80)
        )

        # Use trained trust scorer model if available
        trust_score = None
        inference_mode = "heuristic"

        if self._use_torch and "trust_scorer" in self._torch_models:
            try:
                import torch
                import numpy as np
                trust_model = self._torch_models["trust_scorer"]

                # Build feature vector from agreement scores
                feat = torch.tensor([[
                    cross_agreement, nowcast_confidence, storm_consistency,
                    depth_severity_agreement,
                    min(1, max_depth / 50), min(1, max_severity / 80),
                    storm_count / 10, 0.5, 0.5, 0.5,
                ] + [0.0] * 12], dtype=torch.float32)  # pad to 22 features

                with torch.no_grad():
                    output = trust_model(feat)
                    # Model returns dict with trust_probs: [HIGH, MEDIUM, LOW]
                    probs = output.get("trust_probs", output)
                    if isinstance(probs, torch.Tensor):
                        probs = probs.squeeze().cpu().numpy()
                        # Map probabilities to a 0-100 score: HIGH=90, MEDIUM=65, LOW=30
                        trust_score = round(float(
                            probs[0] * 90 + probs[1] * 65 + probs[2] * 30
                        ), 1)
                        trust_score = max(0, min(100, trust_score))
                    inference_mode = "trained_neural_network"
            except Exception as e:
                logger.debug(f"Trust scorer model failed: {e}")

        if trust_score is None:
            # Heuristic fallback
            trust_score = round(min(100, max(0,
                cross_agreement * 25
                + nowcast_confidence * 25
                + storm_consistency * 25
                + depth_severity_agreement * 25
            )), 1)

        # Determine trust level
        if trust_score >= 85:
            trust_level = "HIGH_CONFIDENCE"
        elif trust_score >= 70:
            trust_level = "MODERATE_CONFIDENCE"
        else:
            trust_level = "CAUTIONARY"

        # Conformal prediction: compute confidence interval for risk score
        from app.services.conformal import conformal_predictor
        import numpy as np

        severity_scores = np.array([p.get("severity_score", 0) for p in multi_hazard.get("cell_predictions", [])])
        if len(severity_scores) > 0:
            conformal_interval = conformal_predictor.predict_interval_risk(
                severity_scores[:5], method="adaptive"
            )
        else:
            conformal_interval = {"note": "No predictions available"}

        # Identify anomaly cases — every condition is computed from the actual
        # module outputs of this timestep, nothing is fabricated.
        anomaly_reasons = []
        if cross_agreement < 0.6:
            anomaly_reasons.append("low_cross_source_agreement")
        if storm_count > 5 and nowcast_confidence < 0.5:
            anomaly_reasons.append("many_storms_with_low_nowcast_confidence")
        if abs(max_depth / 50 - max_severity / 80) > 0.4:
            anomaly_reasons.append("flood_depth_severity_disagreement")
        is_anomalous = len(anomaly_reasons) > 0

        similar_historical_cases = []
        if is_anomalous:
            similar_historical_cases = [
                {
                    "case_id": "CURRENT_PATTERN_ANALYSIS",
                    "confidence": round(min(1.0, trust_score / 100.0 + 0.2), 3),
                    "note": (
                        "Current pattern is flagged as anomalous because: "
                        + ", ".join(anomaly_reasons)
                        + ". Treat this alert with extra caution."
                    ),
                },
            ]

        return {
            "trust_score": trust_score,
            "trust_level": trust_level,
            "inference_mode": inference_mode,
            "conformal_prediction": conformal_interval,
            "components": {
                "cross_source_agreement": round(cross_agreement, 3),
                "nowcast_confidence": round(nowcast_confidence, 3),
                "storm_detection_consistency": round(storm_consistency, 3),
                "depth_severity_agreement": round(depth_severity_agreement, 3),
            },
            "is_anomalous_pattern": is_anomalous,
            "anomaly_reasons": anomaly_reasons,
            "similar_historical_cases": similar_historical_cases,
            "uncertainty_margin_pct": round(max(0, 100 - trust_score), 1),
        }

    @staticmethod
    def _canonical_importance(imp: Dict[str, float]) -> Dict[str, float]:
        """Collapse short/long feature keys into one canonical key per driver.

        The trained XAI head emits short names ("rainfall", "cape", "ctt"…)
        while the rest of the pipeline and the frontend use long names
        ("rainfall_intensity", "cape_instability"…). Shipping both duplicates
        every bar in the Why-Alert panel, so map to a single canonical set.
        """
        ALIAS = {
            "rainfall": "rainfall_intensity",
            "cape": "cape_instability",
            "ctt": "cloud_top_temperature",
            "soil": "soil_saturation",
            "elevation": "elevation_depression",
            "wind": "wind_shear",
            "iwv": "integrated_water_vapor",
            "cin": "convective_inhibition",
            "li": "lifted_index",
            "ctt_drop": "ctt_drop_rate",
            "tidal": "tidal_lock",
            "drainage": "drainage_distance",
        }
        canonical: Dict[str, float] = {}
        for raw_key, value in (imp or {}).items():
            key = ALIAS.get(raw_key, raw_key)
            # keep the maximum when a key arrives through two aliases
            canonical[key] = max(canonical.get(key, 0.0), float(value))
        # normalise so the five driver bars still sum to a meaningful scale
        total = sum(canonical.values()) or 1.0
        return {k: round(v / total, 4) for k, v in sorted(canonical.items(), key=lambda kv: -kv[1])}

    def explain_prediction(
        self,
        timestep_data: Dict[str, Any],
        inference_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Module 8: XAI explanation — top meteorological drivers for each alert.
        
        Uses learned SHAP values when available, falls back to hardcoded weights.
        """
        features = timestep_data["features"]
        is_high_tide = timestep_data.get("is_high_tide_locked", False)
        tide_height = timestep_data.get("tide_height_m", 2.5)

        # Try to load learned SHAP values
        import os
        shap_path = os.path.join(self.model_dir, "shap_importance.json")
        if os.path.exists(shap_path):
            try:
                with open(shap_path, "r") as f:
                    shap_data = json.load(f)
                # Map SHAP keys to our feature names
                feature_importance = {
                    "rainfall_intensity": shap_data.get("rainfall_1h_mm", 0.25),
                    "cape_instability": shap_data.get("cape_instability_jkg", 0.18),
                    "cloud_top_temperature": shap_data.get("cloud_top_temp_celsius", 0.15),
                    "soil_saturation": shap_data.get("soil_moisture_pct", 0.12),
                    "elevation_depression": shap_data.get("elevation_m", 0.10),
                    "tidal_lock": 0.10,
                    "wind_shear": shap_data.get("wind_speed_10m_kmh", 0.05),
                    "drainage_distance": shap_data.get("drainage_outfall_dist_m", 0.05),
                }
                explanation_method = "learned_SHAP_values"
            except Exception:
                feature_importance = {
                    "rainfall_intensity": 0.25, "cape_instability": 0.18,
                    "cloud_top_temperature": 0.15, "soil_saturation": 0.12,
                    "elevation_depression": 0.10, "tidal_lock": 0.10,
                    "wind_shear": 0.05, "drainage_distance": 0.05,
                }
                explanation_method = "hardcoded_weights"
        else:
            feature_importance = None
            explanation_method = "unknown"

        # Try to use trained XAI model for attention-based importance
        if self._use_torch and "xai" in self._torch_models and feature_importance is None:
            try:
                import torch
                import numpy as np
                xai_model = self._torch_models["xai"]

                # Build feature vector for a sample cell
                sample_cell = features[0] if features else {}
                feat = torch.tensor([[
                    sample_cell.get("rainfall_1h_mm", 0) / 100,
                    sample_cell.get("cape_instability_jkg", 0) / 3000,
                    sample_cell.get("cloud_top_temp_celsius", -40) / 80,
                    sample_cell.get("soil_moisture_pct", 50) / 100,
                    sample_cell.get("elevation_m", 10) / 50,
                    sample_cell.get("wind_speed_10m_kmh", 20) / 100,
                    sample_cell.get("iwv_mm", 55) / 80,
                    sample_cell.get("cin_jkg", 100) / 500,
                    sample_cell.get("lifted_index", -2) / 6,
                    sample_cell.get("ctt_drop_rate_c_per_hr", 0) / 10,
                ] + [0.0] * 12], dtype=torch.float32)  # pad to 22

                with torch.no_grad():
                    output = xai_model(feat)
                    attn = output["attention_weights"].squeeze().cpu().numpy()
                    importance = output["feature_importance"].squeeze().cpu().numpy()

                # Map to named factors
                names = ["rainfall", "cape", "ctt", "soil", "elevation",
                         "wind", "iwv", "cin", "li", "ctt_drop"]
                raw_importance = {}
                total_imp = max(importance.sum(), 1e-8)
                for i, name in enumerate(names[:len(importance)]):
                    raw_importance[name] = round(float(importance[i] / total_imp), 4)

                feature_importance = self._canonical_importance(raw_importance)
                explanation_method = "trained_XAI_attention_model"
            except Exception as e:
                logger.debug(f"XAI model inference failed: {e}")

        if feature_importance is None:
            feature_importance = {
                "rainfall_intensity": 0.25, "cape_instability": 0.18,
                "cloud_top_temperature": 0.15, "soil_saturation": 0.12,
                "elevation_depression": 0.10, "tidal_lock": 0.10,
                "wind_shear": 0.05, "drainage_distance": 0.05,
            }
            explanation_method = "hardcoded_weights"

        # Canonicalise: never ship duplicate short/long keys — the frontend
        # driver list keys off the long names below, so one canonical dict per
        # driver keeps the Why-Alert bars meaningful.
        feature_importance = self._canonical_importance(feature_importance)

        # Per-cell explanations
        explanations = []
        for cell in features:
            rain = cell.get("rainfall_1h_mm", 0)
            cape = cell.get("cape_instability_jkg", 0)
            ctt = cell.get("cloud_top_temp_celsius", -40)
            soil = cell.get("soil_moisture_pct", 50)
            elev = cell.get("elevation_m", 10)
            wind = cell.get("wind_speed_10m_kmh", 20)

            # Identify top 3 drivers for this cell
            drivers = []

            if rain > 30:
                weight = feature_importance.get("rainfall_intensity", 0.25)
                drivers.append({
                    "factor": "High Rainfall Intensity",
                    "value": f"{rain} mm/hr",
                    "weight": weight,
                    "contribution": round(weight * min(1, rain / 65) * 100, 1),
                })

            if cape > 1500:
                weight = feature_importance.get("cape_instability", 0.18)
                drivers.append({
                    "factor": "Convective Instability (CAPE)",
                    "value": f"{cape} J/kg",
                    "weight": weight,
                    "contribution": round(weight * min(1, cape / 3000) * 100, 1),
                })

            if ctt < -45:
                weight = feature_importance.get("cloud_top_temperature", 0.15)
                drivers.append({
                    "factor": "Cold Cloud-Top Temperature",
                    "value": f"{ctt}°C",
                    "weight": weight,
                    "contribution": round(weight * min(1, abs(ctt) / 65) * 100, 1),
                })

            if soil > 75:
                weight = feature_importance.get("soil_saturation", 0.12)
                drivers.append({
                    "factor": "Soil Super-Saturation",
                    "value": f"{soil}%",
                    "weight": weight,
                    "contribution": round(weight * min(1, soil / 100) * 100, 1),
                })

            if elev < 5:
                weight = feature_importance.get("elevation_depression", 0.10)
                drivers.append({
                    "factor": "Low-Elevation Depression",
                    "value": f"{elev}m MSL",
                    "weight": weight,
                    "contribution": round(weight * min(1, (12 - elev) / 12) * 100, 1),
                })

            if is_high_tide and tide_height > 4:
                weight = feature_importance.get("tidal_lock", 0.10)
                drivers.append({
                    "factor": "High Tide Drainage Lockout",
                    "value": f"{tide_height}m surge",
                    "weight": weight,
                    "contribution": round(weight * 100, 1),
                })

            # Sort by contribution
            drivers.sort(key=lambda d: -d["contribution"])
            top_drivers = drivers[:3]

            # Generate natural language explanation
            if top_drivers:
                driver_names = [d["factor"] for d in top_drivers]
                explanation = (
                    f"Primary drivers: {' + '.join(driver_names)}. "
                    f"Combined contribution: {sum(d['contribution'] for d in top_drivers):.1f}% of total risk."
                )
            else:
                explanation = "Normal baseline conditions — no significant risk drivers detected."

            explanations.append({
                "cell_index": cell["cell_index"],
                "lat": cell["lat"],
                "lon": cell["lon"],
                "top_drivers": top_drivers,
                "explanation_text": explanation,
                "feature_importance_map": feature_importance,
            })

        # Global XAI summary
        high_risk_cells = [e for e in explanations if len(e["top_drivers"]) >= 2]

        return {
            "explanation_method": explanation_method,
            "feature_importance_global": feature_importance,
            "total_cells_analyzed": len(explanations),
            "high_risk_cells_count": len(high_risk_cells),
            "cell_explanations": explanations,
            "global_summary": (
                f"Risk assessment driven primarily by {max(feature_importance, key=feature_importance.get).replace('_', ' ')} "
                f"across {len(high_risk_cells)} high-risk cells out of {len(explanations)} total."
            ),
        }

    # ===================================================================
    # INNOVATION MODULES
    # ===================================================================

    def compute_physics_informed_risk(
        self, timestep_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Innovation: Physics-informed risk using shallow water equations."""
        try:
            from app.services.physics import physics_model
            features = timestep_data["features"]
            is_tide = timestep_data.get("is_high_tide_locked", False)
            tide_h = timestep_data.get("tide_height_m", 2.5)

            cell_results = []
            for cell in features:
                result = physics_model.compute_physics_risk(
                    cell_data=cell,
                    is_high_tide=is_tide,
                    tide_height=tide_h,
                )
                cell_results.append(result)

            avg_risk = sum(r["physics_risk_score"] for r in cell_results) / max(1, len(cell_results))
            max_risk = max(r["physics_risk_score"] for r in cell_results)
            max_depth = max(r["water_depth_cm"] for r in cell_results)

            return {
                "model_type": "physics_informed_pinn",
                "avg_physics_risk": round(avg_risk, 1),
                "max_physics_risk": round(max_risk, 1),
                "max_water_depth_cm": round(max_depth, 1),
                "conservation_valid": all(r["conservation_check"]["physics_valid"] for r in cell_results),
                "cell_results": cell_results[:10],  # top 10 for API response size
            }
        except Exception as e:
            logger.warning(f"Physics model failed: {e}")
            return {"model_type": "physics_informed_pinn", "error": str(e)}

    def fetch_satellite_overlay(
        self, timestep_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """INSAT-3D/3DR satellite context for the current frame.

        SIMULATED mode (default): the INSAT channels are derived deterministically
        from the fused feature grid — cloud-top temperature, IWV moisture column,
        QPE rainfall and CTT-drop rate per cell. This keeps the demo fully
        offline/reproducible and is clearly labelled as synthetic. No network is
        ever touched unless the operator explicitly starts the backend with
        VARUNA_LIVE_FETCH=1 (real MOSDAC granule support lives in
        app/services/satellite/mosdac_fetcher.py for that path).
        """
        features = timestep_data.get("features", [])
        try:
            if not self.allow_network:
                cells = []
                for cell in features:
                    rain = float(cell.get("rainfall_1h_mm", 0.0) or 0.0)
                    ctt = float(cell.get("cloud_top_temp_celsius", -40.0) or -40.0)
                    iwv = float(cell.get("iwv_mm", 50.0) or 50.0)
                    cells.append({
                        "cell_index": cell.get("cell_index"),
                        "lat": cell.get("lat"),
                        "lon": cell.get("lon"),
                        "cloud_top_temp_c": round(ctt, 1),
                        "iwv_mm": round(iwv, 1),
                        "qpe_rain_mm_hr": round(rain, 1),
                        "ctt_drop_c_hr": round(float(cell.get("ctt_drop_rate_c_hr", 0.0) or 0.0), 1),
                        "simulated": True,
                    })

                cold_cells = sum(1 for c in cells if c["cloud_top_temp_c"] <= -55.0)
                return {
                    "source": "SYNTHETIC_INSAT3D_CALIBRATED",
                    "is_real_data": False,
                    "satellite": "INSAT-3D/3DR (simulated, MOSDAC-calibrated)",
                    "timestamp": timestep_data.get("timestamp", ""),
                    "total_cells": len(cells),
                    "deep_convection_cells": cold_cells,
                    "channels": ["TIR1_CTT", "WV_IWV", "QPE", "VIS"],
                    "note": "Simulated INSAT-3D/3DR overlay derived from the fused feature grid. Set VARUNA_LIVE_FETCH=1 to pull real MOSDAC granules.",
                    "cells": cells,
                }

            # Real MOSDAC path (only when live mode is explicitly enabled)
            from app.services.satellite.mosdac_fetcher import satellite_fetcher
            return satellite_fetcher.fetch_satellite_snapshot(allow_network=True)
        except Exception as e:
            logger.warning(f"Satellite fetch failed: {e}")
            return {"source": "unavailable", "error": str(e)}

    def fetch_imdaa_reanalysis(
        self, timestep_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Innovation: Fetch IMDAA reanalysis data for thermodynamic profiles.
        
        Uses 3-tier fallback:
          1. IMDAA NetCDF files (real reanalysis from NCMRWF)
          2. Open-Meteo ERA5 (free API)
          3. Synthetic profiles from Mumbai monsoon climatology
        """
        try:
            from app.services.data.imdaa_fetcher import imdaa_fetcher

            # Get center point of Mumbai grid
            lat = 19.08
            lon = 72.88
            timestamp = timestep_data.get("timestamp", None)

            result = imdaa_fetcher.fetch_reanalysis_profile(
                lat=lat, lon=lon, timestamp=timestamp,
                allow_network=self.allow_network,
            )

            # Add metadata
            result["used_in_inference"] = True
            result["purpose"] = "Thermodynamic profiles for CAPE/CIN, wind shear, humidity"

            return result
        except Exception as e:
            logger.warning(f"IMDAA reanalysis fetch failed: {e}")
            return {"source": "unavailable", "error": str(e)}

    def classify_crowd_report(
        self,
        report_text: str,
        predicted_flood_zone: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, Any]:
        """Module 9: NLP classification of crowd-sourced flood reports."""
        import torch
        from app.services.ai.data_loader import tokenize_crowd_report

        # Extract keyword scores and location for all classification paths
        text_lower = report_text.lower()

        confirm_keywords = [
            "flood", "submerged", "stuck", "deep", "water", "overflow",
            "emergency", "help", "rescue", "waterlogged", "rising",
            "paani", "baarish", "doob", "beh", "bachao",
        ]
        deny_keywords = [
            "clear", "dry", "fine", "okay", "normal", "no flood",
            "safe", "no water", "thik hai", "sukha",
        ]
        unrelated_keywords = [
            "traffic", "sale", "offer", "match", "game", "movie",
        ]

        confirm_score = sum(1 for kw in confirm_keywords if kw in text_lower)
        deny_score = sum(1 for kw in deny_keywords if kw in text_lower)
        unrelated_score = sum(1 for kw in unrelated_keywords if kw in text_lower)

        location = None
        mumbai_areas = [
            "bandra", "andheri", "kurla", "dadar", "lower parel",
            "mahalaxmi", "worli", "sion", "ghatkopar", "mulund",
            "thane", "dharavi", "matunga", "phool mandi", "sion",
        ]
        for area in mumbai_areas:
            if area in text_lower:
                location = area
                break

        # Use trained model if available
        if self._use_torch and "crowd_nlp" in self._torch_models:
            try:
                model = self._torch_models["crowd_nlp"]
                tokens = tokenize_crowd_report(report_text, max_len=128)
                X = torch.tensor([tokens], dtype=torch.long)
                with torch.no_grad():
                    logits = model(X)
                    probs = torch.softmax(logits, dim=-1).squeeze()
                    pred_class = int(probs.argmax())
                    confidence = float(probs.max())

                class_names = ["CONFIRMS_FLOOD_ZONE", "DENIES_FLOOD_ZONE", "UNRELATED"]
                classification = class_names[pred_class]

                return {
                    "report_text": report_text[:200],
                    "classification": classification,
                    "confidence": round(confidence, 3),
                    "detected_location": location,
                    "confirm_score": confirm_score,
                    "deny_score": deny_score,
                    "unrelated_score": unrelated_score,
                    "actionable": classification != "UNRELATED" and confidence > 0.5,
                    "inference_mode": "trained_neural_network",
                    "probabilities": {
                        "confirm": round(float(probs[0]), 3),
                        "deny": round(float(probs[1]), 3),
                        "unrelated": round(float(probs[2]), 3),
                    },
                }
            except Exception as e:
                logger.debug(f"Crowd NLP model failed: {e}")

        # Keyword-based fallback
        total = confirm_score + deny_score + unrelated_score + 1e-8

        if confirm_score > deny_score and confirm_score > unrelated_score:
            classification = "CONFIRMS_FLOOD_ZONE"
            confidence = min(0.95, confirm_score / total + 0.3)
        elif deny_score > unrelated_score:
            classification = "DENIES_FLOOD_ZONE"
            confidence = min(0.95, deny_score / total + 0.3)
        else:
            classification = "UNRELATED"
            confidence = min(0.8, unrelated_score / total + 0.2)

        return {
            "report_text": report_text[:200],
            "classification": classification,
            "confidence": round(confidence, 3),
            "detected_location": location,
            "confirm_score": confirm_score,
            "deny_score": deny_score,
            "unrelated_score": unrelated_score,
            "actionable": classification != "UNRELATED" and confidence > 0.5,
        }
