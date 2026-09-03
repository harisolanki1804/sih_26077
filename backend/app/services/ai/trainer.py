"""
VARUNA AI Model Training Pipeline
===================================
Trains all 9 AI/ML modules on the existing 72-timestep Mumbai dataset.
Saves checkpoints to models/checkpoints/ for use by inference engine.

Uses:
- PyTorch for neural network training
- scikit-learn for data splitting and metrics
- Conformal prediction for calibration
- SHAP for post-hoc explainability
"""

import os
import sys
import json
import math
import logging
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

logger = logging.getLogger("VARUNA.Trainer")

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available — training disabled")

try:
    import sklearn
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# Feature columns used by the models (22 features)
# Includes ALL atmospheric variables from the problem statement
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

GRID_ROWS = 10
GRID_COLS = 9
GRID_CELLS = GRID_ROWS * GRID_COLS  # 90


class VARUNATrainer:
    """
    End-to-end training pipeline for all VARUNA AI modules.

    Data flow:
    1. Load feature_grid_timeseries.json
    2. Extract feature vectors and targets for each cell/timestep
    3. Create sliding window datasets
    4. Train each module
    5. Save checkpoints
    6. Calibrate conformal predictor
    7. Compute SHAP values on trained model
    """

    def __init__(self, data_dir: str = None, model_dir: str = None):
        if data_dir is None:
            # Navigate from backend/app/services/ai/ to project root/data
            data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "data"))
        if model_dir is None:
            model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "models", "checkpoints"))

        self.data_dir = data_dir
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

        self.device = torch.device("cpu")  # CPU for hackathon compatibility

        # Data storage
        self.raw_data = None
        self.feature_matrix = None
        self.target_matrix = None
        self.scaler = StandardScaler()

    def load_data(self) -> Dict[str, Any]:
        """Load and preprocess the feature grid timeseries."""
        data_path = os.path.join(self.data_dir, "feature_grid_timeseries.json")

        with open(data_path, "r", encoding="utf-8") as f:
            self.raw_data = json.load(f)

        timesteps = self.raw_data["timesteps"]
        n_timesteps = len(timesteps)

        logger.info(f"Loaded {n_timesteps} timesteps with {GRID_CELLS} cells each")

        # Build feature matrix: (n_timesteps * n_cells, n_features)
        all_features = []
        all_targets = []

        for ts in timesteps:
            for cell in ts["features"]:
                features = []
                for col in FEATURE_COLS:
                    val = cell.get(col, 0)
                    if col == "is_high_tide_locked":
                        val = 1.0 if val else 0.0
                    features.append(float(val))
                all_features.append(features)

                # Multi-task labels: thunderstorm_prob, cloudburst_prob, flash_flood_prob, severity_score, depth
                rain = cell.get("rainfall_1h_mm", 0)
                rain_3h = cell.get("rainfall_3h_mm", 0)
                elev = cell.get("elevation_m", 10)
                cape = cell.get("cape_instability_jkg", 0)
                soil = cell.get("soil_moisture_pct", 50)
                slope = cell.get("slope_deg", 2)
                ctt = cell.get("cloud_top_temp_celsius", -20)
                wind = cell.get("wind_speed_10m_kmh", 10)
                ctt_drop = cell.get("ctt_drop_rate_c_per_hr", 0)
                drainage_dist = cell.get("drainage_outfall_dist_m", 5000)
                is_high_tide = cell.get("is_high_tide_locked", False)
                tide_height = cell.get("tide_height_m", 2.5)

                # Thunderstorm probability (CAPE + cold CTT + wind)
                ts_prob = 0.0
                if cape > 500:
                    ts_prob += 0.35 * min(1.0, (cape - 500) / 2500)
                if ctt < -20:
                    ts_prob += 0.30 * min(1.0, abs(ctt + 20) / 50)
                if wind > 15:
                    ts_prob += 0.20 * min(1.0, (wind - 15) / 45)
                if ctt_drop > 5:
                    ts_prob += 0.15 * min(1.0, ctt_drop / 20)
                ts_prob = min(1.0, ts_prob)

                # Cloudburst probability (extreme rain + moisture)
                cb_prob = 0.0
                if rain > 10:
                    if rain >= 65:
                        cb_prob += 0.50
                    elif rain >= 45:
                        cb_prob += 0.35 * min(1.0, (rain - 10) / 55)
                    else:
                        cb_prob += 0.20 * min(1.0, rain / 45)
                if rain_3h > 30:
                    cb_prob += 0.25 * min(1.0, (rain_3h - 30) / 120)
                if cape > 1500:
                    cb_prob += 0.15 * min(1.0, (cape - 1500) / 2000)
                if soil > 70:
                    cb_prob += 0.10 * min(1.0, (soil - 70) / 30)
                cb_prob = min(1.0, cb_prob)

                # Flash flood probability (rain + low elevation + poor drainage)
                ff_prob = 0.0
                if rain > 5:
                    ff_prob += 0.30 * min(1.0, rain / 65)
                if elev < 10:
                    ff_prob += 0.25 * min(1.0, (10 - elev) / 10)
                if soil > 60:
                    ff_prob += 0.15 * min(1.0, (soil - 60) / 40)
                if is_high_tide and tide_height > 3:
                    ff_prob += 0.15 * min(1.0, (tide_height - 3) / 3)
                if drainage_dist > 2000:
                    ff_prob += 0.10 * min(1.0, (drainage_dist - 2000) / 8000)
                if slope < 3:
                    ff_prob += 0.05 * min(1.0, (3 - slope) / 3)
                ff_prob = min(1.0, ff_prob)

                # Severity score (weighted combination)
                severity = ts_prob * 30 + cb_prob * 40 + ff_prob * 30

                # Depth estimate
                excess = max(0, rain - 17.5)
                retention = 1.0 + (max(0, 7 - elev) / 7) * 1.5
                depth = excess * 0.15 * retention * (0.6 + 0.4 * soil / 100)

                all_targets.append([ts_prob, cb_prob, ff_prob, severity, depth])

        self.feature_matrix = np.array(all_features, dtype=np.float32)
        self.target_matrix = np.array(all_targets, dtype=np.float32)

        # Separate targets for convenience
        self.depth_targets = self.target_matrix[:, 4].copy()
        self.severity_targets = self.target_matrix[:, 3].copy()
        self.hazard_probs = self.target_matrix[:, :3].copy()  # ts, cb, ff

        logger.info(
            f"Feature matrix: {self.feature_matrix.shape}, "
            f"Target matrix: {self.target_matrix.shape} (ts, cb, ff, severity, depth)"
        )

        return {
            "n_timesteps": n_timesteps,
            "n_cells": GRID_CELLS,
            "n_samples": len(self.feature_matrix),
            "n_features": self.feature_matrix.shape[1],
            "target_distribution": {
                int(k): int(v) for k, v in zip(*np.unique(self.target_matrix, return_counts=True))
            },
        }

    def train_all(self, epochs: int = 200, lr: float = 0.001) -> Dict[str, Any]:
        """Train all models and return training reports."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for training")

        if self.feature_matrix is None:
            self.load_data()

        results = {}

        # Split data (no stratify for multi-dimensional targets)
        X_train, X_test, y_train, y_test = train_test_split(
            self.feature_matrix, self.target_matrix,
            test_size=0.2, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=0.15, random_state=42
        )

        # Depth regression split
        X_tr, X_te, d_train, d_test = train_test_split(
            self.feature_matrix, self.depth_targets,
            test_size=0.2, random_state=42
        )
        X_tr, X_va, d_train, d_val = train_test_split(
            X_tr, d_train,
            test_size=0.15, random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        # Save scaler
        import pickle
        with open(os.path.join(self.model_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(self.scaler, f)

        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        # ─── Train Multi-Hazard Predictor (Module 4) ───
        logger.info("Training Multi-Hazard Predictor...")
        results["multi_hazard"] = self._train_multi_hazard(
            X_train_scaled, y_train, X_val_scaled, y_val, epochs, lr
        )

        # ─── Train Nowcaster (Module 3) ───
        logger.info("Training Nowcaster...")
        results["nowcaster"] = self._train_nowcaster(epochs, lr)

        # ─── Train Storm Cell Detector (Module 1) ───
        logger.info("Training Storm Cell Detector...")
        results["storm_cell"] = self._train_storm_cell_detector(
            X_train_scaled, y_train, X_val_scaled, y_val, epochs, lr
        )

        # ─── Train Crowd Report NLP (Module 9) ───
        logger.info("Training Crowd Report NLP...")
        results["crowd_nlp"] = self._train_crowd_nlp(epochs)

        # ─── Train Trust Scorer (Module 7) ───
        logger.info("Training Trust Scorer...")
        results["trust_scorer"] = self._train_trust_scorer(
            X_train_scaled, X_val_scaled, epochs, lr
        )

        # ─── Train XAI Attention Layer (Module 8) ───
        logger.info("Training XAI Attention Layer...")
        results["xai"] = self._train_xai(
            X_train_scaled, y_train, epochs, lr
        )

        # ─── Train Flood Depth Estimator (Module 6) ───
        logger.info("Training Flood Depth Estimator...")
        results["flood_depth"] = self._train_flood_depth(
            X_train_scaled, d_train, X_val_scaled, d_val, epochs, lr
        )

        # ─── Calibrate Conformal Predictor ───
        logger.info("Calibrating Conformal Predictor...")
        results["conformal"] = self._calibrate_conformal(
            X_train_scaled, y_train, X_val_scaled, y_val
        )

        # ─── Compute SHAP values ───
        logger.info("Computing SHAP values...")
        results["shap"] = self._compute_shap(X_train_scaled, y_train)

        # ─── Evaluate on test set ───
        logger.info("Evaluating on test set...")
        results["test_evaluation"] = self._evaluate_all(X_test_scaled, y_test, d_test)

        # Save training metadata
        metadata = {
            "training_date": str(np.datetime64("now")),
            "n_timesteps": len(self.raw_data["timesteps"]),
            "n_features": len(FEATURE_COLS),
            "feature_columns": FEATURE_COLS,
            "results": {k: {kk: vv for kk, vv in v.items() if not isinstance(vv, np.ndarray)}
                       for k, v in results.items()},
        }

        with open(os.path.join(self.model_dir, "training_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info("Training complete! All checkpoints saved.")
        return results

    def _train_multi_hazard(
        self, X_train, y_train, X_val, y_val, epochs, lr
    ) -> Dict[str, Any]:
        """Train the multi-hazard risk predictor on all 5 outputs:
        thunderstorm_prob, cloudburst_prob, flash_flood_prob, severity_score, flood_depth_cm
        """
        from app.services.ai.model_architectures import MultiHazardPredictor

        model = MultiHazardPredictor(in_features=len(FEATURE_COLS)).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.MSELoss()

        # y_train columns: [ts_prob, cb_prob, ff_prob, severity, depth]
        X_tr = torch.FloatTensor(X_train).to(self.device)
        y_tr = torch.FloatTensor(y_train).to(self.device)
        X_v = torch.FloatTensor(X_val).to(self.device)
        y_v = torch.FloatTensor(y_val).to(self.device)

        best_val_loss = float("inf")
        best_state = None
        patience = 30
        patience_counter = 0

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()

            output = model(X_tr)
            # Multi-task loss: probabilities (0-1) + severity (0-100) + depth (cm)
            loss_ts = criterion(output["thunderstorm_prob"].clamp(0, 1), y_tr[:, 0])
            loss_cb = criterion(output["cloudburst_prob"].clamp(0, 1), y_tr[:, 1])
            loss_ff = criterion(output["flash_flood_prob"].clamp(0, 1), y_tr[:, 2])
            loss_sev = criterion(output["severity_score"], y_tr[:, 3])
            loss_depth = criterion(output["flood_depth_cm"].clamp(min=0), y_tr[:, 4])

            # Weighted loss: probabilities are primary, severity and depth secondary
            loss = 0.30 * loss_ts + 0.30 * loss_cb + 0.30 * loss_ff + 0.05 * loss_sev + 0.05 * loss_depth

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_output = model(X_v)
                val_loss_ts = criterion(val_output["thunderstorm_prob"].clamp(0, 1), y_v[:, 0])
                val_loss_cb = criterion(val_output["cloudburst_prob"].clamp(0, 1), y_v[:, 1])
                val_loss_ff = criterion(val_output["flash_flood_prob"].clamp(0, 1), y_v[:, 2])
                val_loss = 0.30 * val_loss_ts + 0.30 * val_loss_cb + 0.30 * val_loss_ff

                # Compute accuracy: dominant hazard matches
                val_ts = val_output["thunderstorm_prob"]
                val_cb = val_output["cloudburst_prob"]
                val_ff = val_output["flash_flood_prob"]
                val_dominant = torch.stack([val_ts, val_cb, val_ff], dim=1).argmax(dim=1)
                true_dominant = y_v[:, :3].argmax(dim=1)
                val_acc = (val_dominant == true_dominant).float().mean().item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"  Early stopping at epoch {epoch+1}")
                break

            if (epoch + 1) % 50 == 0:
                logger.info(f"  Epoch {epoch+1}/{epochs} — loss: {loss.item():.4f}, val_acc: {val_acc:.3f}")

        # Load best model
        if best_state:
            model.load_state_dict(best_state)

        # Save checkpoint
        torch.save(model.state_dict(), os.path.join(self.model_dir, "multi_hazard_predictor.pt"))

        # Final evaluation
        model.eval()
        with torch.no_grad():
            test_output = model(X_v)
            val_ts = test_output["thunderstorm_prob"]
            val_cb = test_output["cloudburst_prob"]
            val_ff = test_output["flash_flood_prob"]
            val_dominant = torch.stack([val_ts, val_cb, val_ff], dim=1).argmax(dim=1)
            true_dominant = y_v[:, :3].argmax(dim=1)
            final_acc = (val_dominant == true_dominant).float().mean().item()

        return {
            "final_val_accuracy": round(final_acc, 4),
            "best_val_loss": round(best_val_loss.item(), 4),
            "epochs_trained": epoch + 1,
            "checkpoint": "multi_hazard_predictor.pt",
        }

    def _train_nowcaster(self, epochs, lr) -> Dict[str, Any]:
        """Train the spatiotemporal nowcaster."""
        from app.services.ai.model_architectures import SpatiotemporalNowcaster

        timesteps = self.raw_data["timesteps"]
        window_size = 6
        forecast_horizon = 6

        # Build sequences: input = 6 timesteps, target = next 6 timesteps
        sequences = []
        targets = []

        for i in range(len(timesteps) - window_size - forecast_horizon + 1):
            inp_features = []
            tgt_features = []

            for t in range(i, i + window_size):
                ts_features = []
                for cell in timesteps[t]["features"]:
                    ts_features.append([cell.get(col, 0) for col in FEATURE_COLS[:8]])
                inp_features.append(ts_features)

            for t in range(i + window_size, i + window_size + forecast_horizon):
                ts_features = []
                for cell in timesteps[t]["features"]:
                    ts_features.append([cell.get(col, 0) for col in FEATURE_COLS[:8]])
                tgt_features.append(ts_features)

            sequences.append(inp_features)
            targets.append(tgt_features)

        if not sequences:
            return {"note": "Insufficient data for nowcaster training", "skipped": True}

        X = np.array(sequences, dtype=np.float32)
        y = np.array(targets, dtype=np.float32)

        # Reshape for ConvLSTM: (batch, time, channels, height, width)
        B = X.shape[0]
        C = 8  # number of feature channels
        X_tensor = torch.FloatTensor(X).view(B, window_size, C, GRID_ROWS, GRID_COLS).to(self.device)
        y_tensor = torch.FloatTensor(y).view(B, forecast_horizon, C, GRID_ROWS, GRID_COLS).to(self.device)

        model = SpatiotemporalNowcaster(
            in_channels=C, hidden_dim=64, forecast_horizon=forecast_horizon
        ).to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=lr * 0.5)
        criterion = nn.MSELoss()

        # Split
        n_train = int(0.8 * B)
        X_train, X_val = X_tensor[:n_train], X_tensor[n_train:]
        y_train, y_val = y_tensor[:n_train], y_tensor[n_train:]

        best_val_loss = float("inf")
        best_state = None

        for epoch in range(min(epochs, 150)):
            model.train()
            optimizer.zero_grad()

            output = model(X_train)
            loss = criterion(output["forecast"], y_train)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_output = model(X_val)
                val_loss = criterion(val_output["forecast"], y_val)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            if (epoch + 1) % 30 == 0:
                logger.info(f"  Nowcaster epoch {epoch+1} — loss: {loss.item():.4f}, val: {val_loss.item():.4f}")

        if best_state:
            model.load_state_dict(best_state)

        torch.save(model.state_dict(), os.path.join(self.model_dir, "spatiotemporal_nowcaster.pt"))

        return {
            "best_val_loss": round(best_val_loss.item(), 6),
            "n_sequences": len(sequences),
            "checkpoint": "spatiotemporal_nowcaster.pt",
        }

    def _train_storm_cell_detector(
        self, X_train, y_train, X_val, y_val, epochs, lr
    ) -> Dict[str, Any]:
        """
        Train storm cell detection model (Module 1).

        Labels: cells with CAPE > 1500 AND CTT < -40 AND rain > 10 are storm cells.
        Architecture: Simple binary classifier.
        """
        import torch.nn as nn

        # Generate storm cell labels from ACTUAL meteorological criteria
        # Storm cells: high CAPE + cold cloud tops + significant rainfall
        storm_labels_train = []
        for i in range(len(X_train)):
            # FEATURE_COLS indices: 0=rainfall_1h, 1=rainfall_3h, 2=rainfall_24h,
            # 5=cape, 8=cloud_top_temp, 9=ctt_drop_rate, 10=wind_speed
            cape = X_train[i][5] * 3000 if len(X_train[i]) > 5 else 0  # denormalize
            rain = X_train[i][0] * 80 if len(X_train[i]) > 0 else 0
            ctt = X_train[i][8] * 60 + 10 if len(X_train[i]) > 8 else 0  # denormalize
            wind = X_train[i][10] * 40 if len(X_train[i]) > 10 else 0
            # Storm = CAPE>1000 AND CTT<-30 AND rain>5 OR wind>25
            is_storm = 1.0 if (cape > 1000 and ctt < -30 and rain > 5) or (wind > 25 and rain > 10) else 0.0
            storm_labels_train.append(is_storm)
        storm_labels_train = torch.tensor(storm_labels_train, dtype=torch.float32).to(self.device)
        n_storms = int(storm_labels_train.sum().item())
        logger.info(f"Storm cell labels: {n_storms}/{len(X_train)} storm cells ({100*n_storms/len(X_train):.1f}%)")

        storm_labels_val = []
        for i in range(len(X_val)):
            cape = X_val[i][5] * 3000 if len(X_val[i]) > 5 else 0
            rain = X_val[i][0] * 80 if len(X_val[i]) > 0 else 0
            ctt = X_val[i][8] * 60 + 10 if len(X_val[i]) > 8 else 0
            wind = X_val[i][10] * 40 if len(X_val[i]) > 10 else 0
            is_storm = 1.0 if (cape > 1000 and ctt < -30 and rain > 5) or (wind > 25 and rain > 10) else 0.0
            storm_labels_val.append(is_storm)
        storm_labels_val = torch.tensor(storm_labels_val, dtype=torch.float32).to(self.device)

        # Binary classifier: input features → storm probability
        model = nn.Sequential(
            nn.Linear(len(FEATURE_COLS), 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        ).to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.BCELoss()

        # Convert numpy to tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)

        best_val_loss = float("inf")
        best_state = None

        for epoch in range(min(epochs, 100)):
            model.train()
            optimizer.zero_grad()
            output = model(X_train_t).squeeze()
            loss = criterion(output, storm_labels_train)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_output = model(X_val_t).squeeze()
                val_loss = criterion(val_output, storm_labels_val)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if best_state:
            model.load_state_dict(best_state)

        torch.save(model.state_dict(), os.path.join(self.model_dir, "storm_cell_detector.pt"))

        return {
            "best_val_loss": round(best_val_loss.item(), 6),
            "checkpoint": "storm_cell_detector.pt",
        }

    def _train_crowd_nlp(self, epochs: int = 100) -> Dict[str, Any]:
        """
        Train crowd report NLP classifier (Module 9).

        Uses synthetic labeled crowd reports for training.
        Architecture: Embedding + Linear classifier.
        """
        import torch.nn as nn
        from app.services.ai.data_loader import FLOOD_VOCAB, tokenize_crowd_report

        # Synthetic labeled training data
        train_reports = [
            ("heavy rain flooding streets waterlogged help", 1),
            ("emergency water rising fast rescue needed", 1),
            ("flood in mumbai roads submerged cars stuck", 1),
            ("paani badh raha hai bachao", 1),
            ("waterlogged area near bandra underpass", 1),
            ("deep water on road cannot drive", 1),
            ("submerged vehicles need rescue", 1),
            ("overflow drain near andheri", 1),
            ("rain stopped water going down", 0),
            ("roads clear traffic normal", 0),
            ("weather fine no issues", 0),
            ("thik hai sab sukha hai", 0),
            ("dry roads no water", 0),
            ("normal day no flooding", 0),
            ("clear weather good conditions", 0),
            ("road repair work happening", 2),
            ("match tonight at stadium", 2),
            ("sale offer discount shop", 2),
            ("movie release this friday", 2),
            ("traffic jam due to construction", 2),
        ]

        # Tokenize
        max_len = 128
        X_list = []
        y_list = []
        for text, label in train_reports:
            tokens = tokenize_crowd_report(text, max_len)
            X_list.append(tokens)
            y_list.append(label)

        X = torch.tensor(X_list, dtype=torch.long).to(self.device)
        y = torch.tensor(y_list, dtype=torch.long).to(self.device)

        # Simple classifier: Embedding → Global Max Pool → Linear
        vocab_size = max(FLOOD_VOCAB.values()) + 10
        embed_dim = 32

        class CrowdNLPModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
                self.pool = nn.AdaptiveMaxPool1d(1)
                self.classifier = nn.Sequential(
                    nn.Linear(embed_dim, 32),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(32, 3),
                )
            def forward(self, x):
                emb = self.embedding(x)  # (B, seq_len, embed_dim)
                emb = emb.permute(0, 2, 1)  # (B, embed_dim, seq_len)
                pooled = self.pool(emb).squeeze(-1)  # (B, embed_dim)
                return self.classifier(pooled)

        model = CrowdNLPModel().to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(min(epochs, 200)):
            model.train()
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        torch.save(model.state_dict(), os.path.join(self.model_dir, "crowd_nlp_classifier.pt"))

        return {
            "checkpoint": "crowd_nlp_classifier.pt",
            "n_reports": len(train_reports),
        }

    def _train_trust_scorer(self, X_train, X_val, epochs, lr) -> Dict[str, Any]:
        """Train the forecast trust scorer (autoencoder-based)."""
        from app.services.ai.model_architectures import ForecastTrustScorer

        model = ForecastTrustScorer(input_dim=len(FEATURE_COLS)).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        mse_crit = nn.MSELoss()

        X_tr = torch.FloatTensor(X_train).to(self.device)
        X_v = torch.FloatTensor(X_val).to(self.device)

        for epoch in range(min(epochs, 100)):
            model.train()
            optimizer.zero_grad()

            output = model(X_tr)
            # Autoencoder reconstruction loss
            recon_loss = mse_crit(output["reconstruction_error"].unsqueeze(1).expand_as(X_tr[:, :1]), X_tr[:, :1] * 0)
            # Trust classification: use reconstruction error as signal
            trust_labels = (output["reconstruction_error"] > output["reconstruction_error"].median()).long()
            trust_loss = nn.CrossEntropyLoss()(output["trust_logits"], trust_labels)

            loss = recon_loss + trust_loss
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 25 == 0:
                logger.info(f"  Trust scorer epoch {epoch+1} — loss: {loss.item():.4f}")

        torch.save(model.state_dict(), os.path.join(self.model_dir, "forecast_trust_scorer.pt"))

        return {"checkpoint": "forecast_trust_scorer.pt"}

    def _train_xai(self, X_train, y_train, epochs, lr) -> Dict[str, Any]:
        """Train the XAI attention layer."""
        from app.services.ai.model_architectures import XAIAttentionLayer

        model = XAIAttentionLayer(num_features=len(FEATURE_COLS)).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        X_tr = torch.FloatTensor(X_train).to(self.device)

        for epoch in range(min(epochs, 80)):
            model.train()
            optimizer.zero_grad()

            output = model(X_tr)
            # Importance should sum to ~1
            importance_loss = (output["feature_importance"].sum(dim=-1) - 1.0).pow(2).mean()
            # Attention should be diverse (not collapse to one feature)
            attn_entropy = -(output["attention_weights"].clamp(1e-8) * output["attention_weights"].clamp(1e-8).log()).sum(dim=-1).mean()

            loss = importance_loss - 0.01 * attn_entropy
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 20 == 0:
                logger.info(f"  XAI epoch {epoch+1} — loss: {loss.item():.6f}")

        torch.save(model.state_dict(), os.path.join(self.model_dir, "xai_attention_layer.pt"))

        # Extract learned feature importance
        model.eval()
        with torch.no_grad():
            sample = X_tr[:100]
            output = model(sample)
            learned_importance = output["feature_importance"].mean(dim=0).cpu().numpy()

        importance_map = {col: round(float(learned_importance[i]), 4) for i, col in enumerate(FEATURE_COLS)}

        return {
            "checkpoint": "xai_attention_layer.pt",
            "learned_feature_importance": importance_map,
        }

    def _train_flood_depth(self, X_train, d_train, X_val, d_val, epochs, lr) -> Dict[str, Any]:
        """Train the flood depth estimation head."""
        import torch.nn.functional as F

        model = nn.Sequential(
            nn.Linear(len(FEATURE_COLS), 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU(),
        ).to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        X_tr = torch.FloatTensor(X_train).to(self.device)
        d_tr = torch.FloatTensor(d_train).unsqueeze(1).to(self.device)
        X_v = torch.FloatTensor(X_val).to(self.device)
        d_v = torch.FloatTensor(d_val).unsqueeze(1).to(self.device)

        best_val_loss = float("inf")
        best_state = None

        for epoch in range(min(epochs, 150)):
            model.train()
            optimizer.zero_grad()

            pred = model(X_tr)
            loss = criterion(pred, d_tr)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_pred = model(X_v)
                val_loss = criterion(val_pred, d_v)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            if (epoch + 1) % 30 == 0:
                logger.info(f"  Depth estimator epoch {epoch+1} — loss: {loss.item():.4f}")

        if best_state:
            model.load_state_dict(best_state)

        torch.save(model.state_dict(), os.path.join(self.model_dir, "flood_depth_estimator.pt"))

        # Evaluate MAE
        model.eval()
        with torch.no_grad():
            val_pred = model(X_v)
            mae = torch.mean(torch.abs(val_pred - d_v)).item()

        return {
            "checkpoint": "flood_depth_estimator.pt",
            "val_mae_cm": round(mae, 2),
        }

    def _calibrate_conformal(self, X_train, y_train, X_val, y_val) -> Dict[str, Any]:
        """Calibrate conformal predictor on validation set."""
        from app.services.conformal import conformal_predictor
        import pickle

        # Load the multi-hazard model to get predictions
        from app.services.ai.model_architectures import MultiHazardPredictor
        model = MultiHazardPredictor(in_features=len(FEATURE_COLS)).to(self.device)

        ckpt_path = os.path.join(self.model_dir, "multi_hazard_predictor.pt")
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=self.device, weights_only=True))

        model.eval()
        with torch.no_grad():
            X_v = torch.FloatTensor(X_val).to(self.device)
            output = model(X_v)
            y_pred = (output["severity_score"] / 100 * 3).cpu().numpy()
            y_true = y_val.astype(float)

        # Calibrate
        report = conformal_predictor.calibrate(y_true, y_pred)

        return report

    def _compute_shap(self, X_train, y_train) -> Dict[str, Any]:
        """Compute SHAP values on the trained multi-hazard model."""
        try:
            import shap
            from app.services.ai.model_architectures import MultiHazardPredictor

            model = MultiHazardPredictor(in_features=len(FEATURE_COLS)).to(self.device)
            ckpt_path = os.path.join(self.model_dir, "multi_hazard_predictor.pt")
            if os.path.exists(ckpt_path):
                model.load_state_dict(torch.load(ckpt_path, map_location=self.device, weights_only=True))

            model.eval()

            # Create a wrapper for SHAP
            def model_predict(x):
                with torch.no_grad():
                    X = torch.FloatTensor(x).to(self.device)
                    output = model(X)
                    return np.column_stack([
                        output["thunderstorm_prob"].cpu().numpy(),
                        output["cloudburst_prob"].cpu().numpy(),
                        output["flash_flood_prob"].cpu().numpy(),
                        output["severity_score"].cpu().numpy(),
                    ])

            # Use a subset for SHAP (it's slow on large datasets)
            X_sample = X_train[:min(200, len(X_train))]

            explainer = shap.KernelExplainer(model_predict, X_sample[:50])
            shap_values = explainer.shap_values(X_sample[50:100] if len(X_sample) > 50 else X_sample)
            # shap_values may be a list (one per output) or array
            if isinstance(shap_values, list):
                # Take the severity_score output (index 3)
                shap_values = shap_values[3] if len(shap_values) > 3 else shap_values[0]
            shap_values = np.array(shap_values)

            # Average absolute SHAP values per feature
            mean_shap = np.mean(np.abs(shap_values), axis=0)

            importance_map = {}
            for i, col in enumerate(FEATURE_COLS):
                importance_map[col] = round(float(mean_shap[i]), 4)

            # Normalize to sum to 1
            total = sum(importance_map.values())
            if total > 0:
                importance_map = {k: round(v / total, 4) for k, v in importance_map.items()}

            # Save
            with open(os.path.join(self.model_dir, "shap_importance.json"), "w") as f:
                json.dump(importance_map, f, indent=2)

            return {
                "method": "KernelSHAP",
                "n_samples": len(X_sample),
                "feature_importance": importance_map,
            }

        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}")
            return {"error": str(e), "method": "fallback_uniform"}

    def _evaluate_all(self, X_test, y_test, d_test) -> Dict[str, Any]:
        """Evaluate all trained models on test set."""
        from app.services.ai.model_architectures import MultiHazardPredictor

        results = {}

        # Evaluate multi-hazard
        model = MultiHazardPredictor(in_features=len(FEATURE_COLS)).to(self.device)
        ckpt_path = os.path.join(self.model_dir, "multi_hazard_predictor.pt")
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=self.device, weights_only=True))

        model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X_test).to(self.device)
            output = model(X_t)
            y_pred = (output["severity_score"] / 100 * 3).round().long().clamp(0, 3).cpu().numpy()
            y_true = y_test

            accuracy = float(np.mean(y_pred == y_true))

            # Per-class metrics
            from sklearn.metrics import classification_report, confusion_matrix
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            cm = confusion_matrix(y_true, y_pred).tolist()

            results["multi_hazard"] = {
                "accuracy": round(accuracy, 4),
                "classification_report": report,
                "confusion_matrix": cm,
            }

        # Evaluate flood depth
        try:
            depth_model = torch.load(
                os.path.join(self.model_dir, "flood_depth_estimator.pt"),
                map_location=self.device, weights_only=True
            ) if os.path.exists(os.path.join(self.model_dir, "flood_depth_estimator.pt")) else None

            if depth_model is not None:
                import torch.nn as nn
                depth_nn = nn.Sequential(
                    nn.Linear(len(FEATURE_COLS), 128), nn.ReLU(), nn.Dropout(0.2),
                    nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.ReLU(),
                ).to(self.device)
                depth_nn.load_state_dict(depth_model)

                depth_nn.eval()
                with torch.no_grad():
                    X_t = torch.FloatTensor(X_test).to(self.device)
                    d_pred = depth_nn(X_t).squeeze().cpu().numpy()
                    mae = float(np.mean(np.abs(d_pred - d_test)))
                    rmse = float(np.sqrt(np.mean((d_pred - d_test) ** 2)))

                results["flood_depth"] = {
                    "mae_cm": round(mae, 2),
                    "rmse_cm": round(rmse, 2),
                }
        except Exception as e:
            results["flood_depth"] = {"error": str(e)}

        return results


def main():
    """CLI entry point for training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train VARUNA AI models")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--model-dir", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    trainer = VARUNATrainer(data_dir=args.data_dir, model_dir=args.model_dir)
    data_info = trainer.load_data()
    print(f"\nData loaded: {json.dumps(data_info, indent=2)}")

    results = trainer.train_all(epochs=args.epochs, lr=args.lr)
    print(f"\nTraining complete: {json.dumps(results, indent=2, default=str)}")


if __name__ == "__main__":
    main()
