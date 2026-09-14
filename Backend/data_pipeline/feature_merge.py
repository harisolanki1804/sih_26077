import sys
import os
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
backend_dir = os.path.join(root_dir, "Backend")

for path in [backend_dir, root_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

def merge_satellite_into_features():
    target_feature_path = os.path.join(backend_dir, "data", "feature_grid_timeseries.json")
    derived_dir = os.path.join(backend_dir, "cache", "derived_satellite")
    
    # 1. Handle missing baseline file case cleanly
    if not os.path.exists(target_feature_path):
        print("Warning: feature_grid_timeseries.json missing. Generating container database.")
        os.makedirs(os.path.dirname(target_feature_path), exist_ok=True)
        with open(target_feature_path, "w") as f:
            json.dump({"timesteps": {}}, f)

    with open(target_feature_path, "r") as f:
        raw_data = json.load(f)

    # 2. Normalize data types: Force the master data container to act as a clean dictionary format
    feature_table = {"timesteps": {}}
    
    # Handle list format variations gracefully
    if isinstance(raw_data, list):
        print("Detected that feature_grid_timeseries.json is a list. Restructuring safely...")
        for entry in raw_data:
            if isinstance(entry, dict) and "timestamp" in entry:
                feature_table["timesteps"][entry["timestamp"]] = entry
    elif isinstance(raw_data, dict):
        # Support both 'timesteps' and 'timestep' key configurations safely
        if "timesteps" in raw_data and isinstance(raw_data["timesteps"], dict):
            feature_table["timesteps"] = raw_data["timesteps"]
        elif "timestep" in raw_data and isinstance(raw_data["timestep"], dict):
            feature_table["timesteps"] = raw_data["timestep"]
        elif "timestep" in raw_data and isinstance(raw_data["timestep"], list):
            print("Restructuring list under 'timestep' key into a dictionary map...")
            for entry in raw_data["timestep"]:
                if isinstance(entry, dict) and "timestamp" in entry:
                    feature_table["timesteps"][entry["timestamp"]] = entry
        else:
            # Fallback if top level is dict but structure is flat timestamps
            for k, v in raw_data.items():
                if isinstance(v, dict) and "cells" in v:
                    feature_table["timesteps"][k] = v

    # 3. Read calculated metrics maps from Phase 3
    derived_files = {}
    if os.path.exists(derived_dir):
        for fname in os.listdir(derived_dir):
            if fname.endswith(".json"):
                # Clean filename formatting to match timestamp key strings
                ts_key = fname.replace("derived_", "").replace(".json", "")
                # Reconstruct colon formatting if needed (e.g., Z-terminated ISO dates)
                if "_" in ts_key:
                    ts_key = ts_key.replace("_", ":")
                derived_files[ts_key] = os.path.join(derived_dir, fname)

    all_target_timesteps = list(derived_files.keys()) if derived_files else ["2024-07-26T00:00:00Z"]
    print(f"Processing Phase 4: Re-packing satellite indicators inside master timeseries...")

    for ts in all_target_timesteps:
        gap_fill_flag = False
        available_products = []
        
        if ts in derived_files:
            with open(derived_files[ts], "r") as df:
                sat_data = json.load(df)
            gap_fill_flag = sat_data.get("gap_filled", False)
            available_products = sat_data.get("available_products", ["CTP", "WDP", "UTH", "HEM"])
        else:
            gap_fill_flag = True
            sat_data = {
                "ctt_drop_rate_c_per_hr": [0.0] * 90,
                "wind_speed_shear": [0.0] * 90,
                "qpe_rain_rate": [0.0] * 90
            }
            available_products = []

        if ts not in feature_table["timesteps"]:
            feature_table["timesteps"][ts] = {"cells": []}

        # Inject standard structural parameters for Member 4's UI badge status lookup
        feature_table["timesteps"][ts]["source"] = "INSAT-3D"
        feature_table["timesteps"][ts]["is_real_data"] = not gap_fill_flag
        feature_table["timesteps"][ts]["available_products"] = available_products

        updated_cells = []
        for cell_idx in range(90):
            existing_cell_meta = {}
            
            # Safe boundary checks for pre-existing records
            if "cells" in feature_table["timesteps"][ts] and isinstance(feature_table["timesteps"][ts]["cells"], list):
                if cell_idx < len(feature_table["timesteps"][ts]["cells"]):
                    existing_cell_meta = feature_table["timesteps"][ts]["cells"][cell_idx]
            
            if not existing_cell_meta:
                existing_cell_meta = {"cell_id": cell_idx}

            # Map precise contract variables for Model Validation Gates & UI Components
            existing_cell_meta.update({
                "ctt_drop_rate_c_per_hr": sat_data["ctt_drop_rate_c_per_hr"][cell_idx],
                "wind_speed_shear": sat_data["wind_speed_shear"][cell_idx],
                "qpe_rain_rate": sat_data["qpe_rain_rate"][cell_idx],
                "iwv_proxy_source": "WV-channel BT regression (stated)",
                "cape_cin_source": "IMDAA alternate fallback (stated)",
                "gap_filled": gap_fill_flag
            })
            updated_cells.append(existing_cell_meta)

        feature_table["timesteps"][ts]["cells"] = updated_cells

    # 4. Save out file matching explicit contract shape specifications
    # Ensure uniform 'timesteps' dictionary key is written to file
    output_payload = {"timesteps": feature_table["timesteps"]}
    
    with open(target_feature_path, "w") as f:
        json.dump(output_payload, f, indent=2)

    print(f"Phase 4 Complete! Unified satellite features merged cleanly into: {target_feature_path}")

if __name__ == "__main__":
    merge_satellite_into_features()