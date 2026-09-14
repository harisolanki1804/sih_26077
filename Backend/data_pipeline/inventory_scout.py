import sys
import os
import json
from datetime import datetime, timedelta


current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
backend_dir = os.path.join(root_dir, "Backend")


for path in [backend_dir, root_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)


from app.services.satellite.mosdac_fetcher import MOSDACSatelliteFetcher

def build_inventory_manifest():
    # Initialize your team's satellite fetcher class
    fetcher = MOSDACSatelliteFetcher()
    
    
    search_func = getattr(fetcher, "search_available_data", None) or getattr(fetcher, "search_granules", None) or getattr(fetcher, "search", None)
    
    if not search_func:
        print("Error: Could not automatically find the search method inside MOSDACSatelliteFetcher.")
        print(f"Available methods are: {[m for m in dir(fetcher) if not m.startswith('_')]}")
        return

    products = [
        "3DIMG_L2B_CTP", 
        "3DIMG_L2B_UTH", 
        "3DIMG_L2G_WDP", 
        "3DIMG_L2B_HEM",
        "3DIMG_L2B_CMK"
    ]
    
    start_date = datetime(2024, 7, 26, 0, 0)
    end_date = datetime(2024, 7, 29, 23, 30)
    
    manifest = []
    current_date = start_date
    
    print("Scanning satellite inventories across the 26-29 Jul 2024 window...")
    
    while current_date <= end_date:
        next_time = current_date + timedelta(minutes=30)
        
        for prod in products:
            try:
                # 1. Search with bounding box constraints (Mumbai area)
                res_with_box = search_func(
                    product_id=prod,
                    start_time=current_date,
                    end_time=next_time,
                    boundingBox=[18.9, 72.8, 19.3, 73.0]
                )
                
                # 2. Search without bounding box constraints
                res_no_box = search_func(
                    product_id=prod,
                    start_time=current_date,
                    end_time=next_time,
                    boundingBox=None
                )
                
                best_res = res_with_box if res_with_box and res_with_box.get("available") else res_no_box
                
                if best_res:
                    manifest.append({
                        "product_id": prod,
                        "timestamp": current_date.isoformat(),
                        "datasetId": best_res.get("datasetId", "N/A"),
                        "granule_id": best_res.get("granule_id", "N/A"),
                        "available": best_res.get("available", False),
                        "sizeMB": best_res.get("sizeMB", 0.0),
                        "notes": "Found within bounding box" if res_with_box.get("available") else "Global availability check"
                    })
                else:
                    manifest.append({
                        "product_id": prod,
                        "timestamp": current_date.isoformat(),
                        "datasetId": "N/A",
                        "granule_id": "N/A",
                        "available": False,
                        "sizeMB": 0.0,
                        "notes": "No data returned by server"
                    })
            except Exception as e:
                manifest.append({
                    "product_id": prod,
                    "timestamp": current_date.isoformat(),
                    "datasetId": "N/A",
                    "granule_id": "N/A",
                    "available": False,
                    "sizeMB": 0.0,
                    "notes": f"Search execution error: {str(e)}"
                })
                
        current_date = next_time
        
    output_path = os.path.join(backend_dir, "data_pipeline", "satellite_inventory_manifest.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"\nPhase 1 Complete! Manifest file successfully written to: {output_path}")

if __name__ == "__main__":
    build_inventory_manifest()