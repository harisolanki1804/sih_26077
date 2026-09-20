Top 4 products for the replay (all small, half-hourly, INSAT-3D 3DIMG_*):
1.	3DIMG_L2B_CTP — Cloud Top Properties → CTT
2.	3DIMG_L2B_UTH — Upper Tropospheric Humidity → moisture/IWV proxy
3.	3DIMG_L2G_WDP — Wind Derived Product → winds, shear, divergence/convergence (critical)
4.	3DIMG_L2B_HEM — Hydro-Estimator → QPE rain rate (optional cross-check)
Plus one optional small mask: 3DIMG_L2B_CMK. Skip the ~90 MB L1C_SGP imagery entirely unless you want raw-band demo shots for a short peak window.

Member 1 — Satellite data (owns Backend/app/services/satellite/, data_pipeline/, cache/satellite_cache/)
Phase 1 — Search-only inventory (no download yet) Run search_available_data against all four product IDs for the 26–29 Jul 2024 window, with and without boundingBox. Write Backend/data_pipeline/satellite_inventory_manifest.json: per product per half-hour, datasetId, granule id, available, sizeMB, notes. This proves coverage before anyone downloads.
Phase 2 — Download + decode + per-cell extract (only the 4 confirmed products) For each product that exists in the manifest, download granules for the window, decode with h5py, sample the 90 Mumbai grid centres (the fetcher already has nearest-neighbour sampling — reuse it). Cache decoded granules as JSON under Backend/cache/satellite_cache/. Priority order: CTP → WDP → UTH → HEM → (optional CMK).
Phase 3 — Temporal derivation
•	CTT cooling rate: consecutive CTP frames → ctt_drop_rate_c_per_hr
•	Winds/shear/convergence: read directly from WDP fields — no more fake is_coastal heuristic
•	Cloud mask: use CMK to restrict cooling-rate to cloudy pixels
•	QPE: HEM rain rate
Phase 4 — Merge into feature table Write per-cell, per-timestep satellite values into the same shape Backend/data/feature_grid_timeseries.json already uses. Where a product is missing for a timestamp, set gap_filled: true (Member 4's satellite badge reads this). IWV stays WV-channel BT regression (state it); CAPE/CIN come from IMDAA if it lands, else synthetic (state it).
Deliverable: real satellite values in the feature table + the inventory manifest + the fixed product table + a working decode path.


Member 2 — Backend, API, evaluation (owns Backend/app/api/v1/endpoints/, schemas, evaluation/, services/conformal.py, replay/alert/routing services)
Phase 1 — Fix the two broken endpoints + publish the contract Fix the schema mismatches:
•	GET /api/v1/ai/inference/{ts} — source_contributions required but never produced. Either add the field to the producer or relax the schema.
•	POST /api/v1/ai/crowd-report — returns wrong shape, schema wants confirm_score/deny_score/unrelated_score/actionable. Align one with the other.
Then publish the contract: one short doc (in README.md, RUN_GUIDE.md, or a new Backend/docs/ file) listing the canonical feature column names and the exact JSON shapes the /features/latest, /ai/inference/{ts}, replay, alerts, and satellite endpoints return, plus which fields are real vs synthetic.
Phase 2 — Keep endpoints live as Member 1 feeds real data When Member 1's satellite values land in the feature table with the canonical column names, the existing /features/latest, replay, alerts, physics, scenarios endpoints should start returning real satellite fields automatically. Verify each one still returns 200. If a new field matters to the dashboard, Member 2 adds it to the endpoint response — Member 1 does not edit endpoint files.
Phase 3 — Evaluate on the frozen split (the credibility core) Own Backend/evaluation/run_evaluation.py and the PPT metrics files (PPT_METRICS.txt, PPT_READY_METRICS.txt, evaluation_results.json, nowcast_accuracy.json). Produce, on the held-out time-based split:
•	POD / FAR / CSI per hazard head (flood, lightning, wind, heat, air quality) at 2h, 4h, 6h lead times — the docx headline metric, currently missing
•	RMSE / MAE / R² per head per lead time (extend nowcast_accuracy.json)
•	Brier score + reliability per head; conformal coverage per head (already done at 90.2% aggregate — break it down)
•	Operational: lead time to first alert, false-alarm rate per week, evacuation detour overhead (already in find_safe_path return)
Freeze the resulting metrics file as a regression baseline.
Deliverable: fixed endpoints + published contract + the four-metric-family evaluation file, frozen as baseline.


Member 3 — Models, physics, data prep (owns Backend/app/services/ai/, services/physics/, services/risk_model.py, services/explainability.py, data_pipeline/ml_data_prep.py)
Phase 1 — Freeze the feature contract + time-based split
•	Write the one authoritative feature column list (in data_loader.py or a small feature_contract.py), matching exactly the names Member 2 published and Member 1's satellite producer now writes. Every column is either already-real, about-to-be-real (Phase 2), or explicitly declared synthetic-with-journal.
•	Add a validation gate that fails loudly if a declared feature is missing or zero-variance across a window — hard failure, not silent 0.0.
•	Replace the random 70/15/15 split in ml_data_prep.py with a time-based split on the 72-timestep axis (e.g. train 1–50, calibrate 51–62, test 63–72). Write the boundaries into the contract doc. This is what makes every metric honest.
Phase 2 — Define the residual target The docx says the 3-head model trains on the residual over the physics baseline. The physics baseline is already done and faithful (pinn_risk_model.py: SCS-CN S=(25400/CN)-254, Ia=0.2S; Manning V=(1/n)·R^(2/3)·S^(1/2); plus risk_model.py additive score + DEM depth heuristic). Member 3 makes the trainer compute target = y_true - physics_baseline(cell, timestep) and learn that residual. This is the biggest correctness item on Member 3's side.
Phase 3 — Retrain on the frozen split, recalibrate conformal Retrain multi_hazard_predictor.pt (currently val acc 96.9%) on residuals over the time-based split. The ConvLSTM+Transformer nowcaster (spatiotemporal_nowcaster.pt) can stay as the stretch goal but should also move to residual targets. After retrain, Member 2 recalibrates conformal.py (already calibrated, threshold 0.3057) on the new held-out set.
Deliverable: frozen feature contract + time-based split boundaries + residual-trained models + recalibratable conformal.


Member 4 — Frontend/UI (sole owner of all Frontend/)
Phase 1 — Build shells against mocked JSON (no backend needed)
•	Citizen-report form: text area + area selector + submit, posts to POST /api/v1/ai/crowd-report, shows result, lists recent reports. Build against a mocked crowd-report-response.json first; wire the fixed endpoint once Member 2's done.
•	Satellite status badge/panel: shows LIVE / PARTIAL / FALLBACK for the current replay step, reading source, is_real_data, available_products. Build against a mocked satellite snapshot JSON first.
•	Skill/metrics chart: container for POD/FAR/CSI per head (Member 2's Phase 3 output). Build the chart slot now against a mocked metrics JSON so it exists when numbers arrive.
•	Anything else touching an API shape: Member 4 reads the contract Member 2 publishes and builds against it. Member 4 does not edit any backend file to make the UI work — if a new shape is needed, Member 4 asks Member 2 to add it.
Phase 2 — Wire in real data as phases land
•	Satellite badge reflects Member 1's real source/is_real_data/available_products.
•	Skill chart fills with Member 2's metrics.
•	Citizen-report form posts to the now-fixed endpoint.
•	Relabel every place that claims "live MOSDAC CTT/IWV/CAPE" to match what is actually real at demo time: CTT from CTP (real), UTH real, QPE real, winds/convergence/shear real from WDP, IWV from WV regression (state it), CAPE/CIN from IMDAA-or-synthetic (state it). Never label a reconstructed value as live.
Deliverable: citizen-report form, satellite badge, skill chart slot, all wired to real endpoints and honestly labeled.

How the phases run in parallel
Day 1 (everyone parallel):
  Member 1 — search-only inventory manifest (no download)
  Member 2 — fix 2 endpoints + publish contract doc
  Member 3 — freeze feature columns + time-based split + residual target definition
  Member 4 — UI shells against mocked JSON

Day 2 (Member 1 leads, others unblocked after Day 1 handoffs):
  Member 1 — download + decode + derive + merge 4 products into feature table
  Member 3 — starts retraining on residuals once feature table + split are frozen
  Member 2 — verifies endpoints return real satellite fields; builds evaluation harness against frozen split
  Member 4 — wires satellite badge as Member 1's data lands

Day 3 (Member 3 + Member 2):
  Member 3 — finishes residual retrain
  Member 2 — recalibrates conformal + produces POD/FAR/CSI per head at 2/4/6h + writes frozen metrics file
  Member 4 — wires skill chart + citizen-report form to real endpoints

Day 4 (Member 4 + Member 2 review):
  Member 4 — honest labeling pass across the dashboard
  Member 2 — final review that every claim matches what's actually real

Handoffs (the only serial dependencies):
•	Member 1's inventory manifest → Member 1's download (same person, same day)
•	Member 2's contract doc → Member 1 writes canonical column names, Member 3 freezes them, Member 4 builds against them
•	Member 1's feature table with canonical names → Member 3 retrains, Member 2 evaluates
•	Member 3's retrained model → Member 2 recalibrates + produces final metrics
Each member's "done" signal:
•	Member 1: 4 real products decoded, per-cell, in the feature table, with gap_filled flags; inventory manifest written; product table fixed; h5py in requirements.
•	Member 2: 2 endpoints fixed and returning 200 with correct shapes; contract doc published; POD/FAR/CSI per head at 2/4/6h + the other 3 metric families on the held-out split; metrics file frozen.
•	Member 3: feature contract frozen with validation gate; time-based split boundaries written; models retrained on physics residuals over that split.
•	Member 4: citizen-report form, satellite badge, skill chart all built and wired to real endpoints; no claim labeled live that isn't.
That's the whole pipeline. Hand it out as four columns and let each member work their column in parallel after the one shared prep step.

