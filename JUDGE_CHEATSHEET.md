# VARUNA — One-Page Judge Cheat-Sheet

**What it is:** Hyper-local early warning system = digital twin of Mumbai. Fuses weather data → physics + trained multi-task AI → per-locality alerts for **3 hazards at once** (Thunderstorm / Cloudburst / Flash Flood) with 2–6 h lead time, every alert explainable, runs fully offline.

**Pilot:** 90-cell grid (10×9) over Mumbai (18.88–19.26 N, 72.78–73.00 E), 72 hourly steps of the real **26–29 Jul 2024 Mumbai deluge** (real ERA5 rain + real SRTM 30 m terrain; per-cell satellite/CAPE fields are physically-consistent reconstructions, honestly labelled).

---

## 60-Second Demo Script

1. **Room:** "Mumbai Replay is a digital twin — we replay a real Mumbai flood hour by hour, and the system warns *before* each surge."
2. **⏮ Hour 1:** "Normal — 0 alerts, green, trust 77%, all 3 hazard KPIs ~0."
3. **▶ × few:** "Soil saturating, risk climbing, first WATCH-level alerts."
4. **🔥 Peak (Hour 36):** "All at once — 81 alerts (11 Critical, 34 High), map turns red over Ghatkopar–Powai–Bhandup, nowcast: rain climbing to **134 mm/hr in 6 h**, trust drops to 66% with anomaly warning."
5. **Click alert (Ghatkopar E, 84/100):** "Every point decomposed — 40 rainfall + 22 soil + 14 terrain + 8 instability. No black box."
6. **What-If:** *"What if rain doubles in Andheri?"* → before/after risk jump. *"We rehearse the disaster before it happens."*
7. **Escape Routes (Powai):** "Best = BKC Grounds, 4 km, 6.3 min, 96% safe, max 4 cm on route."
8. **Close:** "Hybrid physics + trained multi-task AI, 3 hazards in one forward pass, fully offline. Swap in live MOSDAC/IMDAA files → same code runs real-time."

---

## Key Numbers

| Metric | Hour 1 (calm) | Hour 36 (peak) |
|---|---|---|
| Avg / Max risk | 5 / 11 | 62 / 81–84 |
| Alerts | 0 | **81** (11 Critical, 34 High) |
| Cells by hazard | 0 ⛈ / 0 🌧 / ~5 🌊 | ~0 ⛈ / **80 🌧 CB** / 9 🌊 FF |
| Max water depth | ~0 cm | ~5 cm |
| Next-6 h trend | Stable ~2 mm/hr | Rising 118→134 mm/hr (+3.9/step) |
| Trust | 77% | 66% (anomaly flagged) |
| Model agreement | 61 / 68 / 70 | 32 / 68 / 100 (cross / nowcast / storm) |
| Status | MONITORING 🟢 | CRITICAL 🔴 |

**Alert rules:** land cell only (sea/creek excluded) · risk ≥ 35 → alert · type = model's dominant hazard · severity from score (35–60 MED / 60–80 HIGH / 80+ CRIT) · old alerts deactivated each step → panel always shows *current hour only*.

**Why thunderstorm ≈ 0 at peak (be ready):** this event is a *monsoon deluge* — cloudburst signature dominates (extreme rain). Thunderstorm needs high CAPE + fast cloud-top cooling without extreme rain → appears in pre-monsoon scenarios. Model predicts all 3; the scenario picks the leader.

---

## Honesty Table (what's real, what's not)

| Component | Status |
|---|---|
| Rainfall 72 h + terrain | ✅ **Real** (ERA5, SRTM 30 m) |
| Satellite / CAPE fields per cell | ⚠️ Physically-consistent reconstruction (deterministic, realistic ranges) |
| Multi-hazard 3-head model | ✅ **Trained PyTorch** checkpoint, runs every step |
| Nowcaster (ConvLSTM+Transformer) | ✅ **Trained** (3.4 M params) + bias correction; deterministic fallback if net output flat |
| Storm detection + tracking | ✅ Trained detector + stateful tracker (IDs, direction) |
| XAI explainer | ✅ Trained attention + physics decomposition |
| Trust / conformal threshold | ✅ Trained calibrator (threshold loaded; UI honestly shows "Off" until guarantee active) |
| Physics risk score + depth | ✅ Deterministic equations (by design — the explainable backbone: 40+25+20+15 additive score) |
| What-If chatbot, evacuation routing | Rule-based over physics (fast, deterministic) — not a "missing" model |
| MOSDAC live granules | Code + credentials wired (`3RIMG_L1C_SGP` verified, 141 granules); demo keeps INSAT **simulated** — label flips to LIVE only when a real cached granule exists |

**One-liner:** *"Deep networks are genuinely trained and executed; physics stays equation-based so every alert can be decomposed — that hybrid is the innovation."*

---

## Production Path (if pushed)

- **Real satellite:** MOSDAC credentials in `.env`; real granule → UI label "LIVE" (never fakes it). Needs ~90 MB bandwidth for a full INSAT granule.
- **Real IMDAA:** registered at rds.ncmrwf.gov.in → download Jul-2024 Mumbai box NetCDF → folder → ingest. Until then genuine ERA5 is the reanalysis stream.
- **Offline claim:** fetch-once-then-serve-locally — models, 72-h dataset, physics, chatbot all on disk; no internet in the demo room.
- **Mumbai → India:** grid, model, alerts are region-agnostic; Live India mode already polls ~930 points; retrain once per climatic zone.
- **Model accuracy (~99% val):** measured on the calibrated pilot distribution (real rain/terrain + reconstructed fields + physics-rule labels) — never quote it as real-world accuracy.

**Tech stack (one line):** Python · PyTorch (multi-task MTL, ConvLSTM+Transformer, attention XAI) · FastAPI · SQLite · React · Leaflet + OpenStreetMap · SRTM/ERA5/Open-Meteo/MOSDAC-data pathway — **$0 data cost**.