/**
 * GeoMap — OSM + smooth Leaflet.heat risk layer (Mumbai pilot)
 * ==============================================================
 * - Mumbai replay: continuous heat gradient (green → yellow → orange → red)
 *   instead of discrete dots, exactly like IMD/Google-style risk maps.
 * - The Mumbai grid follows the SAME convention as the backend dataset:
 *   row-major 10×9 cells whose lat/lon are CELL CENTERS, so heat spots land
 *   exactly on the locality they name.
 * - Clickable invisible pickers keep cell selection + tooltips.
 * - A single static legend lives in MapView (bottom-left); no Leaflet
 *   legend controls are added here, so nothing can ever duplicate.
 */
import { useEffect, useRef } from 'react'
import { fmt } from '../utils/helpers'
import { isInsideIndia } from '../utils/indiaBoundary'
import { loadIndiaBoundary } from '../utils/indiaGeoJson'

// ─── Mumbai grid (10 rows × 9 cols) — same as Backend/data_pipeline ──
const M_ROWS = 10, M_COLS = 9
const M_LAT_MIN = 18.88, M_LAT_MAX = 19.26
const M_LON_MIN = 72.78, M_LON_MAX = 73.00
const M_CELL_LAT = (M_LAT_MAX - M_LAT_MIN) / M_ROWS
const M_CELL_LON = (M_LON_MAX - M_LON_MIN) / M_COLS
const M_CENTER = [19.06, 72.90]

// Locality names per cell — row-major, matches backend replay grid (C01..C90)
const MUMBAI_CELL_NAMES = [
  'Sea','Colaba','Fort','Churchgate','Marine Drive','Nariman Point','Malabar Hill','Walkeshwar','Haji Ali',
  'Sea','Grant Road','Tardeo','Bhuleshwar','Girgaon','Parel','Mahalaxmi','Byculla','Mazgaon',
  'Sea','Mumbai Central','Worli','Matunga','Sion','Wadala','Sewri','Chinchpokli','Reay Road',
  'Mahim','Dadar West','Dadar East','Kurla','Vidyavihar','Ghatkopar','BKC','Kalina','Santacruz',
  'Sea','Bandra','Bandra West','Khar','Chembur','Powai','Hiranandani','Chembur East','Navi Mumbai',
  'Juhu Beach','Juhu','Versova','Lokhandwala','Saki Naka','Ghatkopar E','Vikhroli','Kanjurmarg','Nahur',
  'Amboli','Jogeshwari','Andheri West','Andheri East','Marol','Powai Lake','Chandivali','Bhandup','Mulund',
  'Malvani','Malad West','Goregaon','Kandivali','Borivali','Deonar','Govandi','Mulund East','Thane Creek',
  'Erangal','Kandivali West','Borivali West','Dahisar','Mira Road','Thane West','Wagle Estate','Thane','Kopar Khairane',
  'Madh Island','Marve','Manori','Vasai','Nallasopara','Vashi','Sanpada','Nerul','Belapur',
]

// ─── India grid config (kept basic; focus is the Mumbai pilot) ─
const ROWS = 31, COLS = 30
const LAT_MIN = 6.0, LAT_MAX = 37.0, LON_MIN = 68.0, LON_MAX = 98.0
const CELL_LAT = (LAT_MAX - LAT_MIN) / ROWS
const CELL_LON = (LON_MAX - LON_MIN) / COLS
const INDIA_CITIES = {
  '6,10': 'Kanyakumari', '8,10': 'Kochi', '9,12': 'Bengaluru',
  '10,12': 'Hyderabad', '10,13': 'Chennai', '11,8': 'Goa',
  '12,7': 'Pune', '12,8': 'Mumbai', '13,6': 'Ahmedabad',
  '13,9': 'Bhopal', '14,16': 'Kolkata', '15,6': 'Jodhpur',
  '16,7': 'Delhi', '16,8': 'Agra', '17,7': 'Dehradun',
  '18,6': 'Amritsar', '19,19': 'Guwahati',
}

// ─── OSM tiles (free, no key) ────────────────────────────────
const TILE_URL = 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png'
const TILE_ATTR = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'

// Heat gradient stops: intensity 0–1, aligned with the risk legend
const HEAT_GRADIENT = {
  0.0: '#166534',
  0.2: '#22c55e',
  0.4: '#eab308',
  0.6: '#f97316',
  0.8: '#ef4444',
  1.0: '#991b1b',
}

function riskToIntensity(risk) {
  if (!risk || risk < 5) return 0           // below Low → invisible
  // push the low end up a little so Medium/High contrast stays visible
  return Math.min(1, Math.max(0.22, risk / 100))
}

/**
 * Is this (row, col) cell open sea rather than Mumbai land?
 * Matches the real coastline inside the pilot bbox:
 *  - column 0 (lon ≈ 72.79) is the Arabian Sea for the southern rows;
 *    only the far-north rows (Marve / Manori / Gorai islands) touch land there.
 *  - row 3 / col 1 sits in Mahim Bay.
 * Everything else — including Colaba, Mulund, Thane, Belapur — is land and must render.
 */
function isSeaCell(r, c) {
  if (c === 0 && r <= 6) return true      // open sea west of the city
  if (c === 1 && r === 3) return true     // Mahim Bay
  if (c === 8 && r === 7) return true     // Thane Creek channel
  return false
}

// Cell center used by BOTH the backend dataset and this map
function cellCenter(r, c) {
  return [
    M_LAT_MIN + (r + 0.5) * M_CELL_LAT,
    M_LON_MIN + (c + 0.5) * M_CELL_LON,
  ]
}

export default function GeoMap({ aiData, selectedCell, onSelectCell, mode }) {
  const containerRef = useRef(null)
  const mapRef = useRef(null)
  const layersRef = useRef([])

  // Init map once
  useEffect(() => {
    if (mapRef.current) return
    const el = containerRef.current
    if (!el || !window.L) return
    const L = window.L
    const map = L.map(el, {
      center: M_CENTER,
      zoom: 12,
      zoomControl: true,
      attributionControl: true,
      minZoom: 4,
      maxZoom: 19,
    })
    L.tileLayer(TILE_URL, { maxZoom: 19, attribution: TILE_ATTR }).addTo(map)
    mapRef.current = map
    setTimeout(() => map.invalidateSize(), 300)
  }, [])

  // Redraw on data/mode change
  useEffect(() => {
    const map = mapRef.current
    const L = window.L
    if (!map || !L) return
    // Clear previous layers (heat, pickers, boundary, geo-json are all layers,
    // so this fully clears — no controls are created anymore → no duplicates)
    layersRef.current.forEach(l => { try { map.removeLayer(l) } catch {} })
    layersRef.current = []

    if (mode === 'india') {
      drawIndia(map, L, aiData, layersRef.current)
      map.setView([22.5, 78.0], 5)
    } else {
      drawMumbai(map, L, aiData, selectedCell, onSelectCell, layersRef.current)
      map.setView(M_CENTER, 12)
    }
    map.invalidateSize()
  }, [aiData, selectedCell, mode])

  useEffect(() => {
    const obs = new ResizeObserver(() => mapRef.current?.invalidateSize())
    if (containerRef.current?.parentElement) obs.observe(containerRef.current.parentElement)
    return () => obs.disconnect()
  }, [])

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      <div ref={containerRef} style={{ width: '100%', height: '100%', background: '#e8eef4', borderRadius: 8 }} />
      {/* tiny corner label so the current view is always identifiable */}
      {mode !== 'india' && (
        <div style={{
          position: 'absolute', top: 8, left: '50%', transform: 'translateX(-50%)',
          background: 'rgba(13,17,23,0.85)', color: '#7dd3fc', padding: '2px 12px',
          borderRadius: 999, fontSize: 10, fontWeight: 700, letterSpacing: 1,
          zIndex: 1000, pointerEvents: 'none', whiteSpace: 'nowrap',
        }}>MUMBAI PILOT · RISK HEATMAP</div>
      )}
    </div>
  )
}

// ─── India grid view (rainfall circles — kept lightweight) ────
async function drawIndia(map, L, aiData, layers) {
  try {
    const geojson = await loadIndiaBoundary()
    layers.push(L.geoJSON(geojson, {
      style: { color: '#1e40af', weight: 2, fillColor: '#3b82f6', fillOpacity: 0.05 },
      interactive: false,
    }).addTo(map))
  } catch {}

  const cellMap = {}
  ;(aiData?.cells || []).forEach(c => { cellMap[c.cell_index] = c })

  const heat = []
  const labels = []
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const idx = r * COLS + c
      const lat = LAT_MIN + r * CELL_LAT + CELL_LAT / 2
      const lon = LON_MIN + c * CELL_LON + CELL_LON / 2
      if (!isInsideIndia(lat, lon)) continue
      const rain = (cellMap[idx] || {}).rainfall_1h_mm ?? 0
      if (rain < 1) continue
      const i = Math.min(1, Math.max(0.15, rain / 80))
      heat.push([lat, lon, i])
      const city = INDIA_CITIES[`${r},${c}`]
      if (city && rain > 3) {
        labels.push(L.marker([lat, lon], {
          icon: L.divIcon({
            className: '',
            html: `<div style="font-family:Inter,sans-serif;text-align:center;pointer-events:none;transform:translate(-50%,-50%);text-shadow:0 1px 4px rgba(255,255,255,0.9);line-height:1.2">
              <div style="font-size:11px;font-weight:700;color:#1e293b;white-space:nowrap">${city}</div>
              <div style="font-size:10px;font-weight:800;color:#c2410c;font-family:monospace;white-space:nowrap">${fmt(rain, 0)}mm</div>
            </div>`,
            iconSize: [0, 0], iconAnchor: [0, 0],
          }),
          interactive: false,
          zIndexOffset: 2000,
        }))
      }
    }
  }
  if (heat.length) {
    const hl = L.heatLayer(heat, {
      radius: 28, blur: 26, maxZoom: 20, minOpacity: 0.35, gradient: HEAT_GRADIENT,
    }).addTo(map)
    layers.push(hl)
  }
  labels.forEach(m => { m.addTo(map); layers.push(m) })
}

// ─── Mumbai pilot: smooth severity heat + invisible click pickers ─
function drawMumbai(map, L, aiData, selectedCell, onSelectCell, layers) {
  const riskMap = {}
  ;(aiData?.risk_heatmap?.heatmap || []).forEach(h => { riskMap[h.cell_index] = h })
  const floodMap = {}
  ;(aiData?.flood_depth?.node_estimates || []).forEach(n => { floodMap[n.cell_index] = n })
  const mhMap = {}
  ;(aiData?.multi_hazard?.cell_predictions || []).forEach(p => { mhMap[p.cell_index] = p })

  const heatPts = []
  const pickers = []

  for (let r = 0; r < M_ROWS; r++) {
    for (let c = 0; c < M_COLS; c++) {
      const idx = r * M_COLS + c
      if (isSeaCell(r, c)) continue
      const [lat, lon] = cellCenter(r, c)

      const mhCell = mhMap[idx]
      const riskRaw = riskMap[idx]?.pixel_risk_score ?? 0
      const mhSeverity = mhCell?.severity_score ?? 0
      const risk = riskRaw > 0 ? riskRaw : Math.min(mhSeverity, 70)
      const dominant = mhCell?.dominant_hazard || 'SAFE'
      const name = MUMBAI_CELL_NAMES[idx] || `Cell ${idx + 1}`
      const intensity = riskToIntensity(risk)

      // 1) smooth heat contribution (severity-based, IMD-map style)
      if (intensity > 0) heatPts.push([lat, lon, intensity])

      // 2) invisible click picker for selection + tooltip
      const picker = L.circleMarker([lat, lon], {
        radius: 16, fillOpacity: 0, opacity: 0, color: '#000', weight: 0, interactive: true,
      })
      picker.bindTooltip(
        `<div style="font-family:Inter,sans-serif;font-size:12px;line-height:1.5">
          <b style="font-size:13px">${name}</b> &nbsp;(${dominant})<br/>
          Risk: <b>${fmt(risk, 0)}%</b>
          ${mhCell ? `<br/>TS ${fmt(mhCell.thunderstorm_prob * 100, 0)}% · CB ${fmt(mhCell.cloudburst_prob * 100, 0)}% · FF ${fmt(mhCell.flash_flood_prob * 100, 0)}%` : ''}
          ${floodMap[idx]?.water_depth_cm ? `<br/>Depth: ${fmt(floodMap[idx].water_depth_cm, 0)} cm` : ''}
        </div>`,
        { direction: 'top', className: 'hazard-tooltip', opacity: 0.95 }
      )
      picker.on('click', () => onSelectCell(selectedCell === `C${idx + 1}` ? null : `C${idx + 1}`))
      pickers.push({ picker, idx, lat, lon, name })

      // 3) overflow markers on flood-prone nodes
      if (floodMap[idx]?.is_overflow_node || mhCell?.predicted_depth_cm > 30) {
        pickers.push({
          overflow: L.marker([lat + 0.004, lon], {
            icon: L.divIcon({
              className: '',
              html: '<div style="font-size:15px;text-shadow:0 0 8px rgba(239,68,68,0.95)">⚠️</div>',
              iconSize: [20, 20], iconAnchor: [10, 10],
            }),
            interactive: false, zIndexOffset: 3000,
          }),
        })
      }
    }
  }

  // 1) heat layer below everything else — big radius/blur so neighbouring
  //    cells blend into one continuous field (no more visible dots)
  if (heatPts.length) {
    const hl = L.heatLayer(heatPts, {
      radius: 40, blur: 34, maxZoom: 20, minOpacity: 0.5, max: 1,
      gradient: HEAT_GRADIENT,
    })
    hl.addTo(map)
    layers.push(hl)
  }

  // 2) selected-cell ring (visible outline, drawn above heat)
  if (selectedCell && /^C\d+$/.test(selectedCell)) {
    const idx = parseInt(selectedCell.slice(1), 10) - 1
    const r = Math.floor(idx / M_COLS), c = idx % M_COLS
    if (!isSeaCell(r, c)) {
      const [lat, lon] = cellCenter(r, c)
      const ring = L.circleMarker([lat, lon], {
        radius: 18, fillColor: 'rgba(255,255,255,0.35)', fillOpacity: 0.9,
        color: '#0f172a', weight: 3, dashArray: '4 3',
      })
      ring.addTo(map)
      layers.push(ring)
    }
  }

  // 3) invisible pickers + overflow icons on top
  pickers.forEach(p => {
    const layer = p.picker || p.overflow
    layer.addTo(map)
    layers.push(layer)
  })
}
