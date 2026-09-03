/**
 * GeoMap — OpenStreetMap Grid Visualization
 * ===========================================
 * Clean OSM-based map with grid cells colored by hazard type.
 * Simple, reliable, no external API keys needed.
 */
import { useEffect, useRef } from 'react'
import { fmt } from '../utils/helpers'
import { isInsideIndia, isInsideMumbaiLand } from '../utils/indiaBoundary'
import { loadIndiaBoundary, MUMBAI_GEOJSON } from '../utils/indiaGeoJson'

// ─── India Grid configs ──────────────────────────────────────
const ROWS = 31, COLS = 30
const LAT_MIN = 6.0, LAT_MAX = 37.0, LON_MIN = 68.0, LON_MAX = 98.0
const CELL_LAT = (LAT_MAX - LAT_MIN) / ROWS
const CELL_LON = (LON_MAX - LON_MIN) / COLS

// ─── Mumbai Grid configs ─────────────────────────────────────
const M_ROWS = 10, M_COLS = 9
const M_LAT_MIN = 18.88, M_LAT_MAX = 19.26, M_LON_MIN = 72.78, M_LON_MAX = 73.00
const M_CELL_LAT = (M_LAT_MAX - M_LAT_MIN) / M_ROWS
const M_CELL_LON = (M_LON_MAX - M_LON_MIN) / M_COLS

// ─── Mumbai locality coordinates ─────────────────────────────
const MUMBAI_LOCALITIES = [
  { name: 'Colaba', lat: 18.9210, lon: 72.8147 },
  { name: 'Fort', lat: 18.9380, lon: 72.8355 },
  { name: 'Churchgate', lat: 18.9320, lon: 72.8267 },
  { name: 'Marine Drive', lat: 18.9430, lon: 72.8230 },
  { name: 'Nariman Pt', lat: 18.9310, lon: 72.8180 },
  { name: 'Malabar Hill', lat: 18.9540, lon: 72.7950 },
  { name: 'Walkeshwar', lat: 18.9500, lon: 72.7970 },
  { name: 'Haji Ali', lat: 18.9820, lon: 72.8100 },
  { name: 'Parel', lat: 18.9930, lon: 72.8410 },
  { name: 'Grant Road', lat: 18.9620, lon: 72.8120 },
  { name: 'Mumbai Central', lat: 18.9693, lon: 72.8212 },
  { name: 'Worli', lat: 19.0010, lon: 72.8140 },
  { name: 'Prabhadevi', lat: 18.9920, lon: 72.8350 },
  { name: 'Matunga', lat: 19.0180, lon: 72.8520 },
  { name: 'Sion', lat: 19.0430, lon: 72.8620 },
  { name: 'Wadala', lat: 19.0150, lon: 72.8560 },
  { name: 'Sewri', lat: 19.0010, lon: 72.8560 },
  { name: 'Chinchpokli', lat: 18.9960, lon: 72.8610 },
  { name: 'Mahim', lat: 19.0370, lon: 72.8400 },
  { name: 'Dadar', lat: 19.0178, lon: 72.8478 },
  { name: 'Lower Parel', lat: 19.0070, lon: 72.8340 },
  { name: 'Elphinstone', lat: 19.0010, lon: 72.8390 },
  { name: 'Kurla', lat: 19.0720, lon: 72.8790 },
  { name: 'Vidyavihar', lat: 19.0740, lon: 72.8910 },
  { name: 'Ghatkopar', lat: 19.0860, lon: 72.9080 },
  { name: 'BKC', lat: 19.0530, lon: 72.8620 },
  { name: 'Kalina', lat: 19.0670, lon: 72.8790 },
  { name: 'Bandra', lat: 19.0596, lon: 72.8295 },
  { name: 'Bandra West', lat: 19.0540, lon: 72.8250 },
  { name: 'Khar', lat: 19.0720, lon: 72.8390 },
  { name: 'Santacruz', lat: 19.0830, lon: 72.8420 },
  { name: 'Vile Parle', lat: 19.0970, lon: 72.8470 },
  { name: 'Andheri East', lat: 19.1144, lon: 72.8679 },
  { name: 'Saki Naka', lat: 19.1090, lon: 72.8840 },
  { name: 'Chandivali', lat: 19.1110, lon: 72.8950 },
  { name: 'Powai', lat: 19.1240, lon: 72.9087 },
  { name: 'Juhu Beach', lat: 19.0940, lon: 72.8160 },
  { name: 'Juhu', lat: 19.1075, lon: 72.8263 },
  { name: 'Versova', lat: 19.1230, lon: 72.8150 },
  { name: 'Lokhandwala', lat: 19.1190, lon: 72.8330 },
  { name: 'Marol', lat: 19.1200, lon: 72.8620 },
  { name: 'MIDC Andheri', lat: 19.1220, lon: 72.8760 },
  { name: 'Sahar', lat: 19.0890, lon: 72.8900 },
  { name: 'Vikhroli', lat: 19.1090, lon: 72.9270 },
  { name: 'Bhandup', lat: 19.1470, lon: 72.9370 },
  { name: 'Amboli', lat: 19.1350, lon: 72.8350 },
  { name: 'Jogeshwari', lat: 19.1420, lon: 72.8480 },
  { name: 'Goregaon', lat: 19.1660, lon: 72.8520 },
  { name: 'D.N. Nagar', lat: 19.1310, lon: 72.8350 },
  { name: 'Malad', lat: 19.1870, lon: 72.8460 },
  { name: 'Kandivali', lat: 19.2040, lon: 72.8520 },
  { name: 'Borivali', lat: 19.2307, lon: 72.8567 },
  { name: 'Dahisar', lat: 19.2500, lon: 72.8600 },
  { name: 'Kanjurmarg', lat: 19.1170, lon: 72.9320 },
  { name: 'Malvani', lat: 19.1890, lon: 72.8260 },
  { name: 'Malad West', lat: 19.1828, lon: 72.8402 },
  { name: 'Goregaon E', lat: 19.1650, lon: 72.8650 },
  { name: 'Aarey Colony', lat: 19.1400, lon: 72.8740 },
  { name: 'Mindspace', lat: 19.1100, lon: 72.8870 },
  { name: 'Chembur E', lat: 19.0610, lon: 72.8980 },
  { name: 'Vikhroli E', lat: 19.1100, lon: 72.9370 },
  { name: 'Nahur', lat: 19.1230, lon: 72.9540 },
  { name: 'Thane', lat: 19.1978, lon: 72.9678 },
  { name: 'Erangal', lat: 19.2060, lon: 72.8130 },
  { name: 'Kandivali W', lat: 19.2100, lon: 72.8380 },
  { name: 'Borivali W', lat: 19.2330, lon: 72.8370 },
  { name: 'Magathane', lat: 19.2150, lon: 72.8730 },
  { name: 'Mulund W', lat: 19.1690, lon: 72.9560 },
  { name: 'Mulund E', lat: 19.1690, lon: 72.9710 },
  { name: 'Wagle Estate', lat: 19.1930, lon: 72.9610 },
  { name: 'Thane City', lat: 19.2190, lon: 72.9790 },
  { name: 'Thane Creek', lat: 19.0200, lon: 72.9600 },
  { name: 'Madh Island', lat: 19.2270, lon: 72.7920 },
  { name: 'Marve', lat: 19.2340, lon: 72.8100 },
  { name: 'Manori', lat: 19.2370, lon: 72.8080 },
  { name: 'Gorai', lat: 19.2450, lon: 72.8290 },
  { name: 'Uttan', lat: 19.2600, lon: 72.8340 },
  { name: 'Mira Road', lat: 19.2830, lon: 72.8580 },
  { name: 'Thane W', lat: 19.2000, lon: 72.9550 },
  { name: 'Kopar Khairane', lat: 19.0430, lon: 73.0020 },
  { name: 'Ghansoli', lat: 19.0630, lon: 73.0090 },
  { name: 'Vasai', lat: 19.3020, lon: 72.8260 },
  { name: 'Nallasopara', lat: 19.3380, lon: 72.8350 },
  { name: 'Virar', lat: 19.3850, lon: 72.8310 },
  { name: 'Dahisar N', lat: 19.2700, lon: 72.8620 },
  { name: 'Mira Bhayander', lat: 19.2910, lon: 72.8600 },
  { name: 'Vashi', lat: 19.0635, lon: 72.9988 },
  { name: 'Sanpada', lat: 19.0550, lon: 73.0000 },
  { name: 'Nerul', lat: 19.0360, lon: 73.0130 },
  { name: 'Belapur', lat: 19.0230, lon: 73.0270 },
]

// ─── OpenStreetMap tiles (no API key needed) ─────────────────
const TILE_URL = 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png'
const TILE_ATTR = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'

// ─── City labels for India mode ───────────────────────────────
const INDIA_CITIES = {
  '6,10': 'Kanyakumari', '8,10': 'Kochi', '9,12': 'Bengaluru',
  '10,12': 'Hyderabad', '10,13': 'Chennai', '11,8': 'Goa',
  '12,7': 'Pune', '12,8': 'Mumbai', '13,6': 'Ahmedabad',
  '13,9': 'Bhopal', '14,16': 'Kolkata', '15,6': 'Jodhpur',
  '16,7': 'Delhi', '16,8': 'Agra', '17,7': 'Dehradun',
  '18,6': 'Amritsar', '19,19': 'Guwahati',
}

// ─── Color helpers ────────────────────────────────────────────
function getRiskColor(risk) {
  if (risk >= 80) return '#dc2626'
  if (risk >= 60) return '#ea580c'
  if (risk >= 35) return '#d97706'
  if (risk >= 15) return '#16a34a'
  return '#166534'
}

function getHazardColor(dominant) {
  if (dominant === 'THUNDERSTORM') return '#f59e0b'  // amber/orange
  if (dominant === 'CLOUDBURST') return '#3b82f6'    // blue
  if (dominant === 'FLASH_FLOOD') return '#ef4444'   // red
  return '#6b7280'  // gray for SAFE
}

function getHazardIcon(dominant) {
  if (dominant === 'THUNDERSTORM') return '⛈️'
  if (dominant === 'CLOUDBURST') return '🌧️'
  if (dominant === 'FLASH_FLOOD') return '🌊'
  return '✅'
}

// ─── Component ─────────────────────────────────────────────────
export default function GeoMap({ aiData, overlay, selectedCell, onSelectCell, mode }) {
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
      center: [22.5, 78.0],
      zoom: 5,
      zoomControl: true,
      attributionControl: true,
      minZoom: 4,
      maxZoom: 18,
    })

    // Add OpenStreetMap tiles
    L.tileLayer(TILE_URL, {
      maxZoom: 19,
      attribution: TILE_ATTR,
    }).addTo(map)

    mapRef.current = map
    setTimeout(() => map.invalidateSize(), 300)
  }, [])

  // Redraw on data/mode change
  useEffect(() => {
    const map = mapRef.current
    const L = window.L
    if (!map || !L) return

    // Clear previous layers
    layersRef.current.forEach(l => { try { map.removeLayer(l) } catch {} })
    layersRef.current = []

    if (mode === 'india') {
      drawIndia(map, L, aiData, layersRef.current)
      map.setView([22.5, 78.0], 5)
    } else {
      drawMumbai(map, L, aiData, selectedCell, onSelectCell, layersRef.current)
      map.setView([19.08, 72.88], 12)
    }
    map.invalidateSize()
  }, [aiData, selectedCell, mode])

  // Resize
  useEffect(() => {
    const obs = new ResizeObserver(() => mapRef.current?.invalidateSize())
    if (containerRef.current?.parentElement) obs.observe(containerRef.current.parentElement)
    return () => obs.disconnect()
  }, [])

  return <div ref={containerRef} style={{ width: '100%', height: '100%', background: '#f3f4f6', borderRadius: 8 }} />
}

// ─── India grid view ──────────────────────────────────────────
async function drawIndia(map, L, aiData, layers) {
  // Draw India boundary
  try {
    const geojson = await loadIndiaBoundary()
    layers.push(L.geoJSON(geojson, {
      style: { color: '#1e40af', weight: 2, fillColor: '#3b82f6', fillOpacity: 0.05 },
      interactive: false,
    }).addTo(map))
  } catch {}

  // Build cell data lookup
  const cellMap = {}
  ;(aiData?.cells || []).forEach(c => { cellMap[c.cell_index] = c })

  const markers = []

  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const idx = r * COLS + c
      const lat = LAT_MIN + r * CELL_LAT + CELL_LAT / 2
      const lon = LON_MIN + c * CELL_LON + CELL_LON / 2
      if (!isInsideIndia(lat, lon)) continue

      const cell = cellMap[idx] || {}
      const rain = cell.rainfall_1h_mm ?? 0
      if (rain < 1) continue

      const color = getRiskColor(rain * 1.2)
      const city = INDIA_CITIES[`${r},${c}`] || ''

      // Add colored circle marker
      markers.push(L.circleMarker([lat, lon], {
        radius: 8,
        fillColor: color,
        color: color,
        weight: 1,
        fillOpacity: 0.6,
      }).bindTooltip(`${city || `Cell ${idx}`}: ${fmt(rain, 0)}mm/hr`, { direction: 'top' }))

      // Add city label
      if (city && rain > 3) {
        markers.push(L.marker([lat, lon], {
          icon: L.divIcon({
            className: '',
            html: `<div style="font-family:Inter,sans-serif;text-align:center;pointer-events:none;transform:translate(-50%,-50%);text-shadow:0 1px 4px rgba(255,255,255,0.9);line-height:1.2">
              <div style="font-size:11px;font-weight:700;color:#1e293b;white-space:nowrap">${city}</div>
              <div style="font-size:10px;font-weight:800;color:${color};font-family:monospace;white-space:nowrap">${fmt(rain, 0)}mm</div>
            </div>`,
            iconSize: [0, 0],
            iconAnchor: [0, 0],
          }),
          interactive: false,
          zIndexOffset: 2000,
        }))
      }
    }
  }

  markers.forEach(m => { m.addTo(map); layers.push(m) })
}

// ─── Mumbai grid view with hazard-colored cells ───────────────
function drawMumbai(map, L, aiData, selectedCell, onSelectCell, layers) {
  // Gather data maps
  const riskMap = {}
  ;(aiData?.risk_heatmap?.heatmap || []).forEach(h => { riskMap[h.cell_index] = h })
  const floodMap = {}
  ;(aiData?.flood_depth?.node_estimates || []).forEach(n => { floodMap[n.cell_index] = n })
  const mhMap = {}
  ;(aiData?.multi_hazard?.cell_predictions || []).forEach(p => { mhMap[p.cell_index] = p })

  const markers = []

  for (let r = 0; r < M_ROWS; r++) {
    for (let c = 0; c < M_COLS; c++) {
      const idx = r * M_COLS + c
      const lat = M_LAT_MIN + r * M_CELL_LAT + M_CELL_LAT / 2
      const lon = M_LON_MIN + c * M_CELL_LON + M_CELL_LON / 2

      if (!isInsideMumbaiLand(lat, lon)) continue

      const mhCell = mhMap[idx]
      const dominant = mhCell?.dominant_hazard || 'SAFE'
      const riskRaw = riskMap[idx]?.pixel_risk_score ?? 0
      const mhSeverity = mhCell?.severity_score ?? 0
      const risk = riskRaw > 0 ? riskRaw : Math.min(mhSeverity, 70)
      const isSelected = selectedCell === `C${idx + 1}`

      const color = dominant !== 'SAFE' ? getHazardColor(dominant) : '#9ca3af'
      const icon = getHazardIcon(dominant)

      // Find locality name
      const locality = MUMBAI_LOCALITIES.find(
        loc => Math.abs(loc.lat - lat) < 0.02 && Math.abs(loc.lon - lon) < 0.02
      )
      const name = locality?.name || `Cell ${idx + 1}`

      // Create cell marker
      const marker = L.circleMarker([lat, lon], {
        radius: isSelected ? 14 : 11,
        fillColor: color,
        color: isSelected ? '#1e293b' : 'rgba(0,0,0,0.3)',
        weight: isSelected ? 3 : 1.5,
        fillOpacity: dominant !== 'SAFE' ? 0.8 : 0.3,
      })

      // Tooltip with hazard info
      marker.bindTooltip(
        `<div style="font-family:Inter,sans-serif;font-size:12px;line-height:1.4">
          <b>${name}</b><br/>
          ${icon} ${dominant}<br/>
          Risk: ${fmt(risk, 0)}%
          ${mhCell ? `<br/>TS: ${fmt(mhCell.thunderstorm_prob * 100, 0)}% | CB: ${fmt(mhCell.cloudburst_prob * 100, 0)}% | FF: ${fmt(mhCell.flash_flood_prob * 100, 0)}%` : ''}
        </div>`,
        { direction: 'top', className: 'hazard-tooltip' }
      )

      // Click handler
      marker.on('click', () => onSelectCell(isSelected ? null : `C${idx + 1}`))

      markers.push(marker)

      // Add overflow warning icon
      if (floodMap[idx]?.is_overflow_node || mhCell?.predicted_depth_cm > 30) {
        markers.push(L.marker([lat + 0.005, lon], {
          icon: L.divIcon({
            className: '',
            html: `<div style="font-size:14px;text-shadow:0 0 8px rgba(239,68,68,0.9);animation:pulse 1.5s infinite">⚠️</div>`,
            iconSize: [20, 20],
            iconAnchor: [10, 10],
          }),
          interactive: false,
          zIndexOffset: 3000,
        }))
      }
    }
  }

  // Add all markers to map
  markers.forEach(m => { m.addTo(map); layers.push(m) })

  // Add hazard legend
  const legend = L.control({ position: 'bottomright' })
  legend.onAdd = function () {
    const div = L.DomUtil.create('div', 'hazard-legend')
    div.innerHTML = `
      <div style="background:rgba(255,255,255,0.95);padding:10px 14px;border-radius:8px;font-size:11px;color:#1e293b;box-shadow:0 2px 8px rgba(0,0,0,0.15);border:1px solid #e5e7eb">
        <div style="font-weight:700;margin-bottom:6px;font-size:12px">Hazard Type</div>
        <div style="display:flex;align-items:center;gap:6px;margin:4px 0"><span style="width:12px;height:12px;border-radius:50%;background:#f59e0b;display:inline-block"></span> ⛈️ Thunderstorm</div>
        <div style="display:flex;align-items:center;gap:6px;margin:4px 0"><span style="width:12px;height:12px;border-radius:50%;background:#3b82f6;display:inline-block"></span> 🌧️ Cloudburst</div>
        <div style="display:flex;align-items:center;gap:6px;margin:4px 0"><span style="width:12px;height:12px;border-radius:50%;background:#ef4444;display:inline-block"></span> 🌊 Flash Flood</div>
        <div style="display:flex;align-items:center;gap:6px;margin:4px 0"><span style="width:12px;height:12px;border-radius:50%;background:#9ca3af;display:inline-block"></span> ✅ Safe</div>
      </div>
    `
    return div
  }
  legend.addTo(map)
  layers.push(legend)
}
