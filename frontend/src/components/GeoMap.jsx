/**
 * GeoMap — Professional Heatmap Polygon Visualization
 * =====================================================
 * Uses OpenStreetMap tiles (free, no API key).
 * Colored rectangle cells on the map for risk visualization.
 * Proper locality labels aligned to real Mumbai coordinates.
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
// Use geographic center of Mumbai for proper alignment
// Mumbai covers roughly 18.88-19.26°N, 72.78-73.00°E
const M_ROWS = 10, M_COLS = 9
const M_LAT_MIN = 18.88, M_LAT_MAX = 19.26, M_LON_MIN = 72.78, M_LON_MAX = 73.00
const M_CELL_LAT = (M_LAT_MAX - M_LAT_MIN) / M_ROWS
const M_CELL_LON = (M_LON_MAX - M_LON_MIN) / M_COLS

// ─── Real Mumbai locality coordinates ────────────────────────
// These are the REAL lat/lon centers for each locality
const MUMBAI_LOCALITIES = [
  // Row 0 (South Mumbai: 18.88-18.92)
  { name: 'Colaba', lat: 18.9210, lon: 72.8147 },
  { name: 'Fort', lat: 18.9380, lon: 72.8355 },
  { name: 'Churchgate', lat: 18.9320, lon: 72.8267 },
  { name: 'Marine Drive', lat: 18.9430, lon: 72.8230 },
  { name: 'Nariman Pt', lat: 18.9310, lon: 72.8180 },
  { name: 'Malabar Hill', lat: 18.9540, lon: 72.7950 },
  { name: 'Walkeshwar', lat: 18.9500, lon: 72.7970 },
  { name: 'Haji Ali', lat: 18.9820, lon: 72.8100 },
  { name: 'Parel', lat: 18.9930, lon: 72.8410 },
  // Row 1 (18.92-18.96)
  { name: 'Grant Road', lat: 18.9620, lon: 72.8120 },
  { name: 'Mumbai Central', lat: 18.9693, lon: 72.8212 },
  { name: 'Worli', lat: 19.0010, lon: 72.8140 },
  { name: 'Prabhadevi', lat: 18.9920, lon: 72.8350 },
  { name: 'Matunga', lat: 19.0180, lon: 72.8520 },
  { name: 'Sion', lat: 19.0430, lon: 72.8620 },
  { name: 'Wadala', lat: 19.0150, lon: 72.8560 },
  { name: 'Sewri', lat: 19.0010, lon: 72.8560 },
  { name: 'Chinchpokli', lat: 18.9960, lon: 72.8610 },
  // Row 2 (18.96-19.00)
  { name: 'Mahim', lat: 19.0370, lon: 72.8400 },
  { name: 'Dadar', lat: 19.0178, lon: 72.8478 },
  { name: 'Lower Parel', lat: 19.0070, lon: 72.8340 },
  { name: 'Elphinstone', lat: 19.0010, lon: 72.8390 },
  { name: 'Kurla', lat: 19.0720, lon: 72.8790 },
  { name: 'Vidyavihar', lat: 19.0740, lon: 72.8910 },
  { name: 'Ghatkopar', lat: 19.0860, lon: 72.9080 },
  { name: 'BKC', lat: 19.0530, lon: 72.8620 },
  { name: 'Kalina', lat: 19.0670, lon: 72.8790 },
  // Row 3 (19.00-19.04)
  { name: 'Bandra', lat: 19.0596, lon: 72.8295 },
  { name: 'Bandra West', lat: 19.0540, lon: 72.8250 },
  { name: 'Khar', lat: 19.0720, lon: 72.8390 },
  { name: 'Santacruz', lat: 19.0830, lon: 72.8420 },
  { name: 'Vile Parle', lat: 19.0970, lon: 72.8470 },
  { name: 'Andheri East', lat: 19.1144, lon: 72.8679 },
  { name: 'Saki Naka', lat: 19.1090, lon: 72.8840 },
  { name: 'Chandivali', lat: 19.1110, lon: 72.8950 },
  { name: 'Powai', lat: 19.1240, lon: 72.9087 },
  // Row 4 (19.04-19.08)
  { name: 'Juhu Beach', lat: 19.0940, lon: 72.8160 },
  { name: 'Juhu', lat: 19.1075, lon: 72.8263 },
  { name: 'Versova', lat: 19.1230, lon: 72.8150 },
  { name: 'Lokhandwala', lat: 19.1190, lon: 72.8330 },
  { name: 'Marol', lat: 19.1200, lon: 72.8620 },
  { name: 'MIDC Andheri', lat: 19.1220, lon: 72.8760 },
  { name: 'Sahar', lat: 19.0890, lon: 72.8900 },
  { name: 'Vikhroli', lat: 19.1090, lon: 72.9270 },
  { name: 'Bhandup', lat: 19.1470, lon: 72.9370 },
  // Row 5 (19.08-19.12)
  { name: 'Amboli', lat: 19.1350, lon: 72.8350 },
  { name: 'Jogeshwari', lat: 19.1420, lon: 72.8480 },
  { name: 'Goregaon', lat: 19.1660, lon: 72.8520 },
  { name: 'D.N. Nagar', lat: 19.1310, lon: 72.8350 },
  { name: 'Malad', lat: 19.1870, lon: 72.8460 },
  { name: 'Kandivali', lat: 19.2040, lon: 72.8520 },
  { name: 'Borivali', lat: 19.2307, lon: 72.8567 },
  { name: 'Dahisar', lat: 19.2500, lon: 72.8600 },
  { name: 'Kanjurmarg', lat: 19.1170, lon: 72.9320 },
  // Row 6 (19.12-19.16)
  { name: 'Malvani', lat: 19.1890, lon: 72.8260 },
  { name: 'Malad West', lat: 19.1828, lon: 72.8402 },
  { name: 'Goregaon E', lat: 19.1650, lon: 72.8650 },
  { name: 'Aarey Colony', lat: 19.1400, lon: 72.8740 },
  { name: 'Mindspace', lat: 19.1100, lon: 72.8870 },
  { name: 'Chembur E', lat: 19.0610, lon: 72.8980 },
  { name: 'Vikhroli E', lat: 19.1100, lon: 72.9370 },
  { name: 'Nahur', lat: 19.1230, lon: 72.9540 },
  { name: 'Thane', lat: 19.1978, lon: 72.9678 },
  // Row 7 (19.16-19.20)
  { name: 'Erangal', lat: 19.2060, lon: 72.8130 },
  { name: 'Kandivali W', lat: 19.2100, lon: 72.8380 },
  { name: 'Borivali W', lat: 19.2330, lon: 72.8370 },
  { name: 'Magathane', lat: 19.2150, lon: 72.8730 },
  { name: 'Mulund W', lat: 19.1690, lon: 72.9560 },
  { name: 'Mulund E', lat: 19.1690, lon: 72.9710 },
  { name: 'Wagle Estate', lat: 19.1930, lon: 72.9610 },
  { name: 'Thane City', lat: 19.2190, lon: 72.9790 },
  { name: 'Thane Creek', lat: 19.0200, lon: 72.9600 },
  // Row 8 (19.20-19.24)
  { name: 'Madh Island', lat: 19.2270, lon: 72.7920 },
  { name: 'Marve', lat: 19.2340, lon: 72.8100 },
  { name: 'Manori', lat: 19.2370, lon: 72.8080 },
  { name: 'Gorai', lat: 19.2450, lon: 72.8290 },
  { name: 'Uttan', lat: 19.2600, lon: 72.8340 },
  { name: 'Mira Road', lat: 19.2830, lon: 72.8580 },
  { name: 'Thane W', lat: 19.2000, lon: 72.9550 },
  { name: 'Kopar Khairane', lat: 19.0430, lon: 73.0020 },
  { name: 'Ghansoli', lat: 19.0630, lon: 73.0090 },
  // Row 9 (19.24-19.28)
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

// ─── Stadia Maps Dark Theme Tiles (Free for dev/hackathon) ───
// No API key needed for development. Falls back to CartoDB dark if unavailable.
const TILE_URLS = [
  'https://tiles.stadiamaps.com/tiles/alidade_smooth_dark/{z}/{x}/{y}{r}.png',
  'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
  'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
]
const TILE_ATTR = '&copy; <a href="https://carto.com/attributions">CARTO</a> | &copy; OpenStreetMap contributors'

// ─── City labels for India mode ───────────────────────────────
const INDIA_CITIES = {
  '6,10': 'Kanyakumari', '8,10': 'Kochi', '9,12': 'Bengaluru',
  '10,12': 'Hyderabad', '10,13': 'Chennai', '11,8': 'Goa',
  '12,7': 'Pune', '12,8': 'Mumbai', '13,6': 'Ahmedabad',
  '13,9': 'Bhopal', '14,16': 'Kolkata', '15,6': 'Jodhpur',
  '16,7': 'Delhi', '16,8': 'Agra', '17,7': 'Dehradun',
  '18,6': 'Amritsar', '19,19': 'Guwahati',
}

// ─── Color palettes ────────────────────────────────────────────
function getRiskColor(risk) {
  if (risk >= 80) return { bg: '#dc2626', border: '#b91c1c', text: '#ffffff', label: 'CRITICAL' }
  if (risk >= 60) return { bg: '#ea580c', border: '#c2410c', text: '#ffffff', label: 'HIGH' }
  if (risk >= 35) return { bg: '#d97706', border: '#a16207', text: '#ffffff', label: 'MEDIUM' }
  if (risk >= 15) return { bg: '#16a34a', border: '#15803d', text: '#ffffff', label: 'LOW' }
  return { bg: '#166534', border: '#14532d', text: '#ffffff', label: 'MINIMAL' }
}

function getRainColor(rain) {
  if (rain > 80)  return { bg: '#dc2626', border: '#b91c1c', text: '#ffffff', label: 'Extreme' }
  if (rain > 50)  return { bg: '#ea580c', border: '#c2410c', text: '#ffffff', label: 'Very Heavy' }
  if (rain > 25)  return { bg: '#d97706', border: '#a16207', text: '#ffffff', label: 'Heavy' }
  if (rain > 10)  return { bg: '#ca8a04', border: '#a16207', text: '#ffffff', label: 'Moderate' }
  if (rain > 5)   return { bg: '#0891b2', border: '#0e7490', text: '#ffffff', label: 'Light' }
  if (rain > 1)   return { bg: '#2563eb', border: '#1d4ed8', text: '#ffffff', label: 'Drizzle' }
  return { bg: '#1e3a5f', border: '#1e40af', text: '#94a3b8', label: 'Dry' }
}

// ─── SVG icons ─────────────────────────────────────────────────
function labelIcon(text, value, color, fontSize = 10) {
  return L.divIcon({
    className: '',
    html: `<div style="font-family:Inter,system-ui,sans-serif;text-align:center;pointer-events:none;transform:translate(-50%,-50%);text-shadow:0 1px 6px rgba(0,0,0,0.95),0 0 14px rgba(0,0,0,0.8);line-height:1.3">
      <div style="font-size:${fontSize}px;font-weight:700;color:rgba(255,255,255,0.95);white-space:nowrap">${text}</div>
      ${value ? `<div style="font-size:${fontSize-1}px;font-weight:800;color:${color};font-family:JetBrains Mono,monospace;white-space:nowrap">${value}</div>` : ''}
    </div>`,
    iconSize: [0, 0],
    iconAnchor: [0, 0],
  })
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
      attributionControl: false,
      minZoom: 4,
      maxZoom: 14,
    })

    // Try dark tiles first, then fallback
    for (const url of TILE_URLS) {
      try {
        const layer = L.tileLayer(url, {
          subdomains: 'abcd',
          maxZoom: 19,
          attribution: TILE_ATTR,
        })
        layer.addTo(map)
        break
      } catch {}
    }

    mapRef.current = map
    setTimeout(() => map.invalidateSize(), 300)
  }, [])

  // Redraw on data/mode change
  useEffect(() => {
    const map = mapRef.current
    const L = window.L
    if (!map || !L) return

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

  return <div ref={containerRef} style={{ width: '100%', height: '100%', background: '#0a0e17', borderRadius: 8 }} />
}

// ─── India heatmap ─────────────────────────────────────────────
async function drawIndia(map, L, aiData, layers) {
  try {
    const geojson = await loadIndiaBoundary()
    layers.push(L.geoJSON(geojson, {
      style: { color: '#38bdf8', weight: 1.8, fillColor: '#0284c7', fillOpacity: 0.04 },
      interactive: false,
    }).addTo(map))
  } catch {}

  const cellMap = {}
  ;(aiData?.cells || []).forEach(c => { cellMap[c.cell_index] = c })

  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const idx = r * COLS + c
      const lat = LAT_MIN + r * CELL_LAT + CELL_LAT / 2
      const lon = LON_MIN + c * CELL_LON + CELL_LON / 2
      if (!isInsideIndia(lat, lon)) continue

      const cell = cellMap[idx] || {}
      const rain = cell.rainfall_1h_mm ?? 0
      if (rain < 0.5) continue

      const { bg, border, label } = getRainColor(rain)
      const city = INDIA_CITIES[`${r},${c}`] || ''
      const opacity = Math.min(0.85, 0.15 + rain * 0.01)

      // Draw polygon cell
      const bounds = [
        [LAT_MIN + r * CELL_LAT, LON_MIN + c * CELL_LON],
        [LAT_MIN + (r + 1) * CELL_LAT, LON_MIN + (c + 1) * CELL_LON],
      ]
      const rect = L.rectangle(bounds, {
        color: border,
        weight: 1,
        fillColor: bg,
        fillOpacity: opacity,
        className: 'varuna-cell',
      })

      rect.bindPopup(`
        <div style="font-family:Inter,sans-serif;min-width:140px">
          <div style="font-size:13px;font-weight:700;margin-bottom:4px">📍 ${city || `Cell ${idx}`}</div>
          <div>🌧 <b>${fmt(rain,1)} mm/hr</b> — <span style="color:${bg}">${label}</span></div>
          <div>🌡 ${fmt(cell.temperature_celsius ?? 25,0)}°C</div>
        </div>
      `, { maxWidth: 200, className: 'varuna-popup' })

      rect.addTo(map)
      layers.push(rect)

      if (city && rain > 5) {
        layers.push(L.marker([lat, lon], {
          icon: labelIcon(city, `${fmt(rain,0)}mm`, bg, 10),
          interactive: false, zIndexOffset: 2000,
        }).addTo(map))
      }
    }
  }
}

// ─── Mumbai heatmap — polygon cells with locality labels ───────
function drawMumbai(map, L, aiData, selectedCell, onSelectCell, layers) {
  // Draw Mumbai boundary
  layers.push(L.geoJSON(MUMBAI_GEOJSON, {
    style: { color: '#06b6d4', weight: 2.2, fillColor: '#0891b2', fillOpacity: 0.04, dashArray: '5,5' },
    interactive: false,
  }).addTo(map))

  const riskMap = {}
  ;(aiData?.risk_heatmap?.heatmap || []).forEach(h => { riskMap[h.cell_index] = h })

  const floodMap = {}
  ;(aiData?.flood_depth?.node_estimates || []).forEach(n => { floodMap[n.cell_index] = n })

  const mhMap = {}
  ;(aiData?.multi_hazard?.cell_predictions || []).forEach(p => { mhMap[p.cell_index] = p })

  for (let r = 0; r < M_ROWS; r++) {
    for (let c = 0; c < M_COLS; c++) {
      const idx = r * M_COLS + c

      // Grid cell center
      const lat = M_LAT_MIN + r * M_CELL_LAT + M_CELL_LAT / 2
      const lon = M_LON_MIN + c * M_CELL_LON + M_CELL_LON / 2

      if (!isInsideMumbaiLand(lat, lon)) continue

      const risk = riskMap[idx]?.pixel_risk_score ?? mhMap[idx]?.severity_score ?? 0
      const floodCm = floodMap[idx]?.water_depth_cm ?? mhMap[idx]?.predicted_depth_cm ?? 0
      const overflow = floodMap[idx]?.is_overflow_node ?? false
      const isSelected = selectedCell === `C${idx + 1}`

      const { bg, border, label } = getRiskColor(risk)
      const locality = MUMBAI_LOCALITIES[idx]?.name || `Cell ${idx}`
      const opacity = risk > 0 ? Math.min(0.85, 0.2 + (risk / 100) * 0.65) : 0.05

      // Draw polygon cell
      const bounds = [
        [M_LAT_MIN + r * M_CELL_LAT, M_LON_MIN + c * M_CELL_LON],
        [M_LAT_MIN + (r + 1) * M_CELL_LAT, M_LON_MIN + (c + 1) * M_CELL_LON],
      ]

      const rect = L.rectangle(bounds, {
        color: isSelected ? '#3b82f6' : border,
        weight: isSelected ? 3 : 1.5,
        fillColor: risk > 0 ? bg : 'rgba(255,255,255,0.04)',
        fillOpacity: isSelected ? Math.min(opacity + 0.15, 1) : opacity,
        className: `varuna-cell ${isSelected ? 'selected' : ''}`,
      })

      const inferMode = mhMap[idx]?.model || 'unknown'
      rect.bindPopup(`
        <div style="font-family:Inter,sans-serif;min-width:180px">
          <div style="font-size:14px;font-weight:800;margin-bottom:6px;color:${bg}">📍 ${locality}</div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:4px">
            <div>Risk: <b style="color:${bg}">${fmt(risk,0)}/100</b></div>
            <div>🌊 Flood: <b>${fmt(floodCm,0)} cm</b></div>
          </div>
          ${overflow ? '<div style="color:#ef4444;font-weight:700;margin-top:4px">⚠️ OVERFLOW NODE</div>' : ''}
          <div style="font-size:10px;color:#94a3b8;margin-top:6px;border-top:1px solid #334155;padding-top:4px">
            Model: ${inferMode} | Cell: C${idx + 1}
          </div>
        </div>
      `, { maxWidth: 240, className: 'varuna-popup' })

      rect.on('click', () => onSelectCell(isSelected ? null : `C${idx + 1}`))
      rect.addTo(map)
      layers.push(rect)

      // Locality label (always visible, with background)
      const labelBg = risk > 50 ? 'rgba(0,0,0,0.7)' : 'rgba(0,0,0,0.5)'
      layers.push(L.marker([lat, lon], {
        icon: L.divIcon({
          className: '',
          html: `<div style="font-family:Inter,system-ui,sans-serif;text-align:center;pointer-events:none;transform:translate(-50%,-50%);padding:1px 4px;border-radius:3px;background:${labelBg};backdrop-filter:blur(2px)">
            <div style="font-size:9px;font-weight:700;color:rgba(255,255,255,0.9);white-space:nowrap;line-height:1.2;text-shadow:0 1px 3px rgba(0,0,0,0.8)">${locality}</div>
            ${risk > 0 ? `<div style="font-size:8px;font-weight:800;color:${bg};font-family:JetBrains Mono,monospace;white-space:nowrap">${fmt(risk,0)}</div>` : ''}
          </div>`,
          iconSize: [0, 0],
          iconAnchor: [0, 0],
        }),
        interactive: false, zIndexOffset: 2000,
      }).addTo(map))

      // Overflow indicator
      if (overflow) {
        layers.push(L.marker([lat + 0.003, lon], {
          icon: L.divIcon({
            className: '',
            html: `<div style="font-size:14px;text-shadow:0 0 8px rgba(239,68,68,0.8)">⚠️</div>`,
            iconSize: [16, 16],
            iconAnchor: [8, 8],
          }),
          interactive: false, zIndexOffset: 3000,
        }).addTo(map))
      }
    }
  }
}
