/**
 * LeftPanel — Situation Overview + Active Alerts + Timeline KPIs
 * ===============================================================
 * Displays on the left side of the map.
 * Shows: situation overview, active alerts, storm info, key metrics.
 * All KPIs sync with the current timestep.
 */
import { useState, useEffect } from 'react'
import { riskColor, fmt, fmtPct } from '../utils/helpers'
import { api } from '../utils/api'

const LOCALITY = [
  ['Colaba','Fort','Churchgate','Marine Drive','Nariman Point','Malabar Hill','Walkeshwar','Haji Ali','Parel'],
  ['Grant Road','Mumbai Central','Worli','Prabhadevi','Matunga','Sion','Wadala','Sewri','Chinchpokli'],
  ['Mahim','Dadar','Lower Parel','Elphinstone','Kurla','Vidyavihar','Ghatkopar','BKC','Kalina'],
  ['Bandra','Bandra West','Khar','Santacruz','Vile Parle','Andheri E','Saki Naka','Chandivali','Powai'],
  ['Juhu Beach','Juhu','Versova','Lokhandwala','Marol','MIDC Andheri','Sahar','Vikhroli','Bhandup'],
  ['Amboli','Jogeshwari','Goregaon','D.N. Nagar','Malad','Kandivali','Borivali','Dahisar','Kanjurmarg'],
  ['Malvani','Malad West','Goregaon E','Aarey Colony','Mindspace','Chinchpokli E','Vikhroli E','Nahur','Thane'],
  ['Erangal','Kandivali W','Borivali W','Magathane','Mulund W','Mulund E','Wagle Estate','Thane City','Thane Creek'],
  ['Madh Island','Marve','Manori','Gorai','Uttan','Mira Road','Thane W','Kopar Khairane','Ghansoli'],
  ['Vasai','Nallasopara','Virar','Dahisar N','Mira Bhayander','Vashi','Sanpada','Nerul','Belapur'],
].flat()

export default function LeftPanel({ timestep, aiData, replayStepData, selectedCell, mode }) {
  return (
    <>
      {/* 1. Situation Overview — key metrics */}
      <SituationOverview aiData={aiData} replayStepData={replayStepData} mode={mode} />

      {/* 2. Data Sources & Model Status */}
      <DataSourceSection aiData={aiData} />

      {/* 3. Active Alerts */}
      <AlertsSection timestep={timestep} />

      {/* 4. Stay Alert — timeline synced */}
      <StayAlertSection timestep={timestep} aiData={aiData} mode={mode} />

      {/* 5. Storm Cell Info */}
      <StormInfoSection aiData={aiData} />
    </>
  )
}


/* ═══ SITUATION OVERVIEW ═══ */
function SituationOverview({ aiData, replayStepData, mode }) {
  const risk = aiData?.risk_heatmap || {}
  const mh = aiData?.multi_hazard || {}
  const flood = aiData?.flood_depth || {}
  const storm = aiData?.storm_cells || {}
  const physics = aiData?.physics_risk || {}

  const avgRisk = risk.mean_pixel_risk ?? physics.avg_physics_risk ?? 0
  const maxRisk = risk.max_pixel_risk ?? 0
  const maxDepth = flood.max_water_depth_cm ?? 0
  const stormCount = storm.total_cells_detected ?? 0

  return (
    <div className="section-card compact">
      <h4>📊 Situation Overview</h4>
      <div className="kpi-grid-2">
        <KPICard label="Avg Risk" value={fmt(avgRisk, 0)} color={riskColor(avgRisk)} />
        <KPICard label="Max Risk" value={fmt(maxRisk, 0)} color={riskColor(maxRisk)} />
        <KPICard label="Max Depth" value={`${fmt(maxDepth, 0)}cm`} color={maxDepth > 30 ? '#ef4444' : '#eab308'} />
        <KPICard label="Storms" value={fmt(stormCount, 0)} color={stormCount > 5 ? '#f97316' : '#06b6d4'} />
      </div>
    </div>
  )
}


/* ═══ ACTIVE ALERTS ═══ */
function AlertsSection({ timestep }) {
  const [allAlerts, setAlerts] = useState([])
  const [expanded, setExpanded] = useState(false)

  useEffect(() => {
    api.alerts().then(d => {
      const list = Array.isArray(d) ? d : (d?.alerts || [])
      setAlerts(list)
    }).catch(() => {})
  }, [])  // Fetch once, filter locally

  // Only show alerts for the CURRENT timestep
  const alerts = allAlerts.filter(a => a.timestep === timestep)

  const sorted = [...alerts]
    .sort((a, b) => (b.risk_score_total || 0) - (a.risk_score_total || 0))

  const critical = sorted.filter(a => a.severity === 'CRITICAL')
  const high = sorted.filter(a => a.severity === 'HIGH')
  const shown = expanded ? sorted.slice(0, 15) : sorted.slice(0, 5)

  return (
    <div className="section-card compact">
      <h4>
        🚨 Active Alerts
        {critical.length > 0 && <span className="badge badge-red">{critical.length} Critical</span>}
        {high.length > 0 && <span className="badge badge-orange">{high.length} High</span>}
      </h4>

      {shown.length === 0 && <p className="text-dim">No active alerts at this timestep</p>}

      {shown.map((a, i) => {
        const hazardIcon = a.alert_type === 'CLOUDBURST' ? '🌧️'
          : a.alert_type === 'SEVERE_THUNDERSTORM' ? '⛈️'
          : a.alert_type === 'FLASH_FLOOD' ? '🌊'
          : '⚠️'
        const hazardColor = a.alert_type === 'CLOUDBURST' ? '#3b82f6'
          : a.alert_type === 'SEVERE_THUNDERSTORM' ? '#f59e0b'
          : a.alert_type === 'FLASH_FLOOD' ? '#ef4444'
          : '#eab308'
        return (
          <div key={a.id || i} className="alert-row">
            <span className={`alert-dot ${a.severity === 'CRITICAL' ? 'dot-red' : a.severity === 'HIGH' ? 'dot-orange' : 'dot-yellow'}`} />
            <span className="alert-locality">{a.locality_name || `Cell ${a.cell_id}`}</span>
            <span className="alert-type" style={{color: hazardColor}}>{hazardIcon} {a.alert_type}</span>
            <span className="alert-score">{fmt(a.risk_score_total, 0)}</span>
          </div>
        )
      })}

      {sorted.length > 5 && (
        <button className="btn-expand" onClick={() => setExpanded(!expanded)}>
          {expanded ? '▲ Show less' : `▼ Show all ${sorted.length}`}
        </button>
      )}
    </div>
  )
}


/* ═══ STAY ALERT — TIMELINE SYNCED ═══ */
function StayAlertSection({ timestep, aiData, mode }) {
  if (mode !== 'mumbai') return null

  // Compute alert level based on current timestep
  const risk = aiData?.risk_heatmap || {}
  const avgRisk = risk.mean_pixel_risk ?? 0
  const maxRisk = risk.max_pixel_risk ?? 0
  const storm = aiData?.storm_cells || {}
  const stormCount = storm.total_cells_detected ?? 0
  const flood = aiData?.flood_depth || {}
  const overflowNodes = flood.overflow_nodes_count ?? 0

  // Phase-based alert level
  let alertLevel, alertColor, alertIcon, alertText
  if (timestep <= 12) {
    alertLevel = 'MONITORING'; alertColor = '#22c55e'; alertIcon = '🟢'
    alertText = 'Conditions normal. Monitoring weather systems.'
  } else if (timestep <= 24) {
    alertLevel = 'WATCH'; alertColor = '#eab308'; alertIcon = '🟡'
    alertText = 'Storm building. Stay informed and prepare.'
  } else if (timestep <= 40) {
    alertLevel = 'WARNING'; alertColor = '#f97316'; alertIcon = '🟠'
    alertText = 'Active flooding risk. Avoid low-lying areas.'
  } else if (timestep <= 56) {
    alertLevel = 'ALERT'; alertColor = '#ef4444'; alertIcon = '🔴'
    alertText = 'Severe conditions. Evacuate if instructed.'
  } else {
    alertLevel = 'RECESSION'; alertColor = '#06b6d4'; alertIcon = '🔵'
    alertText = 'Conditions improving. Watch for residual flooding.'
  }

  // Override with real data if available
  if (maxRisk >= 80) { alertLevel = 'CRITICAL'; alertColor = '#ef4444'; alertIcon = '🔴' }

  return (
    <div className="section-card compact">
      <h4>{alertIcon} Stay Alert</h4>
      <div className="alert-status-bar" style={{ borderLeftColor: alertColor }}>
        <div className="alert-level" style={{ color: alertColor }}>{alertLevel}</div>
        <div className="alert-text">{alertText}</div>
      </div>
      <div className="kpi-grid-3 mini">
        <MiniKPI label="Risk" value={fmt(avgRisk, 0)} color={riskColor(avgRisk)} />
        <MiniKPI label="Storms" value={fmt(stormCount, 0)} color={stormCount > 5 ? '#f97316' : '#06b6d4'} />
        <MiniKPI label="Overflow" value={fmt(overflowNodes, 0)} color={overflowNodes > 0 ? '#ef4444' : '#22c55e'} />
      </div>
    </div>
  )
}


/* ═══ STORM INFO ═══ */
function StormInfoSection({ aiData }) {
  const storm = aiData?.storm_cells || {}
  const cells = storm.cells || []
  const peak = storm.peak_storm_intensity ?? 0

  if (cells.length === 0) return null

  return (
    <div className="section-card compact">
      <h4>⛈️ Storm Cells Detected</h4>
      <div className="kpi-grid-2">
        <KPICard label="Active Cells" value={fmt(cells.length, 0)} color="#f97316" />
        <KPICard label="Peak Intensity" value={`${fmt(peak, 0)}%`} color={peak > 70 ? '#ef4444' : '#eab308'} />
      </div>
      {cells.slice(0, 3).map((c, i) => (
        <div key={i} className="storm-cell-row">
          <span className="storm-id">{c.tracking_id || `SC-${i+1}`}</span>
          <span className="storm-intensity" style={{ color: riskColor(c.intensity || 0) }}>
            {fmt(c.intensity || 0, 0)}%
          </span>
          <span className="storm-dir">{c.movement_direction || '—'}</span>
        </div>
      ))}
    </div>
  )
}


/* ═══ DATA SOURCES & MODEL STATUS ═══ */
function DataSourceSection({ aiData }) {
  const mhMode = aiData?.multi_hazard?.inference_mode || 'heuristic'
  const fdMode = aiData?.flood_depth?.inference_mode || 'heuristic'
  const sat = aiData?.satellite_data || {}
  const physics = aiData?.physics_risk || {}
  const trust = aiData?.trust_score || {}

  const isNeuralNet = mhMode === 'trained_neural_network'
  const satSource = sat.source || 'N/A'
  const satReal = sat.is_real_data
  const physicsValid = physics.conservation_valid ?? physics.summary?.physics_valid
  const conformalGuarantee = trust.confidence_guarantee || 'N/A'

  return (
    <div className="section-card compact">
      <h4>🤖 AI Engine Status</h4>
      <div className="kpi-grid-2">
        <KPICard label="Neural Net" value={isNeuralNet ? 'ON' : 'OFF'} color={isNeuralNet ? '#22c55e' : '#eab308'} />
        <KPICard label="Trust" value={fmt(trust.trust_score || 0, 0) + '%'} color={trust.trust_score > 70 ? '#22c55e' : trust.trust_score > 40 ? '#eab308' : '#ef4444'} />
      </div>
      <div className="source-grid" style={{ marginTop: 8, gap: 6 }}>
        <div className="source-item">
          <span className="source-icon" style={{ fontSize: '14px' }}>🛰️</span>
          <div className="source-info">
            <span className="source-name" style={{ fontSize: '10px' }}>INSAT-3D</span>
            <span className="source-status" style={{ color: satReal ? '#22c55e' : '#eab308', fontSize: '10px', fontWeight: 600 }}>{satReal ? 'LIVE' : satSource === 'SYNTHETIC_INSAT3D_CALIBRATED' ? 'Synthetic' : 'Calibrated'}</span>
          </div>
        </div>
        <div className="source-item">
          <span className="source-icon" style={{ fontSize: '14px' }}>🧮</span>
          <div className="source-info">
            <span className="source-name" style={{ fontSize: '10px' }}>Physics Engine</span>
            <span className="source-status" style={{ color: physicsValid ? '#22c55e' : '#ef4444' }}>{physicsValid ? 'Valid' : 'Check'}</span>
          </div>
        </div>
        <div className="source-item">
          <span className="source-icon" style={{ fontSize: '14px' }}>📊</span>
          <div className="source-info">
            <span className="source-name" style={{ fontSize: '10px' }}>Multi-Hazard</span>
            <span className="source-status" style={{ color: isNeuralNet ? '#22c55e' : '#eab308' }}>{mhMode === 'trained_neural_network' ? 'PyTorch' : 'Heuristic'}</span>
          </div>
        </div>
        <div className="source-item">
          <span className="source-icon" style={{ fontSize: '14px' }}>🎯</span>
          <div className="source-info">
            <span className="source-name" style={{ fontSize: '10px' }}>Conformal</span>
            <span className="source-status" style={{ color: '#06b6d4' }}>{conformalGuarantee !== 'N/A' ? '90%' : 'Off'}</span>
          </div>
        </div>
      </div>
    </div>
  )
}


/* ═══ SHARED COMPONENTS ═══ */
function KPICard({ label, value, color }) {
  return (
    <div className="kpi-card">
      <div className="kpi-value" style={{ color }}>{value}</div>
      <div className="kpi-label">{label}</div>
    </div>
  )
}

function MiniKPI({ label, value, color }) {
  return (
    <div className="mini-kpi">
      <span className="mini-kpi-val" style={{ color }}>{value}</span>
      <span className="mini-kpi-label">{label}</span>
    </div>
  )
}
