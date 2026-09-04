/**
 * IntelligencePanel — Right Side Panel (AI + Nowcast + What-If + XAI)
 * ===================================================================
 * Displays on the right side of the map.
 * Shows: What's Coming (nowcast), Confidence, What-If Chatbot, Why This Alert.
 */
import { useMemo, useState, useEffect, useCallback } from 'react'
import { riskColor, fmt, fmtPct } from '../utils/helpers'
import WhatIfChatbot from './WhatIfChatbot'

export default function IntelligencePanel({ timestep, aiData, replayStepData, selectedCell, mode }) {
  if (!aiData) return <div className="loading">Loading weather data...</div>

  return (
    <div className="intel-scroll">
      {/* 1. What's Coming — nowcast (columns layout) */}
      <ComingHoursSection aiData={aiData} timestep={timestep} />

      {/* 2. Safe Escape Routes */}
      <SafeRoutesSection aiData={aiData} selectedCell={selectedCell} />

      {/* 3. Confidence / Trust */}
      <ConfidenceSection aiData={aiData} />

      {/* 4. What-If Chatbot */}
      <WhatIfChatbot timestep={timestep} />

      {/* 5. Why This Alert — XAI */}
      <WhyAlertSection aiData={aiData} selectedCell={selectedCell} />
    </div>
  )
}


/* ═══ WHAT'S COMING — NEXT 6 HOURS ═══ */
function ComingHoursSection({ aiData, timestep }) {
  const nowcast = aiData?.nowcast || {}
  const forecasts = nowcast.forecasts || []
  const trend = nowcast.trend || 'stable'
  const trendRate = nowcast.trend_rate_mm_hr_per_step || 0

  const hours = useMemo(() => {
    if (forecasts.length > 0) {
      return forecasts.slice(0, 6).map((f, i) => ({
        hour_offset: i + 1,
        rain_mm_hr: f.predicted_avg_rainfall_mm_hr ?? 0,
        severity: f.predicted_risk_level || 'LOW',
        confidence: f.confidence || 0.7,
      }))
    }
    const baseRain = aiData?.risk_heatmap?.mean_pixel_risk ?? 20
    return Array.from({ length: 6 }, (_, i) => {
      const factor = trend === 'intensifying' ? (1 + i * 0.15) : trend === 'weakening' ? (1 - i * 0.1) : 1
      const rain = Math.max(0, baseRain * factor * 0.8)
      return {
        hour_offset: i + 1,
        rain_mm_hr: rain,
        severity: rain > 60 ? 'CRITICAL' : rain > 40 ? 'HIGH' : rain > 20 ? 'MEDIUM' : 'LOW',
        confidence: 0.85 - i * 0.05,
      }
    })
  }, [forecasts, aiData, trend])

  const maxRain = Math.max(10, ...hours.map(h => h.rain_mm_hr))
  const trendIcon = trend === 'intensifying' ? '📈' : trend === 'weakening' ? '📉' : '➡️'
  const trendLabel = trend === 'intensifying' ? 'Rising' : trend === 'weakening' ? 'Falling' : 'Stable'
  const trendColor = trend === 'intensifying' ? '#f97316' : trend === 'weakening' ? '#22c55e' : '#94a3b8'

  return (
    <div className="section-card forecast-card">
      {/* Header row with trend */}
      <div className="forecast-header">
        <span className="forecast-header-title">⏰ Next 6 Hours</span>
        <span className="forecast-trend-badge" style={{ color: trendColor, borderColor: trendColor + '40' }}>
          {trendIcon} {trendLabel}
        </span>
      </div>

      {/* Bar chart */}
      <div className="forecast-chart">
        {hours.map((h, i) => {
          const rain = h.rain_mm_hr
          const sev = h.severity || (rain > 60 ? 'CRITICAL' : rain > 40 ? 'HIGH' : rain > 20 ? 'MEDIUM' : 'LOW')
          const color = sev === 'CRITICAL' ? '#ef4444' : sev === 'HIGH' ? '#f97316' : sev === 'MEDIUM' ? '#eab308' : '#22c55e'
          const barHeight = Math.max(4, (rain / maxRain) * 100)
          const conf = Math.round((h.confidence || 0.7) * 100)
          return (
            <div key={i} className="forecast-col">
              <div className="forecast-value" style={{ color }}>{fmt(rain, 0)}</div>
              <div className="forecast-bar-track">
                <div className="forecast-bar-fill" style={{
                  height: `${barHeight}%`,
                  background: `linear-gradient(to top, ${color}cc, ${color})`,
                  boxShadow: rain > 40 ? `0 0 8px ${color}44` : 'none',
                }} />
              </div>
              <div className="forecast-label">+{h.hour_offset}h</div>
            </div>
          )
        })}
      </div>

      {/* Severity legend dots */}
      <div className="forecast-legend">
        <span className="legend-dot" style={{ background: '#ef4444' }} /><span>80+</span>
        <span className="legend-dot" style={{ background: '#f97316' }} /><span>50-80</span>
        <span className="legend-dot" style={{ background: '#eab308' }} /><span>20-50</span>
        <span className="legend-dot" style={{ background: '#22c55e' }} /><span>&lt;20</span>
      </div>

      {/* Trend description */}
      <div className="trend-text" style={{ color: trendColor }}>
        {trend === 'intensifying' && `⚡ Rainfall intensifying at +${fmt(trendRate,1)} mm/hr per step`}
        {trend === 'weakening' && '✅ Conditions improving — storm weakening'}
        {trend === 'steady' && '📊 Conditions stable — monitor for changes'}
        {trend === 'insufficient_data' && '⏳ Collecting data — forecasting unavailable at this timestep'}
      </div>
    </div>
  )
}


/* ═══ SAFE ESCAPE ROUTES ═══ */
function SafeRoutesSection({ aiData, selectedCell }) {
  const [routes, setRoutes] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [cellIdx, setCellIdx] = useState(40) // Default to Juhu area

  const fetchRoutes = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await api.innovations.evacuationRoutes(cellIdx, 'vehicle')
      setRoutes(data)
    } catch (e) {
      console.error('Evacuation route error:', e)
      setError('Could not compute routes')
    }
    setLoading(false)
  }, [cellIdx])

  // Auto-fetch on first render
  useEffect(() => { fetchRoutes() }, [])

  return (
    <div className="section-card compact">
      <h4>🗺️ Safe Escape Routes</h4>
      <div className="evac-controls">
        <select className="evac-select" value={cellIdx} onChange={e => setCellIdx(parseInt(e.target.value))}>
          <option value={36}>Juhu (Cell 37)</option>
          <option value={27}>Bandra (Cell 28)</option>
          <option value={19}>Dadar (Cell 20)</option>
          <option value={38}>Andheri E (Cell 39)</option>
          <option value={46}>Goregaon (Cell 47)</option>
          <option value={30}>Kurla (Cell 31)</option>
          <option value={21}>Sion (Cell 22)</option>
          <option value={40}>Powai (Cell 41)</option>
          <option value={9}>Parel (Cell 10)</option>
          <option value={0}>Colaba (Cell 1)</option>
        </select>
        <button className="btn-find-route" onClick={fetchRoutes} disabled={loading}>
          {loading ? '⏳' : '🔍'} Find
        </button>
      </div>

      {error && <div className="text-dim" style={{ color: '#f97316' }}>⚠️ {error}</div>}

      {routes && routes.primary_route && (
        <div className="route-card" style={{ borderColor: routes.primary_route.safety_score > 80 ? '#22c55e' : '#eab308' }}>
          <div className="route-header">
            <span className="route-rank">✅ Best Route</span>
            <span className="route-time">{routes.primary_route.estimated_time_min || '?'} min</span>
          </div>
          <div className="route-dest" style={{ fontWeight: 700 }}>{routes.primary_route.destination || 'Safe Zone'}</div>
          <div className="route-details">
            <span>{routes.primary_route.distance_km || '?'} km</span>
            <span className="route-safety" style={{ color: routes.primary_route.safety_score > 80 ? '#22c55e' : '#eab308' }}>
              Safety: {routes.primary_route.safety_score || '?'}%
            </span>
          </div>
          {routes.primary_route.max_water_depth_cm > 0 && (
            <div style={{ fontSize: 9, color: '#94a3b8', marginTop: 2 }}>
              🌊 Max depth on route: {routes.primary_route.max_water_depth_cm}cm
            </div>
          )}
        </div>
      )}

      {routes && routes.alternative_routes && routes.alternative_routes.slice(0, 2).map((r, i) => (
        <div key={i} className="route-card">
          <div className="route-header">
            <span className="route-rank" style={{ color: '#94a3b8' }}>Route {i + 2}</span>
            <span className="route-time">{r.estimated_time_min || '?'} min</span>
          </div>
          <div className="route-dest">{r.destination || `Safe Zone ${i+2}`}</div>
          <div className="route-details">
            <span>{r.distance_km || '?'} km</span>
            <span className="route-safety" style={{ color: r.safety_score > 80 ? '#22c55e' : '#eab308' }}>
              Safety: {r.safety_score || '?'}%
            </span>
          </div>
        </div>
      ))}

      {routes?.recommendation && (
        <div style={{ fontSize: 9, color: '#06b6d4', padding: '4px 0', lineHeight: 1.4 }}>
          💡 {routes.recommendation}
        </div>
      )}
    </div>
  )
}

// Need to import api for SafeRoutes
import { api } from '../utils/api'


/* ═══ CONFIDENCE / TRUST ═══ */
function ConfidenceSection({ aiData }) {
  const trust = aiData?.trust_score || {}
  const score = trust.trust_score ?? 50
  const isAnomaly = trust.is_anomalous_pattern
  const conformal = trust.conformal_prediction || {}
  const guarantee = trust.confidence_guarantee || ''
  const uncertainty = trust.uncertainty_margin_pct ?? 0

  let confLabel, confColor
  if (score >= 80) { confLabel = 'Very Confident'; confColor = '#22c55e' }
  else if (score >= 60) { confLabel = 'Confident'; confColor = '#eab308' }
  else if (score >= 40) { confLabel = 'Uncertain'; confColor = '#f97316' }
  else { confLabel = 'Low Confidence'; confColor = '#ef4444' }

  return (
    <div className="section-card compact">
      <h4>🎯 How Sure Are We?</h4>
      <div className="trust-display">
        <div className="trust-arc">
          <svg viewBox="0 0 120 70" className="trust-svg-md">
            <path d="M10 60 A50 50 0 0 1 110 60" fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="8" />
            <path
              d="M10 60 A50 50 0 0 1 110 60"
              fill="none" stroke={confColor} strokeWidth="8"
              strokeDasharray={`${score * 1.57} 157`}
              strokeLinecap="round"
              style={{ transition: 'stroke-dasharray 0.5s' }}
            />
            <text x="60" y="52" textAnchor="middle" fill={confColor} fontSize="22" fontWeight="800" fontFamily="JetBrains Mono">{fmt(score, 0)}%</text>
            <text x="60" y="64" textAnchor="middle" fill="#94a3b8" fontSize="9">{confLabel}</text>
          </svg>
        </div>
      </div>
      {guarantee && (
        <div style={{ fontSize: 10, color: '#06b6d4', textAlign: 'center', marginTop: 4 }}>
          📐 {guarantee}
        </div>
      )}
      {isAnomaly && (
        <div className="anomaly-warning">
          ⚠️ Unusual pattern — cross-check with IMD alerts
        </div>
      )}
      <div style={{ marginTop: 6 }}>
        <div style={{ fontSize: 10, color: '#64748b', marginBottom: 4 }}>Model Agreement</div>
        <AgreementBar label="Cross-Source" value={trust.components?.cross_source_agreement ?? 0} />
        <AgreementBar label="Nowcast" value={trust.components?.nowcast_confidence ?? 0} />
        <AgreementBar label="Storm Detection" value={trust.components?.storm_detection_consistency ?? 0} />
      </div>
    </div>
  )
}

function AgreementBar({ label, value }) {
  const pct = Math.round(value * 100)
  const color = pct > 80 ? '#22c55e' : pct > 50 ? '#eab308' : '#ef4444'
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
      <span style={{ fontSize: 10, color: '#94a3b8', width: 90, flexShrink: 0 }}>{label}</span>
      <div style={{ flex: 1, height: 5, background: 'rgba(255,255,255,0.05)', borderRadius: 3, overflow: 'hidden' }}>
        <div style={{ height: '100%', width: `${pct}%`, background: color, borderRadius: 3, transition: 'width 0.3s' }} />
      </div>
      <span style={{ fontSize: 10, fontFamily: 'JetBrains Mono', color, width: 32, textAlign: 'right' }}>{pct}%</span>
    </div>
  )
}


/* ═══ WHY THIS ALERT — XAI ═══ */
function WhyAlertSection({ aiData, selectedCell }) {
  const xai = aiData?.xai_explanation || {}
  const importanceDict = xai.feature_importance_global || {}
  const summary = xai.global_summary || ''

  const factorNames = {
    'rainfall_intensity': '🌧 Heavy rain',
    'cape_instability': '⚡ Storm energy',
    'cloud_top_temperature': '☁️ Cold clouds',
    'soil_saturation': '💧 Soaked ground',
    'elevation_depression': '🏔 Low areas',
    'tidal_lock': '🌊 Tidal lock',
    'wind_shear': '💨 Wind shear',
    'drainage_distance': '🚰 Drain distance',
    'integrated_water_vapor': '💦 Moisture column',
    'convective_inhibition': '⛔ Storm cap',
    'lifted_index': '🎈 Instability',
    'ctt_drop_rate': '🥶 Cloud cooling',
  }

  const sorted = Object.entries(importanceDict)
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .slice(0, 5)

  return (
    <div className="section-card compact">
      <h4>🔍 Why This Alert?</h4>

      <div className="driver-simple-list">
        {sorted.map(([key, val], i) => (
          <div key={i} className="driver-simple-row">
            <span className="driver-rank">#{i + 1}</span>
            <span className="driver-name">{factorNames[key] || key.replace(/_/g, ' ')}</span>
            <div className="driver-bar-track">
              <div className="driver-bar-fill" style={{
                width: `${Math.abs(val) * 100}%`,
                background: riskColor(Math.abs(val) * 100),
              }} />
            </div>
          </div>
        ))}
      </div>

      {summary && (
        <div className="summary-plain">
          <strong>📝 Summary:</strong> {summary}
        </div>
      )}
    </div>
  )
}
