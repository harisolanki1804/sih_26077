// Shared helpers for VARUNA frontend

export const RISK_COLORS = {
  LOW: '#22c55e',
  MODERATE: '#eab308',
  HIGH: '#f97316',
  SEVERE: '#ef4444',
}

export const SEVERITY_COLORS = {
  low: '#22c55e',
  moderate: '#eab308',
  high: '#f97316',
  critical: '#ef4444',
  LOW: '#22c55e',
  MODERATE: '#eab308',
  HIGH: '#f97316',
  SEVERE: '#ef4444',
  CRITICAL: '#ef4444',
}

export function riskColor(score) {
  if (score >= 75) return '#ef4444'
  if (score >= 50) return '#f97316'
  if (score >= 25) return '#eab308'
  return '#22c55e'
}

export function riskLabel(score) {
  if (score >= 75) return 'SEVERE'
  if (score >= 50) return 'HIGH'
  if (score >= 25) return 'MODERATE'
  return 'LOW'
}

export function fmt(v, d = 1) {
  if (v === null || v === undefined || isNaN(v)) return '--'
  return Number(v).toFixed(d)
}

export function fmtPct(v) {
  if (v === null || v === undefined || isNaN(v)) return '--%'
  return `${(Number(v) * 100).toFixed(0)}%`
}

export function timeSince(dateStr) {
  const diff = Date.now() - new Date(dateStr).getTime()
  const mins = Math.floor(diff / 60000)
  if (mins < 1) return 'just now'
  if (mins < 60) return `${mins}m ago`
  return `${Math.floor(mins / 60)}h ago`
}

export function getPhaseLabel(t) {
  if (t <= 12) return 'Pre-Event Baseline'
  if (t <= 24) return 'Convective Initiation'
  if (t <= 36) return 'Rapid Intensification'
  if (t <= 48) return 'Peak Storm Event'
  if (t <= 60) return 'Sustained Deluge'
  return 'Recession & Recovery'
}

export function getPhaseColor(t) {
  if (t <= 12) return '#22c55e'
  if (t <= 24) return '#eab308'
  if (t <= 36) return '#f97316'
  if (t <= 48) return '#ef4444'
  if (t <= 60) return '#f97316'
  return '#eab308'
}
