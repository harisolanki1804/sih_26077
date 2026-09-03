import { useState, useEffect, useCallback, useRef } from 'react'
import { api } from './utils/api'
import { riskColor, fmt, getPhaseLabel, getPhaseColor } from './utils/helpers'
import MapView from './components/MapView'
import IntelligencePanel from './components/IntelligencePanel'
import LeftPanel from './components/LeftPanel'

export default function App() {
  const [health, setHealth] = useState(null)
  const [timestep, setTimestep] = useState(1)
  const [replayData, setReplayData] = useState(null)
  const [replayStepData, setReplayStepData] = useState(null)
  const [aiData, setAiData] = useState(null)
  const [playing, setPlaying] = useState(false)
  const [selectedCell, setSelectedCell] = useState(null)
  const [loading, setLoading] = useState(false)
  const [mapMode, setMapMode] = useState('india')
  const [liveData, setLiveData] = useState(null)
  const [liveSummary, setLiveSummary] = useState(null)
  const intervalRef = useRef(null)
  const liveIntervalRef = useRef(null)

  useEffect(() => {
    api.health().then(setHealth).catch(() => {})
    const iv = setInterval(() => api.health().then(setHealth).catch(() => {}), 15000)
    return () => clearInterval(iv)
  }, [])

  const fetchLiveData = useCallback(async () => {
    try {
      const [indiaData, summaryData] = await Promise.all([
        api.realtime.india(),
        api.realtime.summary(),
      ])
      setLiveData(indiaData)
      setLiveSummary(summaryData)
      // Only set aiData from live data when in India mode
      if (mapMode === 'india') {
        setAiData(indiaData)
      }
    } catch (e) {
      console.error('Live data fetch failed:', e)
    }
  }, [mapMode])

  useEffect(() => {
    fetchLiveData()
    liveIntervalRef.current = setInterval(fetchLiveData, 5 * 60 * 1000)
    return () => { if (liveIntervalRef.current) clearInterval(liveIntervalRef.current) }
  }, [fetchLiveData])

  useEffect(() => {
    api.replayStatus().then(d => {
      setReplayData(d)
      setTimestep(d.current_timestep)
      setReplayStepData(d)
    }).catch(() => {})
  }, [])

  const fetchReplayStep = useCallback(async (ts) => {
    if (mapMode !== 'mumbai') return
    setLoading(true)
    try {
      const step = await api.replayStep(ts)
      setReplayStepData(step)
      setReplayData(prev => ({ ...prev, ...step }))
      if (step.ai_pipeline_results) {
        setAiData(step.ai_pipeline_results)
      }
    } catch (e) {
      console.error('Replay step failed:', e)
    }
    setLoading(false)
  }, [mapMode])

  useEffect(() => { fetchReplayStep(timestep) }, [timestep, fetchReplayStep])

  useEffect(() => {
    if (playing && mapMode === 'mumbai') {
      intervalRef.current = setInterval(() => {
        setTimestep(prev => {
          if (prev >= 72) { setPlaying(false); return 72 }
          return prev + 1
        })
      }, 1500)
    } else {
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
    return () => { if (intervalRef.current) clearInterval(intervalRef.current) }
  }, [playing, mapMode])

  const jumpTo = (ts) => { setPlaying(false); setTimestep(ts) }

  const switchMode = (mode) => {
    setMapMode(mode)
    setPlaying(false)
    setSelectedCell(null)
    if (mode === 'india') fetchLiveData()
    else fetchReplayStep(timestep)
  }

  const isOnline = health?.status === 'ONLINE'
  const peakTimestep = replayStepData?.peak_timestep || 36
  const mapData = mapMode === 'india' ? liveData : aiData

  // Determine inference mode for display
  const inferenceMode = aiData?.multi_hazard?.inference_mode || aiData?.flood_depth?.inference_mode || 'heuristic'

  return (
    <div className="app-layout" style={{ gridTemplateRows: '48px 1fr' }}>
      {/* ═══ TOPBAR ═══ */}
      <header className="topbar">
        <div className="brand">
          <span className="brand-icon">🌊</span>
          <div>
            <h1>VARUNA</h1>
            <span className="brand-sub">
              {mapMode === 'india' ? 'India Weather Monitor' : 'Mumbai Flood Digital Twin'}
            </span>
          </div>
        </div>

        <div className="mode-toggle-bar">
          <button className={`mode-toggle ${mapMode === 'india' ? 'active' : ''}`} onClick={() => switchMode('india')}>
            🇮🇳 Live India
          </button>
          <button className={`mode-toggle ${mapMode === 'mumbai' ? 'active' : ''}`} onClick={() => switchMode('mumbai')}>
            🏙️ Mumbai Replay
          </button>
        </div>

        {mapMode === 'mumbai' && (
          <div className="replay-controls-topbar">
            <button className="ctrl-btn" onClick={() => jumpTo(1)}>⏮</button>
            <button className="ctrl-btn" onClick={() => jumpTo(Math.max(1, timestep - 1))} disabled={timestep <= 1}>◀</button>
            <button className={`ctrl-btn ${playing ? 'play-active' : 'play'}`} onClick={() => setPlaying(!playing)}>
              {playing ? '⏸' : '▶'}
            </button>
            <button className="ctrl-btn" onClick={() => jumpTo(Math.min(72, timestep + 1))} disabled={timestep >= 72}>▶</button>
            <button className="ctrl-btn peak" onClick={() => jumpTo(peakTimestep)}>🔥 Peak</button>
            <div className="progress-section">
              <div className="progress-bar">
                <div className="progress-fill" style={{ width: `${(timestep / 72) * 100}%`, background: getPhaseColor(timestep) }} />
              </div>
              <div className="progress-info">
                <span className="phase-tag" style={{ color: getPhaseColor(timestep) }}>{getPhaseLabel(timestep)}</span>
                <span>{timestep}/72</span>
              </div>
            </div>
          </div>
        )}

        <div className="topbar-chips">
          <span className={`chip ${isOnline ? 'chip-ok' : 'chip-err'}`}>
            {isOnline ? '● LIVE' : '● OFFLINE'}
          </span>
          {mapMode === 'mumbai' && (
            <>
              <span className="chip chip-blue">🌊 Tide: {fmt(replayStepData?.tide_height_m)}m</span>
              <span className="chip chip-cyan">🌧 {fmt(replayStepData?.avg_rainfall_1h_mm)} mm/hr</span>
              {aiData?.trust_score && (
                <span className="chip chip-purple">🛡 {fmt(aiData.trust_score.trust_score, 0)}%</span>
              )}
              {/* Inference mode indicator */}
              <span className={`chip ${inferenceMode === 'trained_neural_network' ? 'chip-green' : 'chip-amber'}`}>
                🧠 {inferenceMode === 'trained_neural_network' ? 'Neural Net' : 'Heuristic'}
              </span>
            </>
          )}
          {mapMode === 'india' && liveSummary && (
            <>
              <span className="chip chip-cyan">🇮🇳 {liveSummary.total_cells_monitored || 930} cells</span>
              <span className="chip chip-blue">🌧 {liveSummary.areas_heavy_rain || 0} heavy</span>
            </>
          )}
        </div>
      </header>

      {/* ═══ MAIN: Left Panel + Map + Right Panel ═══ */}
      <div className="app-body three-col">
        {/* LEFT PANEL */}
        <div className="side-panel left-panel">
          <div className="panel-scroll">
            <LeftPanel
              timestep={timestep}
              aiData={mapData}
              replayStepData={replayStepData}
              selectedCell={selectedCell}
              mode={mapMode}
            />
          </div>
        </div>

        {/* CENTER — Map (no overlay tabs) */}
        <div className="center-panel">
          <div className="map-container">
            <MapView
              timestep={timestep}
              overlay="risk"
              aiData={mapData}
              replayStepData={replayStepData}
              selectedCell={selectedCell}
              onSelectCell={setSelectedCell}
              mode={mapMode}
            />
          </div>
        </div>

        {/* RIGHT PANEL */}
        <div className="side-panel right-panel">
          <div className="panel-scroll">
            <IntelligencePanel
              timestep={timestep}
              aiData={mapData}
              replayStepData={replayStepData}
              selectedCell={selectedCell}
              mode={mapMode}
            />
          </div>
        </div>
      </div>
    </div>
  )
}
