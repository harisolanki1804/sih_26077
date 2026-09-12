import GeoMap from './GeoMap'

export default function MapView({ timestep, overlay, aiData, replayStepData, selectedCell, onSelectCell, mode }) {
  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      {/* Legend — bottom-left */}
      <div className="map-legend-container">
        <div className="map-legend">
          {mode === 'india' ? (
            <>
              <div className="legend-title">Rainfall Intensity</div>
              <div className="legend-item"><span className="dot" style={{ background: '#ef4444' }} /> Extreme (&gt;80mm)</div>
              <div className="legend-item"><span className="dot" style={{ background: '#f97316' }} /> Very Heavy (50-80)</div>
              <div className="legend-item"><span className="dot" style={{ background: '#eab308' }} /> Heavy (25-50)</div>
              <div className="legend-item"><span className="dot" style={{ background: '#06b6d4' }} /> Moderate (10-25)</div>
              <div className="legend-item"><span className="dot" style={{ background: '#3b82f6' }} /> Light (1-10)</div>
            </>
          ) : (
            <>
              <div className="legend-title">Flood Risk Level</div>
              <div className="legend-item"><span className="dot" style={{ background: '#ef4444' }} /> Critical (80+)</div>
              <div className="legend-item"><span className="dot" style={{ background: '#f97316' }} /> High (60-80)</div>
              <div className="legend-item"><span className="dot" style={{ background: '#eab308' }} /> Medium (35-60)</div>
              <div className="legend-item"><span className="dot" style={{ background: '#22c55e' }} /> Low (&lt;35)</div>
              <div className="legend-item"><span className="dot" style={{ background: '#14532d' }} /> Minimal</div>
            </>
          )}
        </div>
      </div>

      <GeoMap
        timestep={timestep}
        overlay={overlay}
        aiData={aiData}
        selectedCell={selectedCell}
        onSelectCell={onSelectCell}
        mode={mode}
      />
    </div>
  )
}
