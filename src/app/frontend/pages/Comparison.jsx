import { useState, useEffect } from 'react'
import { getMetrics } from '../api/central'

const METRICS = [
  { key: 'auc', label: 'AUC' },
  { key: 'f1_macro', label: 'F1 Macro' },
  { key: 'f1_micro', label: 'F1 Micro' },
  { key: 'precision_macro', label: 'Precision' },
  { key: 'recall_macro', label: 'Recall' },
  { key: 'hamming_loss', label: 'Hamming ↓' },
]

const MODEL_COLORS = ['#4a9eff', '#3a7bd5', '#e09a2a', '#d07020', '#aa5010']

function Bar({ value, max, color }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
      <div style={{ width: 120, height: 6, background: '#111d2e', borderRadius: 3, flexShrink: 0 }}>
        <div style={{
          height: '100%',
          width: `${(value / max) * 100}%`,
          background: color,
          borderRadius: 3,
          transition: 'width 0.5s ease',
        }} />
      </div>
      <span style={{ fontSize: 11, color: '#7a9ab5', width: 48 }}>{value.toFixed(4)}</span>
    </div>
  )
}

function PrivacyBadge({ epsilon }) {
  if (epsilon === null || epsilon === undefined)
    return <span style={{ fontSize: 9, letterSpacing: 2, padding: '3px 8px', background: '#1a0e0e', border: '1px solid #4a1515', color: '#e05c5c', borderRadius: 3 }}>NO PRIVACY</span>
  if (epsilon <= 1)
    return <span style={{ fontSize: 9, letterSpacing: 2, padding: '3px 8px', background: '#0a1e0a', border: '1px solid #2a5a2a', color: '#4aaa4a', borderRadius: 3 }}>STRONG</span>
  if (epsilon <= 3)
    return <span style={{ fontSize: 9, letterSpacing: 2, padding: '3px 8px', background: '#1a1400', border: '1px solid #4a3a00', color: '#e09a2a', borderRadius: 3 }}>MODERATE</span>
  return <span style={{ fontSize: 9, letterSpacing: 2, padding: '3px 8px', background: '#150e00', border: '1px solid #3a2a00', color: '#aa6010', borderRadius: 3 }}>WEAK</span>
}

export default function Comparison() {
  const [models, setModels] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [highlight, setHighlight] = useState(null)

  useEffect(() => {
    getMetrics()
      .then(data => setModels(data))
      .catch(() => setError('failed to fetch metrics — is central server running?'))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ color: '#3d5166', fontSize: 12, letterSpacing: 1 }}>loading...</div>

  const shownMetrics = highlight ? METRICS.filter(m => m.key === highlight) : METRICS

  return (
    <div>
      <div style={{ marginBottom: 32 }}>
        <div style={{ fontSize: 10, letterSpacing: 3, color: '#3a7bd5', marginBottom: 8 }}>
          CENTRAL SERVER
        </div>
        <h1 style={{ fontSize: 22, color: '#c8d6e5', fontWeight: 600, margin: 0 }}>
          Model Comparison
        </h1>
        <div style={{ marginTop: 8, fontSize: 11, color: '#3d5166' }}>
          privacy–utility tradeoff · tested on global held-out set
        </div>
      </div>

      {error && (
        <div style={{
          padding: '12px 16px', background: '#1a0e0e', border: '1px solid #4a1515',
          borderRadius: 4, fontSize: 11, color: '#e05c5c', marginBottom: 24,
        }}>
          ✕ {error}
        </div>
      )}

      {/* metric filter */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 24, flexWrap: 'wrap' }}>
        <button onClick={() => setHighlight(null)} style={pillStyle(!highlight)}>ALL</button>
        {METRICS.map(m => (
          <button key={m.key} onClick={() => setHighlight(highlight === m.key ? null : m.key)} style={pillStyle(highlight === m.key)}>
            {m.label.toUpperCase()}
          </button>
        ))}
      </div>

      {/* model cards */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        {models.map((model, mi) => {
          const color = MODEL_COLORS[mi % MODEL_COLORS.length]
          return (
            <div key={model.model_name} style={{
              background: '#0d1420',
              border: `1px solid ${color}33`,
              borderLeft: `3px solid ${color}`,
              borderRadius: 6,
              padding: '20px 24px',
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 16 }}>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: 13, color, fontWeight: 600, marginBottom: 3 }}>
                    {model.model_name}
                  </div>
                  <div style={{ fontSize: 10, color: '#3d5166', letterSpacing: 1 }}>
                    {model.epsilon != null ? `ε = ${model.epsilon} · δ = 1e-5` : 'no differential privacy'}
                  </div>
                </div>
                <PrivacyBadge epsilon={model.epsilon} />
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '12px 24px' }}>
                {shownMetrics.map(m => {
                  const val = model[m.key]
                  const allVals = models.map(x => x[m.key]).filter(v => v != null)
                  const maxVal = Math.max(...allVals) || 1
                  return (
                    <div key={m.key}>
                      <div style={{ fontSize: 9, letterSpacing: 1, color: '#3d5166', marginBottom: 6 }}>
                        {m.label.toUpperCase()}
                      </div>
                      {val != null
                        ? <Bar value={val} max={maxVal} color={color} />
                        : <span style={{ fontSize: 11, color: '#2a3d52' }}>—</span>
                      }
                    </div>
                  )
                })}
              </div>
            </div>
          )
        })}
      </div>

      {!loading && models.length === 0 && !error && (
        <div style={{
          padding: '48px', background: '#0d1420', border: '1px solid #1e2d40',
          borderRadius: 6, textAlign: 'center', color: '#2a3d52', fontSize: 12, letterSpacing: 1,
        }}>
          no model metrics available — run evaluation scripts first
        </div>
      )}
    </div>
  )
}

function pillStyle(active) {
  return {
    padding: '5px 12px', fontSize: 10, letterSpacing: 1,
    background: active ? '#1a3a6e' : 'transparent',
    border: `1px solid ${active ? '#2a5aaa' : '#1e2d40'}`,
    color: active ? '#4a9eff' : '#3d5166',
    borderRadius: 3, cursor: 'pointer',
    fontFamily: "'IBM Plex Mono', monospace",
  }
}