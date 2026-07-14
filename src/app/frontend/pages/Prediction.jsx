import { useState, useEffect } from 'react'
import { useAuth } from '../context/AuthContext'
import { predict, addNote, getCategories } from '../api/hospital'

export default function Prediction() {
  const { token, hospitalId } = useAuth()
  const [text, setText] = useState('')
  const [result, setResult] = useState(null)
  const [categories, setCategories] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [saved, setSaved] = useState(false)
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    getCategories(hospitalId).then(data => setCategories(data.categories)).catch(() => {})
  }, [hospitalId])

  const handlePredict = async () => {
    if (!text.trim()) return
    setLoading(true)
    setError('')
    setResult(null)
    setSaved(false)
    try {
      const data = await predict(hospitalId, token, text)
      setResult(data)
    } catch {
      setError('prediction failed — is the hospital server running?')
    } finally {
      setLoading(false)
    }
  }

  const handleSave = async () => {
    if (!result) return
    setSaving(true)
    try {
      await addNote(hospitalId, token, text, result.predictions)
      setSaved(true)
    } catch {
      setError('failed to save note')
    } finally {
      setSaving(false)
    }
  }

  const toggleCategory = (idx) => {
    setResult(prev => {
      const updated = [...prev.predictions]
      updated[idx] = updated[idx] === 1 ? 0 : 1
      return { ...prev, predictions: updated }
    })
    setSaved(false)
  }

  return (
    <div>
      <div style={{ marginBottom: 36 }}>
        <div style={{ fontSize: 10, letterSpacing: 3, color: '#3a7bd5', marginBottom: 8 }}>
          HOSPITAL_{hospitalId}
        </div>
        <h1 style={{ fontSize: 22, color: '#c8d6e5', fontWeight: 600, margin: 0 }}>
          ICD-9 Prediction
        </h1>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24, alignItems: 'start' }}>
        {/* input */}
        <div>
          <div style={{ fontSize: 10, letterSpacing: 2, color: '#3d5166', marginBottom: 10 }}>
            CLINICAL NOTE
          </div>
          <textarea
            value={text}
            onChange={e => setText(e.target.value)}
            placeholder="Enter clinical note..."
            style={{
              width: '100%',
              height: 280,
              background: '#0d1420',
              border: '1px solid #1e2d40',
              borderRadius: 4,
              padding: '16px',
              color: '#c8d6e5',
              fontSize: 12,
              fontFamily: "'IBM Plex Mono', monospace",
              resize: 'vertical',
              outline: 'none',
              boxSizing: 'border-box',
              lineHeight: 1.7,
            }}
          />
          <button
            onClick={handlePredict}
            disabled={loading || !text.trim()}
            style={{
              marginTop: 12,
              width: '100%',
              padding: '12px',
              background: loading ? '#0d1420' : '#1a3a6e',
              border: '1px solid #2a5aaa',
              borderRadius: 4,
              color: loading ? '#3d5166' : '#4a9eff',
              fontSize: 12,
              letterSpacing: 2,
              cursor: loading || !text.trim() ? 'not-allowed' : 'pointer',
              fontFamily: "'IBM Plex Mono', monospace",
            }}
          >
            {loading ? 'RUNNING INFERENCE...' : 'PREDICT →'}
          </button>

          {error && (
            <div style={{
              marginTop: 12,
              padding: '10px 14px',
              background: '#1a0e0e',
              border: '1px solid #4a1515',
              borderRadius: 4,
              fontSize: 11,
              color: '#e05c5c',
            }}>
              ✕ {error}
            </div>
          )}
        </div>

        {/* results */}
        <div>
          <div style={{ fontSize: 10, letterSpacing: 2, color: '#3d5166', marginBottom: 10 }}>
            PREDICTED CATEGORIES
          </div>

          {!result && (
            <div style={{
              height: 280,
              background: '#0d1420',
              border: '1px solid #1e2d40',
              borderRadius: 4,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#2a3d52',
              fontSize: 11,
              letterSpacing: 1,
            }}>
              {loading ? 'running local inference...' : 'awaiting input'}
            </div>
          )}

          {result && (
            <>
              <div style={{
                background: '#0d1420',
                border: '1px solid #1e2d40',
                borderRadius: 4,
                padding: '16px',
                maxHeight: 280,
                overflowY: 'auto',
              }}>
                {result.probabilities.map((prob, i) => {
                  const active = result.predictions[i] === 1
                  const label = categories[i] ?? `label_${i}`
                  return (
                    <div
                      key={i}
                      onClick={() => toggleCategory(i)}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: 10,
                        padding: '7px 0',
                        borderBottom: '1px solid #111d2e',
                        cursor: 'pointer',
                      }}
                    >
                      <div style={{
                        width: 8,
                        height: 8,
                        borderRadius: 2,
                        background: active ? '#3a7bd5' : '#1e2d40',
                        flexShrink: 0,
                        transition: 'background 0.15s',
                      }} />
                      <div style={{ flex: 1, fontSize: 11, color: active ? '#c8d6e5' : '#3d5166' }}>
                        {label}
                      </div>
                      <div style={{ width: 80, height: 4, background: '#111d2e', borderRadius: 2, flexShrink: 0 }}>
                        <div style={{
                          height: '100%',
                          width: `${prob * 100}%`,
                          background: active ? '#3a7bd5' : '#2a3d52',
                          borderRadius: 2,
                        }} />
                      </div>
                      <div style={{ fontSize: 10, color: '#3d5166', width: 36, textAlign: 'right', flexShrink: 0 }}>
                        {(prob * 100).toFixed(0)}%
                      </div>
                    </div>
                  )
                })}
              </div>

              <button
                onClick={handleSave}
                disabled={saving || saved}
                style={{
                  marginTop: 12,
                  width: '100%',
                  padding: '12px',
                  background: saved ? '#0a1e0a' : '#0d1420',
                  border: `1px solid ${saved ? '#2a5a2a' : '#1e2d40'}`,
                  borderRadius: 4,
                  color: saved ? '#4aaa4a' : '#5a7a99',
                  fontSize: 12,
                  letterSpacing: 2,
                  cursor: saving || saved ? 'default' : 'pointer',
                  fontFamily: "'IBM Plex Mono', monospace",
                }}
              >
                {saved ? '✓ SAVED TO NOTES' : saving ? 'SAVING...' : 'CONFIRM & SAVE NOTE'}
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  )
}