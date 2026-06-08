import { useState, useEffect } from 'react'
import { listHospitals, removeHospital, getCentralStatus } from '../api/central'

export default function Hospitals() {
  const [hospitals, setHospitals] = useState([])
  const [epsilon, setEpsilon] = useState(8.0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [removing, setRemoving] = useState(null)

  useEffect(() => {
    fetchAll()
  }, [])

  const fetchAll = async () => {
    setLoading(true)
    setError('')
    try {
      const [hospitalsData, sysData] = await Promise.all([
        listHospitals(),
        getCentralStatus(),
      ])
      setHospitals(hospitalsData)
      setEpsilon(sysData.epsilon)
    } catch {
      setError('failed to fetch — is central server running?')
    } finally {
      setLoading(false)
    }
  }

  const handleRemove = async (id) => {
    setRemoving(id)
    try {
      await removeHospital(id)
      setHospitals(h => h.filter(x => x.hospital_id !== id))
    } catch {
      setError(`failed to remove hospital_${id}`)
    } finally {
      setRemoving(null)
    }
  }

  const active = hospitals.filter(h => h.status === 'active')
  const frozen = hospitals.filter(h => h.status === 'frozen')

  return (
    <div>
      <div style={{ marginBottom: 32 }}>
        <div style={{ fontSize: 10, letterSpacing: 3, color: '#3a7bd5', marginBottom: 8 }}>
          CENTRAL SERVER
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <h1 style={{ fontSize: 22, color: '#c8d6e5', fontWeight: 600, margin: 0, flex: 1 }}>
            Hospital Registry
          </h1>
          <button
            onClick={fetchAll}
            style={{
              padding: '7px 16px', background: 'transparent',
              border: '1px solid #1e2d40', borderRadius: 4,
              color: '#3d5166', fontSize: 10, letterSpacing: 2,
              cursor: 'pointer', fontFamily: "'IBM Plex Mono', monospace",
            }}
          >
            REFRESH
          </button>
        </div>
        <div style={{ marginTop: 8, fontSize: 11, color: '#3d5166' }}>
          {active.length} active · {frozen.length} frozen · {hospitals.length} total
        </div>
      </div>

      {error && (
        <div style={{
          padding: '12px 16px', background: '#1a0e0e', border: '1px solid #4a1515',
          borderRadius: 4, fontSize: 11, color: '#e05c5c', marginBottom: 20,
        }}>
          ✕ {error}
        </div>
      )}

      {loading ? (
        <div style={{ color: '#3d5166', fontSize: 12, letterSpacing: 1 }}>loading...</div>
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {hospitals.map(h => {
            const isFrozen = h.status === 'frozen'
            return (
              <div key={h.hospital_id} style={{
                background: '#0d1420',
                border: `1px solid ${isFrozen ? '#1e3a1e' : '#1e2d40'}`,
                borderRadius: 6,
                padding: '20px 24px',
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16 }}>
                  <div style={{
                    width: 8, height: 8, borderRadius: '50%',
                    background: isFrozen ? '#e05c5c' : '#4aaa4a',
                    flexShrink: 0,
                  }} />
                  <div style={{ fontSize: 13, color: '#c8d6e5', fontWeight: 600, flex: 1 }}>
                    Hospital_{h.hospital_id}
                  </div>
                  <span style={{
                    fontSize: 9, letterSpacing: 2, padding: '3px 8px',
                    color: isFrozen ? '#e05c5c' : '#4aaa4a',
                    background: isFrozen ? '#1a0e0e' : '#0a1e0a',
                    border: `1px solid ${isFrozen ? '#4a1515' : '#2a5a2a'}`,
                    borderRadius: 3,
                  }}>
                    {h.status.toUpperCase()}
                  </span>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px 24px', marginBottom: 16 }}>
                  {[
                    { label: 'PORT', value: h.port },
                    { label: 'ROUNDS', value: h.rounds_completed },
                    { label: 'NOTES', value: h.notes_collected },
                    { label: 'BUDGET ε', value: h.budget_remaining.toFixed(3), accent: isFrozen ? '#e05c5c' : '#4a9eff' },
                  ].map(({ label, value, accent }) => (
                    <div key={label}>
                      <div style={{ fontSize: 9, color: '#3d5166', marginBottom: 3, letterSpacing: 1 }}>{label}</div>
                      <div style={{ fontSize: 13, color: accent ?? '#7a9ab5' }}>{value}</div>
                    </div>
                  ))}
                </div>

                <div style={{ marginBottom: 16 }}>
                  <div style={{ fontSize: 9, color: '#3d5166', marginBottom: 6, letterSpacing: 1 }}>PRIVACY BUDGET</div>
                  <div style={{ height: 4, background: '#111d2e', borderRadius: 2 }}>
                    <div style={{
                      height: '100%',
                      width: `${Math.min((h.budget_remaining / epsilon) * 100, 100)}%`,
                      background: isFrozen ? '#e05c5c' : '#3a7bd5',
                      borderRadius: 2,
                      transition: 'width 0.4s',
                    }} />
                  </div>
                </div>

                <button
                  onClick={() => handleRemove(h.hospital_id)}
                  disabled={removing === h.hospital_id}
                  style={{
                    padding: '6px 14px', background: 'transparent',
                    border: '1px solid #2a1515', borderRadius: 3,
                    color: '#aa4a4a', fontSize: 10, letterSpacing: 1,
                    cursor: removing === h.hospital_id ? 'not-allowed' : 'pointer',
                    fontFamily: "'IBM Plex Mono', monospace",
                  }}
                >
                  {removing === h.hospital_id ? 'REMOVING...' : 'REMOVE'}
                </button>
              </div>
            )
          })}
        </div>
      )}

      {!loading && hospitals.length === 0 && !error && (
        <div style={{
          padding: '48px', background: '#0d1420', border: '1px solid #1e2d40',
          borderRadius: 6, textAlign: 'center', color: '#2a3d52', fontSize: 12, letterSpacing: 1,
        }}>
          no hospitals registered
        </div>
      )}
    </div>
  )
}