import { useState, useEffect } from 'react'
import { useAuth } from '../context/AuthContext'
import { getBudget, getHospitalStatus, triggerTraining } from '../api/hospital'
import { getGlobalModel, getCentralStatus } from '../api/central'

function Gauge({ value, max, label, unit }) {
  const pct = Math.min(value / max, 1)
  const color = pct > 0.6 ? '#3a7bd5' : pct > 0.25 ? '#e09a2a' : '#e05c5c'
  const r = 44
  const circ = 2 * Math.PI * r
  const dash = circ * pct

  return (
    <div style={{ textAlign: 'center' }}>
      <svg width={110} height={110} viewBox="0 0 110 110">
        <circle cx={55} cy={55} r={r} fill="none" stroke="#111d2e" strokeWidth={8} />
        <circle
          cx={55} cy={55} r={r} fill="none"
          stroke={color} strokeWidth={8}
          strokeDasharray={`${dash} ${circ}`}
          strokeLinecap="round"
          transform="rotate(-90 55 55)"
          style={{ transition: 'stroke-dasharray 0.6s ease' }}
        />
        <text x={55} y={50} textAnchor="middle" fill={color} fontSize={14} fontFamily="IBM Plex Mono" fontWeight={600}>
          {value.toFixed(2)}
        </text>
        <text x={55} y={66} textAnchor="middle" fill="#3d5166" fontSize={9} fontFamily="IBM Plex Mono">
          {unit}
        </text>
      </svg>
      <div style={{ fontSize: 10, letterSpacing: 2, color: '#3d5166', marginTop: 4 }}>{label}</div>
    </div>
  )
}

export default function Training() {
  const { token, hospitalId } = useAuth()
  const [budget, setBudget] = useState(null)
  const [hospStatus, setHospStatus] = useState(null)
  const [globalModel, setGlobalModel] = useState(null)
  const [sysStatus, setSysStatus] = useState(null)
  const [triggering, setTriggering] = useState(false)
  const [triggerMsg, setTriggerMsg] = useState('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchAll()
  }, [])

  const fetchAll = async () => {
    setLoading(true)
    const [b, s, g, sys] = await Promise.allSettled([
      getBudget(hospitalId, token),
      getHospitalStatus(hospitalId, token),
      getGlobalModel(),
      getCentralStatus(),
    ])
    if (b.status === 'fulfilled') setBudget(b.value)
    if (s.status === 'fulfilled') setHospStatus(s.value)
    if (g.status === 'fulfilled') setGlobalModel(g.value)
    if (sys.status === 'fulfilled') setSysStatus(sys.value)
    setLoading(false)
  }

  const handleTrigger = async () => {
    setTriggering(true)
    setTriggerMsg('')
    try {
      const res = await triggerTraining(hospitalId, token)
      setTriggerMsg(`status: ${res.status} · rounds: ${res.rounds_completed}`)
      setTimeout(fetchAll, 2000)
    } catch {
      setTriggerMsg('trigger failed')
    } finally {
      setTriggering(false)
    }
  }

  if (loading) return <div style={{ color: '#3d5166', fontSize: 12, letterSpacing: 1 }}>loading...</div>

  const epsilon = budget?.epsilon ?? 8.0
  const remaining = budget?.budget_remaining ?? 0
  const isFrozen = budget?.status === 'frozen'

  return (
    <div>
      <div style={{ marginBottom: 32 }}>
        <div style={{ fontSize: 10, letterSpacing: 3, color: '#3a7bd5', marginBottom: 8 }}>
          HOSPITAL_{hospitalId}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <h1 style={{ fontSize: 22, color: '#c8d6e5', fontWeight: 600, margin: 0, flex: 1 }}>
            Training & Privacy
          </h1>
          {isFrozen && (
            <span style={{
              fontSize: 10, letterSpacing: 2, color: '#e05c5c',
              background: '#1a0e0e', border: '1px solid #4a1515',
              borderRadius: 3, padding: '5px 12px',
            }}>
              BUDGET EXHAUSTED — FROZEN
            </span>
          )}
        </div>
      </div>

      {/* budget + stats */}
      <div style={{
        background: '#0d1420', border: '1px solid #1e2d40', borderRadius: 6,
        padding: '28px 32px', marginBottom: 20,
        display: 'flex', alignItems: 'center', gap: 40,
      }}>
        <Gauge value={remaining} max={epsilon} label="BUDGET REMAINING" unit="ε" />
        <div style={{ flex: 1, display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '16px 32px' }}>
          {[
            { label: 'EPSILON (ε)', value: epsilon },
            { label: 'DELTA (δ)', value: '1e-5' },
            { label: 'ROUNDS DONE', value: hospStatus?.rounds_completed ?? 0 },
            { label: 'PENDING NOTES', value: `${hospStatus?.notes_collected ?? 0} / 5`, accent: '#4a9eff' },
            { label: 'MAX GRAD NORM', value: '3.0' },
            { label: 'STATUS', value: isFrozen ? 'FROZEN' : 'ACTIVE', accent: isFrozen ? '#e05c5c' : '#4aaa4a' },
          ].map(({ label, value, accent }) => (
            <div key={label}>
              <div style={{ fontSize: 9, letterSpacing: 2, color: '#3d5166', marginBottom: 6 }}>{label}</div>
              <div style={{ fontSize: 20, color: accent ?? '#c8d6e5', fontWeight: 600 }}>{value}</div>
            </div>
          ))}
        </div>
      </div>

      {/* central status */}
      {sysStatus && (
        <div style={{
          background: '#0d1420', border: '1px solid #1e2d40', borderRadius: 6,
          padding: '20px 24px', marginBottom: 20,
        }}>
          <div style={{ fontSize: 9, letterSpacing: 2, color: '#3d5166', marginBottom: 16 }}>
            SYSTEM — CENTRAL SERVER
          </div>
          <div style={{ display: 'flex', gap: 40 }}>
            {[
              { label: 'ACTIVE', value: sysStatus.active_hospitals, color: '#4aaa4a' },
              { label: 'FROZEN', value: sysStatus.frozen_hospitals, color: '#e05c5c' },
              { label: 'GLOBAL ROUNDS', value: globalModel?.rounds_completed ?? 0, color: '#c8d6e5' },
            ].map(({ label, value, color }) => (
              <div key={label}>
                <div style={{ fontSize: 9, color: '#3d5166', marginBottom: 4 }}>{label}</div>
                <div style={{ fontSize: 20, color }}>{value}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* manual trigger */}
      <div style={{
        background: '#0d1420', border: '1px solid #1e2d40', borderRadius: 6,
        padding: '20px 24px',
      }}>
        <div style={{ fontSize: 9, letterSpacing: 2, color: '#3d5166', marginBottom: 12 }}>
          MANUAL TRAINING TRIGGER
        </div>
        <div style={{ fontSize: 11, color: '#3d5166', marginBottom: 16, lineHeight: 1.7 }}>
          training triggers automatically at 5 pending notes — use this to trigger manually.
        </div>
        <button
          onClick={handleTrigger}
          disabled={triggering || isFrozen}
          style={{
            padding: '10px 24px',
            background: isFrozen ? '#0d1420' : '#1a3a6e',
            border: `1px solid ${isFrozen ? '#1e2d40' : '#2a5aaa'}`,
            borderRadius: 4,
            color: isFrozen ? '#2a3d52' : '#4a9eff',
            fontSize: 11, letterSpacing: 2,
            cursor: isFrozen || triggering ? 'not-allowed' : 'pointer',
            fontFamily: "'IBM Plex Mono', monospace",
          }}
        >
          {triggering ? 'TRIGGERING...' : 'TRIGGER TRAINING →'}
        </button>
        {triggerMsg && (
          <div style={{ marginTop: 12, fontSize: 11, color: '#5a7a99', letterSpacing: 1 }}>
            → {triggerMsg}
          </div>
        )}
      </div>
    </div>
  )
}