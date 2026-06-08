import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import { loginHospital } from '../api/hospital'

export default function Login() {
  const [hospitalId, setHospitalId] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const { login } = useAuth()
  const navigate = useNavigate()

  const handleSubmit = async () => {
    const id = parseInt(hospitalId)
    if (isNaN(id) || id < 0 || id > 9) {
      setError('hospital id must be 0–9')
      return
    }
    if (!password) {
      setError('password required')
      return
    }
    setError('')
    setLoading(true)
    try {
      const data = await loginHospital(id, password)
      login(data.access_token, data.hospital_id)
      navigate('/predict')
    } catch {
      setError('invalid credentials')
    } finally {
      setLoading(false)
    }
  }

  const inputStyle = {
    width: '100%',
    background: '#0d1420',
    border: '1px solid #1e2d40',
    borderRadius: 4,
    padding: '12px 16px',
    color: '#c8d6e5',
    fontSize: 13,
    fontFamily: "'IBM Plex Mono', monospace",
    outline: 'none',
    boxSizing: 'border-box',
  }

  return (
    <div style={{
      minHeight: '100vh',
      background: '#0a0e17',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      fontFamily: "'IBM Plex Mono', monospace",
    }}>
      <div style={{
        position: 'fixed',
        inset: 0,
        backgroundImage: 'linear-gradient(#1a2535 1px, transparent 1px), linear-gradient(90deg, #1a2535 1px, transparent 1px)',
        backgroundSize: '48px 48px',
        opacity: 0.3,
        pointerEvents: 'none',
      }} />

      <div style={{
        width: 380,
        background: '#0d1420',
        border: '1px solid #1e2d40',
        borderRadius: 8,
        padding: '40px 36px',
        position: 'relative',
        zIndex: 1,
      }}>
        <div style={{ marginBottom: 36 }}>
          <div style={{ fontSize: 10, letterSpacing: 3, color: '#3a7bd5', marginBottom: 8 }}>
            PRIVACYNLP
          </div>
          <div style={{ fontSize: 20, color: '#c8d6e5', fontWeight: 600, marginBottom: 6 }}>
            Hospital Login
          </div>
          <div style={{ fontSize: 11, color: '#3d5166', lineHeight: 1.6 }}>
            federated · differential privacy · icd-9
          </div>
        </div>

        <div style={{ marginBottom: 20 }}>
          <div style={{ fontSize: 10, letterSpacing: 2, color: '#3d5166', marginBottom: 8 }}>
            HOSPITAL ID (0–9)
          </div>
          <input
            type="number"
            min={0}
            max={9}
            value={hospitalId}
            onChange={e => setHospitalId(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleSubmit()}
            placeholder="0"
            style={inputStyle}
          />
        </div>

        <div style={{ marginBottom: 28 }}>
          <div style={{ fontSize: 10, letterSpacing: 2, color: '#3d5166', marginBottom: 8 }}>
            PASSWORD
          </div>
          <input
            type="password"
            value={password}
            onChange={e => setPassword(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleSubmit()}
            placeholder="••••••••"
            style={inputStyle}
          />
        </div>

        {error && (
          <div style={{
            marginBottom: 20,
            padding: '10px 14px',
            background: '#1a0e0e',
            border: '1px solid #4a1515',
            borderRadius: 4,
            fontSize: 11,
            color: '#e05c5c',
            letterSpacing: 1,
          }}>
            ✕ {error}
          </div>
        )}

        <button
          onClick={handleSubmit}
          disabled={loading}
          style={{
            width: '100%',
            padding: '13px',
            background: loading ? '#0d1420' : '#1a3a6e',
            border: '1px solid #2a5aaa',
            borderRadius: 4,
            color: loading ? '#3d5166' : '#4a9eff',
            fontSize: 12,
            letterSpacing: 2,
            cursor: loading ? 'not-allowed' : 'pointer',
            fontFamily: "'IBM Plex Mono', monospace",
          }}
        >
          {loading ? 'AUTHENTICATING...' : 'LOGIN →'}
        </button>

        <div style={{ marginTop: 24, fontSize: 10, color: '#2a3d52', textAlign: 'center', lineHeight: 1.8 }}>
          patient data never leaves the hospital<br />
          only model weights are shared
        </div>
      </div>
    </div>
  )
}