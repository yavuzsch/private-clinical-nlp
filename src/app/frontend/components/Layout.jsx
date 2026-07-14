import { NavLink, useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'

const NAV_ITEMS = [
  { to: '/predict', label: 'Prediction' },
  { to: '/notes', label: 'Notes' },
  { to: '/training', label: 'Training' },
  { to: '/comparison', label: 'Comparison' },
  { to: '/hospitals', label: 'Hospitals' },
]

export default function Layout({ children }) {
  const { hospitalId, logout } = useAuth()
  const navigate = useNavigate()

  const handleLogout = () => {
    logout()
    navigate('/login')
  }

  return (
    <div style={{ display: 'flex', minHeight: '100vh', background: '#0a0e17', color: '#c8d6e5', fontFamily: "'IBM Plex Mono', monospace" }}>
      <aside style={{
        width: 220,
        minHeight: '100vh',
        background: '#0d1420',
        borderRight: '1px solid #1e2d40',
        display: 'flex',
        flexDirection: 'column',
        padding: '32px 0 24px',
        position: 'fixed',
        top: 0,
        left: 0,
        bottom: 0,
        zIndex: 10,
      }}>
        <div style={{ padding: '0 24px 32px' }}>
          <div style={{ fontSize: 10, letterSpacing: 3, color: '#3a7bd5', marginBottom: 4 }}>
            PrivacyNLP
          </div>
          <div style={{ fontSize: 11, color: '#3d5166' }}>
            federated · dp
          </div>
        </div>

        <div style={{ margin: '0 16px 28px', padding: '10px 14px', background: '#111d2e', border: '1px solid #1e3150', borderRadius: 4 }}>
          <div style={{ fontSize: 9, color: '#3d5166', letterSpacing: 2, marginBottom: 4 }}>ACTIVE HOSPITAL</div>
          <div style={{ fontSize: 13, color: '#4a9eff', fontWeight: 600 }}>Hospital_{hospitalId}</div>
        </div>

        <nav style={{ flex: 1 }}>
          {NAV_ITEMS.map(({ to, label }) => (
            <NavLink
              key={to}
              to={to}
              style={({ isActive }) => ({
                display: 'flex',
                alignItems: 'center',
                padding: '11px 24px',
                fontSize: 12,
                letterSpacing: 1,
                color: isActive ? '#4a9eff' : '#5a7a99',
                textDecoration: 'none',
                background: isActive ? '#0a1828' : 'transparent',
                borderLeft: isActive ? '2px solid #3a7bd5' : '2px solid transparent',
                transition: 'all 0.15s',
              })}
            >
              {label.toUpperCase()}
            </NavLink>
          ))}
        </nav>

        <button
          onClick={handleLogout}
          style={{
            margin: '0 16px',
            padding: '10px',
            background: 'transparent',
            border: '1px solid #1e2d40',
            borderRadius: 4,
            color: '#3d5166',
            fontSize: 11,
            letterSpacing: 2,
            cursor: 'pointer',
            fontFamily: "'IBM Plex Mono', monospace",
          }}
        >
          LOGOUT
        </button>
      </aside>

      <main style={{ marginLeft: 220, flex: 1, padding: '40px 48px' }}>
        {children}
      </main>
    </div>
  )
}