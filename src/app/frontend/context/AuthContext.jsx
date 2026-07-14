import { createContext, useContext, useState } from 'react'

const AuthContext = createContext(null)

export function AuthProvider({ children }) {
  const [token, setToken] = useState(localStorage.getItem('token') || null)
  const [hospitalId, setHospitalId] = useState(
    localStorage.getItem('hospitalId') !== null
      ? parseInt(localStorage.getItem('hospitalId'))
      : null
  )

  const login = (newToken, newHospitalId) => {
    setToken(newToken)
    setHospitalId(newHospitalId)
    localStorage.setItem('token', newToken)
    localStorage.setItem('hospitalId', String(newHospitalId))
  }

  const logout = () => {
    setToken(null)
    setHospitalId(null)
    localStorage.removeItem('token')
    localStorage.removeItem('hospitalId')
  }

  return (
    <AuthContext.Provider value={{ token, hospitalId, login, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  return useContext(AuthContext)
}