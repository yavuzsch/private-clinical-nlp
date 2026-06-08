const BASE = 'http://localhost:8000'

export async function listHospitals() {
  const res = await fetch(`${BASE}/hospitals`)
  if (!res.ok) throw new Error('failed to fetch hospitals')
  return res.json()
}

export async function removeHospital(hospitalId) {
  const res = await fetch(`${BASE}/hospitals/${hospitalId}`, {
    method: 'DELETE',
  })
  if (!res.ok) throw new Error('failed to remove hospital')
  return res.json()
}

export async function getGlobalModel() {
  const res = await fetch(`${BASE}/global-model`)
  if (!res.ok) throw new Error('failed to fetch global model')
  return res.json()
}

export async function getCentralStatus() {
  const res = await fetch(`${BASE}/status`)
  if (!res.ok) throw new Error('failed to fetch system status')
  return res.json()
}

export async function getMetrics() {
  const res = await fetch(`${BASE}/models/metrics`)
  if (!res.ok) throw new Error('failed to fetch metrics')
  return res.json()
}