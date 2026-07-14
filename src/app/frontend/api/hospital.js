const base = (hospitalId) => `/hospital/${hospitalId}`

export async function loginHospital(hospitalId, password) {
  const res = await fetch(`${base(hospitalId)}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ hospital_id: hospitalId, password }),
  })
  if (!res.ok) throw new Error('invalid credentials')
  return res.json()
}

export async function getCategories(hospitalId) {
  const res = await fetch(`${base(hospitalId)}/categories`)
  if (!res.ok) throw new Error('failed to fetch categories')
  return res.json()
}

export async function predict(hospitalId, token, text) {
  const res = await fetch(`${base(hospitalId)}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
    body: JSON.stringify({ text }),
  })
  if (!res.ok) throw new Error('prediction failed')
  return res.json()
}

export async function getNotes(hospitalId, token) {
  const res = await fetch(`${base(hospitalId)}/notes`, {
    headers: { Authorization: `Bearer ${token}` },
  })
  if (!res.ok) throw new Error('failed to fetch notes')
  return res.json()
}

export async function addNote(hospitalId, token, text, predictions) {
  const res = await fetch(`${base(hospitalId)}/notes`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
    body: JSON.stringify({ text, predictions }),
  })
  if (!res.ok) throw new Error('failed to add note')
  return res.json()
}

export async function deleteNote(hospitalId, token, noteId) {
  const res = await fetch(`${base(hospitalId)}/notes/${noteId}`, {
    method: 'DELETE',
    headers: { Authorization: `Bearer ${token}` },
  })
  if (!res.ok) throw new Error('failed to delete note')
  return res.json()
}

export async function updateNote(hospitalId, token, noteId, { text, predictions }) {
  const res = await fetch(`${base(hospitalId)}/notes/${noteId}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
    body: JSON.stringify({ text, predictions }),
  })
  if (!res.ok) throw new Error('failed to update note')
  return res.json()
}

export async function triggerTraining(hospitalId, token) {
  const res = await fetch(`${base(hospitalId)}/train`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${token}` },
  })
  if (!res.ok) throw new Error('failed to trigger training')
  return res.json()
}

export async function getBudget(hospitalId, token) {
  const res = await fetch(`${base(hospitalId)}/budget`, {
    headers: { Authorization: `Bearer ${token}` },
  })
  if (!res.ok) throw new Error('failed to fetch budget')
  return res.json()
}

export async function getHospitalStatus(hospitalId, token) {
  const res = await fetch(`${base(hospitalId)}/status`, {
    headers: { Authorization: `Bearer ${token}` },
  })
  if (!res.ok) throw new Error('failed to fetch status')
  return res.json()
}