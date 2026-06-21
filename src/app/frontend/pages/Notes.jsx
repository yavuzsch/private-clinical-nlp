import { useState, useEffect } from 'react'
import { useAuth } from '../context/AuthContext'
import { getNotes, deleteNote, updateNote, getCategories } from '../api/hospital'

function NoteCard({ note, categories, onDelete, onUpdate }) {
  const [editing, setEditing] = useState(false)
  const [editText, setEditText] = useState(note.text)
  const [editPreds, setEditPreds] = useState([...note.predictions])
  const [saving, setSaving] = useState(false)

  const toggleCat = (i) => {
    const next = [...editPreds]
    next[i] = next[i] === 1 ? 0 : 1
    setEditPreds(next)
  }

  const handleSave = async () => {
    setSaving(true)
    try {
      await onUpdate(note.id, editText, editPreds)
      setEditing(false)
    } finally {
      setSaving(false)
    }
  }

  const handleCancel = () => {
    setEditText(note.text)
    setEditPreds([...note.predictions])
    setEditing(false)
  }

  return (
    <div style={{
      background: '#0d1420',
      border: `1px solid ${note.used_in_training ? '#1e3a1e' : '#1e2d40'}`,
      borderRadius: 6,
      padding: '20px 24px',
      marginBottom: 16,
    }}>
      {/* header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 14 }}>
        <div style={{ fontSize: 10, color: '#3d5166', flex: 1, letterSpacing: 1 }}>
          {new Date(note.created_at).toLocaleString()}
        </div>
        {note.used_in_training && (
          <span style={{
            fontSize: 9, letterSpacing: 2, color: '#4aaa4a',
            background: '#0a1e0a', border: '1px solid #2a5a2a',
            borderRadius: 3, padding: '3px 8px',
          }}>
            USED IN TRAINING
          </span>
        )}
        {!editing && (
          <>
            <button onClick={() => setEditing(true)} style={btnStyle('#1e2d40', '#5a7a99')}>EDIT</button>
            <button onClick={() => onDelete(note.id)} style={btnStyle('#2a1515', '#aa4a4a')}>DELETE</button>
          </>
        )}
      </div>

      {/* text */}
      {editing ? (
        <textarea
          value={editText}
          onChange={e => setEditText(e.target.value)}
          style={{
            width: '100%',
            height: 120,
            background: '#111d2e',
            border: '1px solid #2a3d52',
            borderRadius: 4,
            padding: '12px',
            color: '#c8d6e5',
            fontSize: 12,
            fontFamily: "'IBM Plex Mono', monospace",
            resize: 'vertical',
            outline: 'none',
            boxSizing: 'border-box',
            marginBottom: 14,
            lineHeight: 1.6,
          }}
        />
      ) : (
        <div style={{
          fontSize: 12, color: '#7a9ab5', lineHeight: 1.7, marginBottom: 14,
          maxHeight: 80, overflow: 'hidden',
          display: '-webkit-box', WebkitLineClamp: 3, WebkitBoxOrient: 'vertical',
        }}>
          {note.text}
        </div>
      )}

      {/* categories */}
      {editing ? (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 16 }}>
          {categories.map((cat, i) => (
            <button
              key={cat}
              onClick={() => toggleCat(i)}
              style={{
                padding: '4px 10px', fontSize: 10, letterSpacing: 1,
                background: editPreds[i] === 1 ? '#1a3a6e' : '#111d2e',
                border: `1px solid ${editPreds[i] === 1 ? '#2a5aaa' : '#1e2d40'}`,
                color: editPreds[i] === 1 ? '#4a9eff' : '#3d5166',
                borderRadius: 3, cursor: 'pointer',
                fontFamily: "'IBM Plex Mono', monospace",
              }}
            >
              {cat}
            </button>
          ))}
        </div>
      ) : (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
          {note.categories.map(cat => (
            <span key={cat} style={{
              padding: '3px 9px', fontSize: 10, letterSpacing: 1,
              background: '#111d2e', border: '1px solid #1e3150',
              color: '#4a9eff', borderRadius: 3,
            }}>
              {cat}
            </span>
          ))}
        </div>
      )}

      {editing && (
        <div style={{ display: 'flex', gap: 8, marginTop: 14 }}>
          <button onClick={handleSave} disabled={saving} style={btnStyle('#1a3a6e', '#4a9eff')}>
            {saving ? 'SAVING...' : 'SAVE'}
          </button>
          <button onClick={handleCancel} style={btnStyle('#1e2d40', '#5a7a99')}>CANCEL</button>
        </div>
      )}
    </div>
  )
}

function btnStyle(bg, color) {
  return {
    padding: '5px 12px', background: bg,
    border: `1px solid ${color}33`, borderRadius: 3,
    color, fontSize: 10, letterSpacing: 1,
    cursor: 'pointer', fontFamily: "'IBM Plex Mono', monospace",
  }
}

export default function Notes() {
  const { token, hospitalId } = useAuth()
  const [notes, setNotes] = useState([])
  const [categories, setCategories] = useState([])
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState('all')

  useEffect(() => {
    fetchAll()
  }, [])

  const fetchAll = async () => {
    setLoading(true)
    try {
      const [notesData, catsData] = await Promise.all([
        getNotes(hospitalId, token),
        getCategories(hospitalId),
      ])
      setNotes(notesData.reverse())
      setCategories(catsData.categories)
    } catch {
      // ignore
    } finally {
      setLoading(false)
    }
  }

  const handleDelete = async (id) => {
    await deleteNote(hospitalId, token, id)
    setNotes(n => n.filter(x => x.id !== id))
  }

  const handleUpdate = async (id, text, predictions) => {
    const updated = await updateNote(hospitalId, token, id, { text, predictions })
    setNotes(n => n.map(x => x.id === id ? updated : x))
  }

  const filtered = notes.filter(n => {
    if (filter === 'pending') return !n.used_in_training
    if (filter === 'used') return n.used_in_training
    return true
  })

  return (
    <div>
      <div style={{ marginBottom: 32 }}>
        <div style={{ fontSize: 10, letterSpacing: 3, color: '#3a7bd5', marginBottom: 8 }}>
          HOSPITAL_{hospitalId}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <h1 style={{ fontSize: 22, color: '#c8d6e5', fontWeight: 600, margin: 0, flex: 1 }}>
            Clinical Notes
          </h1>
          <div style={{ display: 'flex', gap: 6 }}>
            {['all', 'pending', 'used'].map(f => (
              <button
                key={f}
                onClick={() => setFilter(f)}
                style={{
                  padding: '6px 14px', fontSize: 10, letterSpacing: 1,
                  background: filter === f ? '#1a3a6e' : 'transparent',
                  border: `1px solid ${filter === f ? '#2a5aaa' : '#1e2d40'}`,
                  color: filter === f ? '#4a9eff' : '#3d5166',
                  borderRadius: 3, cursor: 'pointer',
                  fontFamily: "'IBM Plex Mono', monospace",
                }}
              >
                {f.toUpperCase()}
              </button>
            ))}
          </div>
        </div>
        <div style={{ marginTop: 8, fontSize: 11, color: '#3d5166' }}>
          {notes.filter(n => !n.used_in_training).length} pending · {notes.filter(n => n.used_in_training).length} used in training
        </div>
      </div>

      {loading && <div style={{ color: '#3d5166', fontSize: 12, letterSpacing: 1 }}>loading notes...</div>}

      {!loading && filtered.length === 0 && (
        <div style={{
          padding: '48px', background: '#0d1420', border: '1px solid #1e2d40',
          borderRadius: 6, textAlign: 'center', color: '#2a3d52', fontSize: 12, letterSpacing: 1,
        }}>
          no notes found
        </div>
      )}

      {filtered.map(note => (
        <NoteCard
          key={note.id}
          note={note}
          categories={categories}
          onDelete={handleDelete}
          onUpdate={handleUpdate}
        />
      ))}
    </div>
  )
}