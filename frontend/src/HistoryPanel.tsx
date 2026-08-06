import { useCallback, useEffect, useState } from 'react'
import { fetchChatSessions, fetchHistory, fetchUsers, type ChatSessionSummary } from './authApi'
import type { ActivityRecord, AuthUser } from './authStorage'

type Props = {
  user: AuthUser
  onOpenChatSession?: (sessionId: string) => void
}

function formatDate(iso: string) {
  try {
    return new Date(iso).toLocaleString('fr-FR')
  } catch {
    return iso
  }
}

function actionLabel(action: string) {
  const map: Record<string, string> = {
    explain_classic: 'Explain classique',
    explain_sequential: 'Explain séquentiel',
    explain_graph: 'Explain graphe',
    explain_ensemble: 'Score ensemble',
    system_shap: 'Système SHAP',
    system_rules: 'Règles métier',
    system_ews: 'Early Warning',
    system_recommendation: 'Recommandation',
    chat: 'Chat agent',
    client_chat: 'Assistant client',
    report: 'Rapport RAG',
    report_download: 'Téléchargement PDF/MD',
  }
  return map[action] ?? action
}

export default function HistoryPanel({ user, onOpenChatSession }: Props) {
  const [items, setItems] = useState<ActivityRecord[]>([])
  const [chatSessions, setChatSessions] = useState<ChatSessionSummary[]>([])
  const [scope, setScope] = useState<'mine' | 'all'>('mine')
  const [users, setUsers] = useState<AuthUser[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const reload = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const hist = await fetchHistory(200)
      setItems(hist.items)
      setScope(hist.scope)
      const chats = await fetchChatSessions(30)
      setChatSessions(chats.items)
      if (user.role === 'admin') {
        const u = await fetchUsers()
        setUsers(u)
      }
    } catch (err: any) {
      setError(err?.message ?? String(err))
    } finally {
      setLoading(false)
    }
  }, [user.role])

  useEffect(() => {
    reload()
  }, [reload])

  return (
    <div style={{ display: 'grid', gap: 14 }}>
      <div className="orchestrationMeta">
        <span className="badge">{user.role === 'admin' ? 'Admin — vue globale' : 'Agent — mon historique'}</span>
        <span className="badge">{`scope: ${scope}`}</span>
        <span className="badge">{`${items.length} activité(s)`}</span>
        <button type="button" className="button buttonSmall" onClick={reload} disabled={loading}>
          {loading ? '...' : 'Actualiser'}
        </button>
      </div>

      {error && (
        <div className="error">
          <div className="errorTitle">Erreur</div>
          <pre className="pre">{error}</pre>
        </div>
      )}

      {user.role === 'admin' && users.length > 0 && (
        <>
          <div className="sectionTitle">Utilisateurs ({users.length})</div>
          <div className="tableWrap">
            <table className="table">
              <thead>
                <tr>
                  <th>username</th>
                  <th>email</th>
                  <th>role</th>
                  <th>créé le</th>
                </tr>
              </thead>
              <tbody>
                {users.map((u) => (
                  <tr key={u.id}>
                    <td>{u.username}</td>
                    <td>{u.email}</td>
                    <td>
                      <span className={`badge ${u.role === 'admin' ? 'badge-amber' : ''}`}>{u.role}</span>
                    </td>
                    <td>{u.created_at ? formatDate(u.created_at) : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {chatSessions.length > 0 && (
        <>
          <div className="sectionTitle">Conversations enregistrées</div>
          <div className="sessionList">
            {chatSessions.map((s) => (
              <div key={`${s.username}-${s.session_id}`} className="sessionItem">
                <div className="sessionItemHead">
                  <strong>{s.title || s.session_id}</strong>
                  {s.last_intent && <span className="badge">{s.last_intent}</span>}
                </div>
                <div className="muted sessionMeta">
                  {user.role === 'admin' && <span>{s.username} · </span>}
                  {s.message_count} msg
                  {s.last_cin && <> · CIN {s.last_cin}</>}
                  {s.updated_at && <> · {formatDate(s.updated_at)}</>}
                </div>
                {s.last_preview && <div className="historyPreview">{s.last_preview}</div>}
                {onOpenChatSession && (
                  <button type="button" className="button buttonSmall" onClick={() => onOpenChatSession(s.session_id)}>
                    Reprendre
                  </button>
                )}
              </div>
            ))}
          </div>
        </>
      )}

      <div className="sectionTitle">
        {user.role === 'admin' ? 'Activité — tous les agents' : 'Mes analyses & rapports'}
      </div>

      {!loading && items.length === 0 && <div className="muted">Aucune activité enregistrée.</div>}

      {items.length > 0 && (
        <div className="tableWrap">
          <table className="table">
            <thead>
              <tr>
                <th>Date</th>
                {user.role === 'admin' && <th>Utilisateur</th>}
                <th>Action</th>
                <th>CIN</th>
                <th>Modèle / Intent</th>
                <th>Aperçu</th>
              </tr>
            </thead>
            <tbody>
              {items.map((r) => (
                <tr key={r.id}>
                  <td>{formatDate(r.created_at)}</td>
                  {user.role === 'admin' && <td>{r.username}</td>}
                  <td>{actionLabel(r.action)}</td>
                  <td>{r.cin ?? '—'}</td>
                  <td>{r.model ?? r.intent ?? '—'}</td>
                  <td className="historyPreview">{r.message_preview ?? '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
