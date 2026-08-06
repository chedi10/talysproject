import { useEffect, useMemo, useState } from 'react'
import AppShell from './AppShell'
import HistoryPanel from './HistoryPanel'
import { createUser, fetchAdminStats, fetchUsers, logoutApi, type AuthUser, type SystemStats } from './authApi'

type Props = {
  user: AuthUser
  onLogout: () => void
  onAgentMode?: () => void
}

type View = 'dashboard' | 'users' | 'activity' | 'system'

function BarChart({ data, labelKey }: { data: Record<string, number>; labelKey?: (k: string) => string }) {
  const entries = Object.entries(data).sort((a, b) => b[1] - a[1])
  const max = Math.max(...entries.map(([, v]) => v), 1)
  return (
    <div className="barChart">
      {entries.map(([k, v]) => (
        <div key={k} className="barChartRow">
          <span className="barChartLabel">{labelKey ? labelKey(k) : k}</span>
          <div className="barChartTrack">
            <div className="barChartFill" style={{ width: `${(v / max) * 100}%` }} />
          </div>
          <span className="barChartValue">{v}</span>
        </div>
      ))}
    </div>
  )
}

const ACTION_LABELS: Record<string, string> = {
  explain_classic: 'ML Tabular',
  explain_sequential: 'ML Séquentiel',
  explain_graph: 'ML Graphe',
  system_shap: 'SHAP',
  system_rules: 'Règles',
  system_ews: 'Early Warning',
  system_recommendation: 'Recommandation',
  chat: 'Chat',
  report: 'Rapport',
  report_download: 'PDF/MD',
}

export default function AdminDashboard({ user, onLogout, onAgentMode }: Props) {
  const [view, setView] = useState<View>('dashboard')
  const [stats, setStats] = useState<SystemStats | null>(null)
  const [users, setUsers] = useState<AuthUser[]>([])
  const [error, setError] = useState<string | null>(null)
  const [creating, setCreating] = useState(false)
  const [userFilter, setUserFilter] = useState('')
  const [form, setForm] = useState({ username: '', email: '', password: '', role: 'agent' as 'client' | 'agent' | 'admin', cin: '' })

  useEffect(() => {
    if (view === 'dashboard' || view === 'system') loadStats()
    if (view === 'users' || view === 'dashboard') loadUsers()
  }, [view])

  async function loadStats() {
    try {
      setStats(await fetchAdminStats())
    } catch (err: any) {
      setError(err?.message ?? String(err))
    }
  }

  async function loadUsers() {
    try {
      setUsers(await fetchUsers())
    } catch (err: any) {
      setError(err?.message ?? String(err))
    }
  }

  async function handleLogout() {
    await logoutApi()
    onLogout()
  }

  async function onCreateUser(e: React.FormEvent) {
    e.preventDefault()
    setCreating(true)
    setError(null)
    try {
      await createUser(form)
      setForm({ username: '', email: '', password: '', role: 'agent', cin: '' })
      await loadUsers()
      await loadStats()
    } catch (err: any) {
      setError(err?.message ?? String(err))
    } finally {
      setCreating(false)
    }
  }

  const filteredUsers = useMemo(() => {
    const q = userFilter.trim().toLowerCase()
    if (!q) return users
    return users.filter(
      (u) =>
        u.username.toLowerCase().includes(q) ||
        u.email.toLowerCase().includes(q) ||
        (u.cin ?? '').includes(q) ||
        u.role.includes(q),
    )
  }, [users, userFilter])

  const navItems = [
    { id: 'dashboard', label: 'Tableau de bord', icon: '📊' },
    { id: 'users', label: 'Utilisateurs', icon: '👥' },
    { id: 'activity', label: 'Activité', icon: '📋' },
    { id: 'system', label: 'Système', icon: '⚙️' },
  ]

  const apiBase = (import.meta as any).env?.VITE_API_BASE_URL ?? 'http://localhost:8000'

  return (
    <AppShell
      user={user}
      navItems={navItems}
      activeNav={view}
      onNavChange={(id) => setView(id as View)}
      onLogout={handleLogout}
      subtitle="Administration — pilotage de la plateforme Talys"
    >
      <div className="portalToolbar">
        {onAgentMode && (
          <button type="button" className="button" onClick={onAgentMode}>
            Mode agent — scoring & systèmes
          </button>
        )}
        <button type="button" className="button buttonGhost buttonSmall" onClick={() => { loadStats(); loadUsers() }}>
          Actualiser
        </button>
      </div>

      {error && (
        <div className="card error" style={{ marginBottom: 12 }}>
          <pre className="pre">{error}</pre>
        </div>
      )}

      {view === 'dashboard' && (
        <div className="portalGrid">
          <section className="card portalHero adminHero">
            <div className="portalHeroContent">
              <div className="cardTitle">Vue d'ensemble — Talys Scoring</div>
              <p className="muted">
                Plateforme microfinance : {stats?.clients?.toLocaleString() ?? '—'} clients,{' '}
                {stats?.credits?.toLocaleString() ?? '—'} crédits, graphe relationnel enrichi (GAT).
              </p>
            </div>
            {stats && (
              <div className="portalHeroBadge">
                <span className="portalHeroBadgeLabel">Taux de défaut portefeuille</span>
                <strong>{(stats.default_rate * 100).toFixed(1)}%</strong>
                <span className="muted">{stats.credits_en_defaut} crédits</span>
              </div>
            )}
          </section>

          {stats ? (
            <>
              <section className="card">
                <div className="cardTitle">Indicateurs clés</div>
                <div className="kpis">
                  <div className="kpi">
                    <div className="kpiLabel">Clients</div>
                    <div className="kpiValue">{stats.clients.toLocaleString()}</div>
                  </div>
                  <div className="kpi">
                    <div className="kpiLabel">Crédits</div>
                    <div className="kpiValue">{stats.credits.toLocaleString()}</div>
                  </div>
                  <div className="kpi">
                    <div className="kpiLabel">Relations graphe</div>
                    <div className="kpiValue">{stats.relations.toLocaleString()}</div>
                  </div>
                  <div className="kpi">
                    <div className="kpiLabel">Utilisateurs</div>
                    <div className="kpiValue">{stats.users}</div>
                  </div>
                  <div className="kpi">
                    <div className="kpiLabel">Activité 7 j</div>
                    <div className="kpiValue">{stats.activity_last_7_days}</div>
                  </div>
                  <div className="kpi">
                    <div className="kpiLabel">Sessions chat</div>
                    <div className="kpiValue">{stats.chat_sessions}</div>
                  </div>
                </div>
              </section>

              <div className="grid">
                <section className="card">
                  <div className="cardTitle">Utilisateurs par rôle</div>
                  <BarChart
                    data={stats.users_by_role}
                    labelKey={(k) => (k === 'admin' ? 'Administrateur' : k === 'agent' ? 'Agent' : 'Client')}
                  />
                </section>
                <section className="card">
                  <div className="cardTitle">Statut KYC (portefeuille)</div>
                  <BarChart data={stats.kyc_breakdown} />
                </section>
              </div>

              <section className="card">
                <div className="cardTitle">Activité par type (top 8)</div>
                <BarChart data={stats.activity_by_action} labelKey={(k) => ACTION_LABELS[k] ?? k} />
              </section>
            </>
          ) : (
            <section className="card muted">Chargement des statistiques...</section>
          )}
        </div>
      )}

      {view === 'users' && (
        <div className="grid">
          <section className="card">
            <div className="cardTitle">Utilisateurs ({filteredUsers.length})</div>
            <input
              className="input"
              placeholder="Rechercher username, email, CIN..."
              value={userFilter}
              onChange={(e) => setUserFilter(e.target.value)}
              style={{ marginBottom: 12 }}
            />
            <div className="tableWrap">
              <table className="table">
                <thead>
                  <tr>
                    <th>Username</th>
                    <th>Email</th>
                    <th>Rôle</th>
                    <th>CIN</th>
                    <th>Créé le</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredUsers.map((u) => (
                    <tr key={u.id}>
                      <td><strong>{u.username}</strong></td>
                      <td>{u.email}</td>
                      <td>
                        <span className={`badge ${u.role === 'admin' ? 'badge-amber' : u.role === 'client' ? 'badge-blue' : 'badge-green'}`}>
                          {u.role}
                        </span>
                      </td>
                      <td>{u.cin ?? '—'}</td>
                      <td>{u.created_at ? new Date(u.created_at).toLocaleDateString('fr-FR') : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
          <section className="card">
            <div className="cardTitle">Créer un utilisateur</div>
            <form className="form" onSubmit={onCreateUser}>
              <label className="label">
                Username
                <input className="input" value={form.username} onChange={(e) => setForm({ ...form, username: e.target.value })} required />
              </label>
              <label className="label">
                Email
                <input className="input" type="email" value={form.email} onChange={(e) => setForm({ ...form, email: e.target.value })} required />
              </label>
              <label className="label">
                Mot de passe
                <input className="input" type="password" value={form.password} onChange={(e) => setForm({ ...form, password: e.target.value })} required minLength={6} />
              </label>
              <label className="label">
                Rôle
                <select className="input" value={form.role} onChange={(e) => setForm({ ...form, role: e.target.value as any })}>
                  <option value="agent">Agent crédit</option>
                  <option value="client">Client</option>
                  <option value="admin">Administrateur</option>
                </select>
              </label>
              {form.role === 'client' && (
                <label className="label">
                  CIN client
                  <input className="input" value={form.cin} onChange={(e) => setForm({ ...form, cin: e.target.value })} placeholder="88710263" required />
                </label>
              )}
              <button className="button" type="submit" disabled={creating}>
                {creating ? 'Création...' : 'Créer le compte'}
              </button>
            </form>
          </section>
        </div>
      )}

      {view === 'activity' && (
        <section className="card">
          <div className="cardTitle">Activité globale & conversations</div>
          <HistoryPanel user={user} />
        </section>
      )}

      {view === 'system' && (
        <div className="grid">
          <section className="card">
            <div className="cardTitle">État du système</div>
            {stats ? (
              <ul className="systemStatusList">
                <li><span>Base de données</span><strong>{stats.database}</strong></li>
                <li><span>Clients / crédits / tx</span><strong>{stats.clients} / {stats.credits} / {stats.transactions}</strong></li>
                <li><span>Remboursements</span><strong>{stats.remboursements.toLocaleString()}</strong></li>
                <li><span>Arêtes graphe</span><strong>{stats.relations.toLocaleString()}</strong></li>
                <li><span>Modèle tabular</span><strong>{stats.model_loaded ? stats.model_name : 'Non chargé'}</strong></li>
                <li><span>Modèle GAT</span><strong>{stats.graph_model ?? '—'} {stats.graph_auc != null ? `(AUC ${stats.graph_auc})` : ''}</strong></li>
                <li><span>Journal d'activité</span><strong>{stats.activity_log} entrées</strong></li>
                <li><span>API</span><a className="link" href={`${apiBase}/docs`} target="_blank" rel="noreferrer">Swagger OpenAPI</a></li>
              </ul>
            ) : (
              <div className="muted">Chargement...</div>
            )}
          </section>
          <section className="card">
            <div className="cardTitle">Stack technique</div>
            <ul className="actionList">
              <li>Backend FastAPI + SQLite (auth, activité, chat)</li>
              <li>ML : Deep Tabular, Transformer, GAT (graphe enrichi)</li>
              <li>Systèmes : SHAP, Règles, EWS, Recommandation</li>
              <li>Agent LangGraph + RAG + rapports PDF</li>
            </ul>
          </section>
        </div>
      )}
    </AppShell>
  )
}
