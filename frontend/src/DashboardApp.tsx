import { useCallback, useEffect, useMemo, useState } from 'react'
import { chat, downloadReport, explainByCin, fetchEarlyWarningSystem, fetchRecommendationSystem, fetchRulesSystem, fetchShapSystem, generateReport, type ChatResponse, type ModelKind, type RagSource } from './api'
import ChatWorkspace from './ChatWorkspace'
import GraphNetworkViz from './GraphNetworkViz'
import { fetchChatMessages, fetchChatSessions, logoutApi, type AuthUser, type ChatSessionSummary } from './authApi'
import HistoryPanel from './HistoryPanel'
import SystemWorkspace, { type SystemKind } from './SystemWorkspace'
import AppShell from './AppShell'
import './style.css'

type Status = 'idle' | 'loading' | 'error' | 'success'
type View = 'explain' | 'shap' | 'rules' | 'ews' | 'recommendation' | 'chat' | 'history'

type Props = {
  user: AuthUser
  onLogout: () => void
  onBackToAdmin?: () => void
}

function riskBadge(level: string) {
  if (level === 'FAIBLE') return 'badge badge-green'
  if (level === 'MODERE') return 'badge badge-amber'
  return 'badge badge-red'
}

function RagSourcesPanel({ sources }: { sources: RagSource[] }) {
  if (!sources?.length) {
    return <div className="muted">Aucune source documentaire récupérée.</div>
  }
  return (
    <div className="sourceList">
      {sources.map((s, i) => (
        <div key={`${s.source}-${s.chunk_id}-${i}`} className="sourceItem">
          <div className="sourceHead">
            <code>
              {s.source}#{s.chunk_id}
            </code>
            <span className="badge">{(s.score * 100).toFixed(0)}%</span>
          </div>
          <pre className="pre sourceExcerpt">{s.text.length > 320 ? `${s.text.slice(0, 320)}…` : s.text}</pre>
        </div>
      ))}
    </div>
  )
}

function agreementLabel(agreement: string) {
  if (agreement === 'unanimous') return 'Unanime'
  if (agreement === 'majority') return 'Majorité'
  return 'Divergence'
}

function EnsembleBreakdown({ models, voteDefault, voteNonDefault, agreement }: {
  models: Array<{
    model_key: string
    model_name: string
    weight: number
    available: boolean
    default_proba?: number | null
    risk_level?: string | null
    error?: string | null
  }>
  voteDefault: number
  voteNonDefault: number
  agreement: string
}) {
  const maxProba = Math.max(
    ...models.filter((m) => m.available && m.default_proba != null).map((m) => Number(m.default_proba)),
    0.01,
  )
  return (
    <div className="ensemblePanel">
      <div className="sectionTitle">Décomposition ensemble</div>
      <div className="ensembleVoteRow">
        <span className="badge badge-red">Défaut : {voteDefault}</span>
        <span className="badge badge-green">Non-défaut : {voteNonDefault}</span>
        <span className="badge badge-blue">{agreementLabel(agreement)}</span>
      </div>
      <div className="ensembleModelList">
        {models.map((m) => (
          <div key={m.model_key} className={`ensembleModelRow ${m.available ? '' : 'ensembleModelRowOff'}`}>
            <div className="ensembleModelHead">
              <strong>{m.model_name}</strong>
              <span className="muted">poids {(m.weight * 100).toFixed(0)} %</span>
            </div>
            {m.available && m.default_proba != null ? (
              <>
                <div className="ensembleBarTrack">
                  <div
                    className="ensembleBarFill"
                    style={{ width: `${Math.min(100, (Number(m.default_proba) / maxProba) * 100)}%` }}
                  />
                </div>
                <div className="ensembleModelMeta">
                  <span>{(Number(m.default_proba) * 100).toFixed(1)} %</span>
                  <span className={riskBadge(String(m.risk_level ?? ''))}>{m.risk_level}</span>
                </div>
              </>
            ) : (
              <div className="muted ensembleUnavailable">{m.error ?? 'Modèle indisponible'}</div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

function ReportPanel({
  cin,
  reportData,
  reportLoading,
  reportDownloading,
  onGenerate,
  onDownload,
}: {
  cin: string
  reportData: { markdown?: string; sources?: RagSource[] } | null
  reportLoading: boolean
  reportDownloading: 'md' | 'pdf' | null
  onGenerate: () => void
  onDownload: (format: 'md' | 'pdf') => void
}) {
  const canReport = cin.trim().length >= 6
  return (
    <div style={{ display: 'grid', gap: 10 }}>
      <div className="actions">
        <button className="button" onClick={onGenerate} disabled={!canReport || reportLoading}>
          {reportLoading ? 'Génération rapport...' : 'Aperçu rapport RAG'}
        </button>
        <button className="button" type="button" onClick={() => onDownload('md')} disabled={!canReport || reportDownloading !== null}>
          {reportDownloading === 'md' ? 'Téléchargement...' : 'Télécharger .md'}
        </button>
        <button className="button" type="button" onClick={() => onDownload('pdf')} disabled={!canReport || reportDownloading !== null}>
          {reportDownloading === 'pdf' ? 'Téléchargement...' : 'Télécharger .pdf'}
        </button>
      </div>
      {reportData?.markdown && (
        <>
          <div className="sectionTitle">Rapport (Markdown)</div>
          <pre className="pre">{reportData.markdown}</pre>
        </>
      )}
      {reportData?.sources && reportData.sources.length > 0 && (
        <>
          <div className="sectionTitle">Sources RAG</div>
          <RagSourcesPanel sources={reportData.sources} />
        </>
      )}
    </div>
  )
}

export default function DashboardApp({ user, onLogout, onBackToAdmin }: Props) {
  const [view, setView] = useState<View>('explain')
  const [cin, setCin] = useState('')
  const [model, setModel] = useState<ModelKind>('ensemble')
  const [status, setStatus] = useState<Status>('idle')
  const [reportLoading, setReportLoading] = useState(false)
  const [reportDownloading, setReportDownloading] = useState<'md' | 'pdf' | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [data, setData] = useState<any>(null)
  const [chatMeta, setChatMeta] = useState<ChatResponse | null>(null)
  const [reportData, setReportData] = useState<any>(null)
  const [systemResult, setSystemResult] = useState<any>(null)

  const [sessionId, setSessionId] = useState(`${user.username}-${Date.now()}`)
  const [chatSessions, setChatSessions] = useState<ChatSessionSummary[]>([])
  const [chatMsg, setChatMsg] = useState('')
  const [chatHistory, setChatHistory] = useState<Array<{ role: 'user' | 'assistant'; content: string }>>([])

  function newChatSession() {
    setSessionId(`${user.username}-${Date.now()}`)
    setChatHistory([])
    setChatMeta(null)
    setChatMsg('')
  }

  const loadChatSessions = useCallback(async () => {
    try {
      const res = await fetchChatSessions(20)
      setChatSessions(res.items)
    } catch {
      setChatSessions([])
    }
  }, [])

  useEffect(() => {
    if (view === 'chat') loadChatSessions()
  }, [view, chatHistory.length, loadChatSessions])

  const canSubmit = useMemo(() => cin.trim().length >= 6 && status !== 'loading', [cin, status])
  const canChat = useMemo(() => chatMsg.trim().length > 0 && sessionId.trim().length > 2 && status !== 'loading', [chatMsg, sessionId, status])
  const activeCin = useMemo(() => {
    const fromChat = chatMeta?.cin?.trim()
    const fromInput = cin.trim()
    return fromInput.length >= 6 ? fromInput : fromChat && fromChat.length >= 6 ? fromChat : fromInput
  }, [cin, chatMeta])

  useEffect(() => {
    if (view !== 'chat') return
    const sid = sessionId.trim()
    if (sid.length < 3) return
    let cancelled = false
    ;(async () => {
      try {
        const res = await fetchChatMessages(sid)
        if (!cancelled) {
          setChatHistory(
            (res.messages ?? []).map((m) => ({
              role: m.role === 'assistant' ? 'assistant' : 'user',
              content: m.content,
            }))
          )
        }
      } catch {
        if (!cancelled) setChatHistory([])
      }
    })()
    return () => {
      cancelled = true
    }
  }, [view, sessionId, user.id])

  async function handleLogout() {
    await logoutApi()
    onLogout()
  }

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault()
    setStatus('loading')
    setError(null)
    setData(null)
    setSystemResult(null)
    try {
      if (view === 'explain') {
        const out = await explainByCin(model, cin.trim())
        setData(out)
      } else if (view === 'shap') {
        setSystemResult(await fetchShapSystem(cin.trim()))
      } else if (view === 'rules') {
        setSystemResult(await fetchRulesSystem(cin.trim()))
      } else if (view === 'ews') {
        setSystemResult(await fetchEarlyWarningSystem(cin.trim()))
      } else if (view === 'recommendation') {
        setSystemResult(await fetchRecommendationSystem(cin.trim()))
      }
      setStatus('success')
    } catch (err: any) {
      setError(err?.message ?? String(err))
      setStatus('error')
    }
  }

  async function onChat(e: React.FormEvent) {
    e.preventDefault()
    setStatus('loading')
    setError(null)
    const msg = chatMsg.trim()
    setChatMsg('')
    setChatHistory((h) => [...h, { role: 'user', content: msg }])
    try {
      const res = await chat(sessionId.trim(), msg)
      setChatHistory((h) => [...h, { role: 'assistant', content: res.answer }])
      setChatMeta(res)
      if (res.cin && String(res.cin).trim().length >= 6) {
        setCin(String(res.cin).trim())
      }
      if (res.report_available && res.cin) {
        try {
          setReportData(await generateReport(String(res.cin)))
        } catch {
          /* optional */
        }
      }
      setStatus('success')
    } catch (err: any) {
      setError(err?.message ?? String(err))
      setStatus('error')
    }
  }

  async function onGenerateReport(targetCin?: string) {
    const raw = (targetCin ?? activeCin).trim()
    if (raw.length < 6) {
      setError('CIN invalide pour le rapport.')
      setStatus('error')
      return
    }
    try {
      setReportLoading(true)
      setError(null)
      const res = await generateReport(raw)
      setReportData(res)
      setStatus('success')
    } catch (err: any) {
      setError(err?.message ?? String(err))
      setStatus('error')
    } finally {
      setReportLoading(false)
    }
  }

  async function onDownloadReport(format: 'md' | 'pdf', targetCin?: string) {
    const raw = (targetCin ?? activeCin).trim()
    if (raw.length < 6) {
      setError('CIN invalide.')
      setStatus('error')
      return
    }
    try {
      setReportDownloading(format)
      setError(null)
      await downloadReport(raw, format)
      setStatus('success')
    } catch (err: any) {
      setError(err?.message ?? String(err))
      setStatus('error')
    } finally {
      setReportDownloading(null)
    }
  }

  const agentNav = [
    { id: 'explain', label: 'Modèles ML', icon: '🤖' },
    { id: 'shap', label: 'SHAP', icon: '🔍' },
    { id: 'rules', label: 'Règles métier', icon: '📜' },
    { id: 'ews', label: 'Alertes', icon: '⚠️' },
    { id: 'recommendation', label: 'Recommandation', icon: '⭐' },
    { id: 'chat', label: 'Assistant IA', icon: '💬' },
    { id: 'history', label: 'Historique', icon: '📋' },
  ]

  const isSystemView = (v: View): v is SystemKind =>
    v === 'shap' || v === 'rules' || v === 'ews' || v === 'recommendation'

  return (
    <AppShell
      user={user}
      navItems={agentNav}
      activeNav={view}
      onNavChange={(id) => setView(id as View)}
      onLogout={handleLogout}
      subtitle="Espace agent — scoring ML et systèmes décisionnels autonomes"
    >
      {onBackToAdmin && (
        <div className="portalToolbar">
          <button type="button" className="button buttonGhost buttonSmall" onClick={onBackToAdmin}>
            ← Retour administration
          </button>
        </div>
      )}
      {isSystemView(view) ? (
        <SystemWorkspace
          kind={view}
          cin={cin}
          onCinChange={setCin}
          onSubmit={onSubmit}
          loading={status === 'loading'}
          canSubmit={canSubmit}
          error={status === 'error' ? error : null}
          result={status === 'success' ? systemResult : null}
        />
      ) : (
      <div className="grid">
        <section className="card">
          {view === 'history' ? (
            <>
              <div className="cardTitle">Historique</div>
              <p className="muted">Vos analyses et conversations enregistrées en base SQLite.</p>
            </>
          ) : view === 'explain' ? (
            <>
              <div className="cardTitle">Modèles ML — scoring par CIN</div>
              <p className="muted">Classic, Séquentiel, Graphe ou Ensemble unifié (fusion pondérée AUC + vote).</p>
              <form className="form" onSubmit={onSubmit}>
                <label className="label">
                  CIN client
                  <input className="input" value={cin} onChange={(e) => setCin(e.target.value)} placeholder="88710263" />
                </label>
                <label className="label">
                  Modèle ML
                    <select className="input" value={model} onChange={(e) => setModel(e.target.value as ModelKind)}>
                    <option value="ensemble">Ensemble — score unifié (recommandé)</option>
                    <option value="classic">Deep Tabular (MLP + Embeddings)</option>
                    <option value="sequential">Temporal Transformer</option>
                    <option value="graph">GAT — Graph Attention</option>
                  </select>
                </label>
                <button className="button" disabled={!canSubmit} type="submit">
                  {status === 'loading' ? 'Analyse...' : 'Scorer + Expliquer'}
                </button>
              </form>
            </>
          ) : (
            <>
              <div className="cardTitle">Assistant conversationnel</div>
              <div className="actions" style={{ marginBottom: 10 }}>
                <button type="button" className="button buttonSmall" onClick={newChatSession}>
                  + Nouvelle conversation
                </button>
              </div>
              {chatSessions.length > 0 && (
                <div className="sessionListCompact">
                  {chatSessions.slice(0, 5).map((s) => (
                    <button
                      key={s.session_id}
                      type="button"
                      className={`sessionChip ${s.session_id === sessionId ? 'sessionChipActive' : ''}`}
                      onClick={() => setSessionId(s.session_id)}
                      title={s.last_preview ?? undefined}
                    >
                      {(s.title || s.session_id).slice(0, 28)}
                      {s.last_cin ? ` · ${s.last_cin}` : ''}
                    </button>
                  ))}
                </div>
              )}
              <form className="form" onSubmit={onChat}>
                <label className="label">
                  Session
                  <input className="input" value={sessionId} onChange={(e) => setSessionId(e.target.value)} />
                </label>
                <label className="label">
                  Message
                  <input className="input" value={chatMsg} onChange={(e) => setChatMsg(e.target.value)} placeholder="Analyse le CIN 88710263" />
                </label>
                <button className="button" disabled={!canChat} type="submit">
                  {status === 'loading' ? '...' : 'Envoyer'}
                </button>
              </form>
              <div className="actions" style={{ marginTop: 10, flexWrap: 'wrap' }}>
                {(chatMeta?.suggested_prompts ?? [
                  'Analyse institutionnelle du CIN 88710263',
                  'Rapport complet pour le CIN 88710263',
                  'Compare les modèles ML pour 88710263',
                  'Score séquentiel du CIN 88710263',
                ]).map((p) => (
                  <button key={p} className="button buttonSmall" type="button" onClick={() => setChatMsg(p)}>
                    {p.length > 36 ? `${p.slice(0, 36)}…` : p}
                  </button>
                ))}
              </div>
            </>
          )}

          {status === 'error' && (
            <div className="error" style={{ marginTop: 12 }}>
              <div className="errorTitle">Erreur</div>
              <pre className="pre">{error}</pre>
            </div>
          )}
        </section>

        <section className="card">
          {view === 'history' ? (
            <>
              <div className="cardTitle">Historique & activité</div>
              <HistoryPanel
                user={user}
                onOpenChatSession={(sid) => {
                  setSessionId(sid)
                  setView('chat')
                }}
              />
            </>
          ) : view === 'chat' ? (
            <ChatWorkspace
              chatHistory={chatHistory}
              chatMeta={chatMeta}
              reportData={reportData}
              reportLoading={reportLoading}
              reportDownloading={reportDownloading}
              activeCin={activeCin}
              onGenerateReport={() => onGenerateReport(activeCin)}
              onDownloadReport={(fmt) => onDownloadReport(fmt, activeCin)}
            />
          ) : view === 'explain' && status !== 'success' ? (
            <div className="muted">Lancez un scoring ML pour voir le résultat.</div>
          ) : view === 'explain' ? (
            <>
              <div className="cardTitle">{model === 'ensemble' ? 'Résultat — Score ensemble' : 'Résultat — Modèle ML'}</div>
              <div className="kpis">
                <div className="kpi">
                  <div className="kpiLabel">KYC</div>
                  <div className="kpiValue">{data?.kyc_score ?? '—'}</div>
                </div>
                <div className="kpi">
                  <div className="kpiLabel">{model === 'ensemble' ? 'Proba ensemble' : 'Proba défaut'}</div>
                  <div className="kpiValue">{data?.default_proba ?? '—'}</div>
                </div>
                <div className="kpi">
                  <div className="kpiLabel">Risque</div>
                  <div className={`kpiValue ${riskBadge(String(data?.risk_level ?? ''))}`}>{data?.risk_level ?? '—'}</div>
                </div>
                <div className="kpi">
                  <div className="kpiLabel">Modèle</div>
                  <div className="kpiValue kpiValueSmall">{data?.model_used ?? '—'}</div>
                </div>
              </div>
              {model === 'ensemble' && Array.isArray(data?.models) && (
                <EnsembleBreakdown
                  models={data.models}
                  voteDefault={data.vote_default ?? 0}
                  voteNonDefault={data.vote_non_default ?? 0}
                  agreement={data.agreement ?? 'split'}
                />
              )}
              {Array.isArray(data?.credits) && (
                <div className="tableWrap">
                  <div className="sectionTitle">Crédits ({data?.n_credits ?? data.credits.length})</div>
                  <table className="table">
                    <thead>
                      <tr>
                        <th>credit_id</th>
                        <th>default_proba</th>
                        <th>risk_level</th>
                        <th>prediction</th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.credits.map((c: any) => (
                        <tr key={c.credit_id}>
                          <td>{c.credit_id}</td>
                          <td>{c.default_proba}</td>
                          <td>
                            <span className={riskBadge(String(c.risk_level))}>{c.risk_level}</span>
                          </td>
                          <td>{c.prediction}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
              {(model === 'graph' || model === 'ensemble') && data?.network && (
                <>
                  <div className="sectionTitle">Réseau GAT — client & voisins</div>
                  <GraphNetworkViz network={data.network} />
                </>
              )}
              <div className="sectionTitle">Explication (LLM + RAG)</div>
              <pre className="pre">{data?.message ?? ''}</pre>
              <div className="sectionTitle">Rapport</div>
              <ReportPanel
                cin={activeCin}
                reportData={reportData}
                reportLoading={reportLoading}
                reportDownloading={reportDownloading}
                onGenerate={() => onGenerateReport(activeCin)}
                onDownload={(fmt) => onDownloadReport(fmt, activeCin)}
              />
            </>
          ) : (
            <div className="muted">Sélectionnez une vue.</div>
          )}
        </section>
      </div>
      )}
    </AppShell>
  )
}
