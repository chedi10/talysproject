import { useMemo, useState } from 'react'
import type { ModelKind } from './api'
import { explainByCin } from './api'
import './style.css'

type Status = 'idle' | 'loading' | 'error' | 'success'

function riskBadge(level: string) {
  if (level === 'FAIBLE') return 'badge badge-green'
  if (level === 'MODERE') return 'badge badge-amber'
  return 'badge badge-red'
}

export default function App() {
  const [cin, setCin] = useState('')
  const [model, setModel] = useState<ModelKind>('sequential')
  const [status, setStatus] = useState<Status>('idle')
  const [error, setError] = useState<string | null>(null)
  const [data, setData] = useState<any>(null)

  const canSubmit = useMemo(() => cin.trim().length >= 6 && status !== 'loading', [cin, status])

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault()
    setStatus('loading')
    setError(null)
    setData(null)
    try {
      const out = await explainByCin(model, cin.trim())
      setData(out)
      setStatus('success')
    } catch (err: any) {
      setError(err?.message ?? String(err))
      setStatus('error')
    }
  }

  return (
    <div className="page">
      <header className="header">
        <div className="title">
          <div className="h1">Talys — Scoring risque crédit</div>
          <div className="subtitle">Interface React (CIN → explication complète)</div>
        </div>
        <a className="link" href="http://localhost:8000/docs" target="_blank" rel="noreferrer">
          Docs API
        </a>
      </header>

      <main className="grid">
        <section className="card">
          <div className="cardTitle">Demande</div>
          <form className="form" onSubmit={onSubmit}>
            <label className="label">
              CIN
              <input
                className="input"
                value={cin}
                onChange={(e) => setCin(e.target.value)}
                placeholder="Tape le CIN (ex: 88710263)"
              />
            </label>

            <label className="label">
              Modèle
              <select className="input" value={model} onChange={(e) => setModel(e.target.value as ModelKind)}>
                <option value="classic">Classique (tabulaire)</option>
                <option value="sequential">Séquentiel (transactions + remboursements)</option>
                <option value="graph">Graphe (GraphSAGE)</option>
              </select>
            </label>

            <button className="button" disabled={!canSubmit} type="submit">
              {status === 'loading' ? 'Analyse...' : 'Expliquer'}
            </button>
          </form>

          {status === 'error' && (
            <div className="error">
              <div className="errorTitle">Erreur</div>
              <pre className="pre">{error}</pre>
            </div>
          )}
        </section>

        <section className="card">
          <div className="cardTitle">Résultat</div>
          {status !== 'success' ? (
            <div className="muted">Lance une explication pour voir le résultat ici.</div>
          ) : (
            <>
              <div className="kpis">
                <div className="kpi">
                  <div className="kpiLabel">KYC</div>
                  <div className="kpiValue">{data?.kyc_score ?? '—'}</div>
                </div>
                <div className="kpi">
                  <div className="kpiLabel">Proba défaut</div>
                  <div className="kpiValue">{data?.default_proba ?? '—'}</div>
                </div>
                <div className="kpi">
                  <div className="kpiLabel">Risque</div>
                  <div className={`kpiValue ${riskBadge(String(data?.risk_level ?? ''))}`}>{data?.risk_level ?? '—'}</div>
                </div>
                <div className="kpi">
                  <div className="kpiLabel">Modèle</div>
                  <div className="kpiValue">{data?.model_used ?? '—'}</div>
                </div>
              </div>

              {Array.isArray(data?.credits) && (
                <div className="tableWrap">
                  <div className="sectionTitle">Crédits analysés ({data?.n_credits ?? data?.credits?.length ?? 0})</div>
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

              <div className="sectionTitle">Explication (LLM)</div>
              <pre className="pre">{data?.message ?? ''}</pre>
            </>
          )}
        </section>
      </main>

      <footer className="footer muted">Astuce: un client multi-crédits est CIN <code>88710263</code>.</footer>
    </div>
  )
}

