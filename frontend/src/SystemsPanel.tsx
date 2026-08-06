import type {
  ShapExplanation,
  BusinessRulesBlock,
  EarlyWarningBlock,
  AiRecommendation,
  ClientProfile,
} from './AnalysisPanel'

export type SystemClientContext = {
  cin: string
  credit_id: number
  kyc_score: number
  institutional_score: number
  risk_level: string
  risk_factors?: string[]
  client_profile?: ClientProfile
  credit_snapshot?: Record<string, unknown>
}

function severityBadge(severity: string) {
  if (severity === 'CRITICAL') return 'badge badge-red'
  if (severity === 'WARNING') return 'badge badge-amber'
  return 'badge'
}

function actionBadge(action: string) {
  if (action === 'block') return 'badge badge-red'
  if (action === 'manual_review') return 'badge badge-amber'
  if (action === 'alert') return 'badge badge-blue'
  return 'badge'
}

function decisionBadge(decision: string) {
  if (decision === 'ACCEPTER') return 'decisionCard decision-accept'
  if (decision === 'REFUSER') return 'decisionCard decision-refuse'
  return 'decisionCard decision-conditional'
}

function watchlistBadge(priority: string) {
  if (priority === 'HIGH') return 'badge badge-red'
  if (priority === 'MEDIUM') return 'badge badge-amber'
  if (priority === 'LOW') return 'badge badge-blue'
  return 'badge'
}

function formatValue(v: unknown): string {
  if (v == null) return '—'
  if (typeof v === 'object') return JSON.stringify(v)
  if (typeof v === 'number') return Number.isInteger(v) ? String(v) : v.toFixed(2)
  return String(v)
}

function ShapBar({ label, impact, maxAbs }: { label: string; impact: number; maxAbs: number }) {
  const pct = maxAbs > 0 ? Math.min(100, (Math.abs(impact) / maxAbs) * 100) : 0
  const positive = impact > 0
  return (
    <div className="shapRow">
      <div className="shapLabel" title={label}>{label}</div>
      <div className="shapTrack">
        <div className={`shapFill ${positive ? 'shap-up' : 'shap-down'}`} style={{ width: `${pct}%` }} />
      </div>
      <div className={`shapValue ${positive ? 'shap-up-text' : 'shap-down-text'}`}>
        {impact > 0 ? '+' : ''}{impact.toFixed(3)}
      </div>
    </div>
  )
}

function MetricTrend({ series }: { series: { label: string; points: Array<{ value: number; is_current: boolean; date: string }> } }) {
  const values = series.points.map((p) => p.value)
  const min = Math.min(...values)
  const max = Math.max(...values)
  const range = max - min || 1
  return (
    <div className="trendBlock">
      <div className="trendLabel">{series.label}</div>
      <div className="trendChart">
        {series.points.map((p) => {
          const h = ((p.value - min) / range) * 100
          return (
            <div key={`${p.date}-${p.value}`} className={`trendBar ${p.is_current ? 'trendBarCurrent' : ''}`} title={`${p.date}: ${p.value}`}>
              <div className="trendBarFill" style={{ height: `${Math.max(8, h)}%` }} />
              <span className="trendBarVal">{typeof p.value === 'number' && p.value < 1 ? `${(p.value * 100).toFixed(0)}%` : p.value.toFixed(1)}</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

function ComplianceGauge({ score }: { score: number }) {
  const color = score >= 80 ? 'gaugeGreen' : score >= 50 ? 'gaugeAmber' : 'gaugeRed'
  return (
    <div className="complianceGauge">
      <div className={`gaugeRing ${color}`}>
        <span className="gaugeValue">{score}</span>
      </div>
      <div className="gaugeLabel">Conformité /100</div>
    </div>
  )
}

export function ShapSystemPanel({ shap }: { shap: ShapExplanation }) {
  const allImpacts = [...shap.increases_risk, ...shap.decreases_risk]
  const maxAbs = Math.max(...allImpacts.map((x) => Math.abs(x.impact)), 0.001)
  const cc = shap.credit_context

  return (
    <section className="analysisBlock">
      <div className="cardTitle">Analyse explicative</div>
      <div className="systemMetaRow">
        <span className="badge">{shap.method}</span>
        {shap.base_prediction != null && (
          <span className="badge badge-amber">Proba base XAI: {(shap.base_prediction * 100).toFixed(1)}%</span>
        )}
      </div>
      {cc && (cc.montant || cc.objet) && (
        <div className="creditSnapshot">
          {cc.montant != null && <span>Montant <strong>{cc.montant.toLocaleString()} TND</strong></span>}
          {cc.duree_mois != null && <span>Durée <strong>{cc.duree_mois} mois</strong></span>}
          {cc.dti != null && <span>DTI <strong>{(cc.dti * 100).toFixed(0)}%</strong></span>}
          {cc.objet && <span>Objet <strong>{cc.objet}</strong></span>}
        </div>
      )}
      <p className="muted">{shap.summary}</p>

      {shap.driver_details && shap.driver_details.length > 0 && (
        <>
          <div className="sectionTitle">Top facteurs — détail vs portefeuille</div>
          <div className="tableWrap">
            <table className="table tableCompact">
              <thead>
                <tr>
                  <th>Variable</th>
                  <th>Valeur client</th>
                  <th>Médiane portefeuille</th>
                  <th>Impact</th>
                  <th>vs portefeuille</th>
                </tr>
              </thead>
              <tbody>
                {shap.driver_details.map((d) => (
                  <tr key={d.feature}>
                    <td>{d.label}</td>
                    <td>{d.value != null ? formatValue(d.value) : '—'}</td>
                    <td>{d.portfolio_median != null ? formatValue(d.portfolio_median) : '—'}</td>
                    <td className={d.impact > 0 ? 'shap-up-text' : 'shap-down-text'}>
                      {d.impact > 0 ? '+' : ''}{d.impact.toFixed(4)}
                    </td>
                    <td>{d.vs_portfolio_pct != null ? `${d.vs_portfolio_pct > 0 ? '+' : ''}${d.vs_portfolio_pct}%` : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      <div className="shapColumns">
        <div>
          <div className="shapColTitle shap-up-text">↑ Augmente le risque</div>
          {shap.increases_risk.map((f) => (
            <ShapBar key={f.feature} label={f.label} impact={f.impact} maxAbs={maxAbs} />
          ))}
        </div>
        <div>
          <div className="shapColTitle shap-down-text">↓ Diminue le risque</div>
          {shap.decreases_risk.map((f) => (
            <ShapBar key={f.feature} label={f.label} impact={f.impact} maxAbs={maxAbs} />
          ))}
        </div>
      </div>
    </section>
  )
}

export function RulesSystemPanel({ rules }: { rules: BusinessRulesBlock }) {
  const triggered = rules.rules.filter((r) => r.triggered)
  const passing = rules.rules.filter((r) => !r.triggered)

  return (
    <section className="analysisBlock">
      <div className="rulesHeader">
        <div className="cardTitle">Moteur de règles métier</div>
        <ComplianceGauge score={rules.compliance_score ?? 100} />
      </div>
      <div className="systemMetaRow">
        <span className="badge">{rules.triggered_count} règle(s) déclenchée(s)</span>
        <span className={severityBadge(rules.highest_severity)}>{rules.highest_severity}</span>
        {rules.requires_manual_review && <span className="badge badge-amber">Revue manuelle</span>}
      </div>
      <p className="muted">{rules.summary}</p>

      {triggered.length > 0 && (
        <>
          <div className="sectionTitle">Règles déclenchées</div>
          <div className="rulesGrid">
            {triggered.map((r) => (
              <div key={r.rule_id} className="ruleCard ruleCardTriggered">
                <div className="ruleCardHead">
                  <span className={severityBadge(r.severity)}>{r.severity}</span>
                  <span className={actionBadge(r.action)}>{r.action}</span>
                  {r.policy_ref && <code className="policyRef">{r.policy_ref}</code>}
                </div>
                <strong>{r.name}</strong>
                <p className="ruleMessage">{r.message}</p>
                <div className="ruleValues">
                  <span>Valeur: <strong>{formatValue(r.value)}</strong></span>
                  <span>Seuil: <strong>{formatValue(r.threshold)}</strong></span>
                </div>
              </div>
            ))}
          </div>
        </>
      )}

      {triggered.length === 0 && (
        <div className="systemOk">✓ Conformité totale — aucune règle bloquante.</div>
      )}

      {passing.length > 0 && (
        <>
          <div className="sectionTitle">Règles conformes ({passing.length})</div>
          <div className="rulesGrid rulesGridCompact">
            {passing.slice(0, 6).map((r) => (
              <div key={r.rule_id} className="ruleCard ruleCardOk">
                <span className="badge badge-green">OK</span> {r.name}
              </div>
            ))}
          </div>
        </>
      )}
    </section>
  )
}

export function EarlyWarningSystemPanel({ ews }: { ews: EarlyWarningBlock }) {
  return (
    <section className="analysisBlock">
      <div className="cardTitle">Surveillance Early Warning</div>
      <div className="systemMetaRow">
        <span className="badge">{ews.alert_count} alerte(s)</span>
        {ews.critical_count > 0 && <span className="badge badge-red">{ews.critical_count} critique(s)</span>}
        {ews.watchlist_priority && ews.watchlist_priority !== 'NONE' && (
          <span className={watchlistBadge(ews.watchlist_priority)}>Watchlist {ews.watchlist_priority}</span>
        )}
        {ews.degradation_detected && <span className="badge badge-red">Dégradation</span>}
        {ews.n_credits_historique != null && (
          <span className="badge">{ews.n_credits_historique} crédit(s) historique(s)</span>
        )}
      </div>
      <p className="muted">{ews.summary}</p>

      {ews.trend_series && ews.trend_series.length > 0 && (
        <>
          <div className="sectionTitle">Tendances historiques</div>
          <div className="trendGrid">
            {ews.trend_series.map((s) => (
              <MetricTrend key={s.metric} series={s} />
            ))}
          </div>
        </>
      )}

      {ews.alerts.length === 0 ? (
        <div className="systemOk">✓ Profil stable — aucune alerte active.</div>
      ) : (
        <>
          <div className="sectionTitle">Alertes actives</div>
          <div className="alertGrid">
            {ews.alerts.map((a, i) => (
              <div key={`${a.code}-${i}`} className="alertCard">
                <div className="alertCardHead">
                  <span className={severityBadge(a.severity)}>{a.severity}</span>
                  <code>{a.code}</code>
                </div>
                <p>{a.message}</p>
                <div className="alertMetrics">
                  <span>Actuel: <strong>{formatValue(a.current)}</strong></span>
                  {a.baseline != null && <span>Baseline: <strong>{formatValue(a.baseline)}</strong></span>}
                </div>
              </div>
            ))}
          </div>
        </>
      )}
    </section>
  )
}

export function RecommendationSystemPanel({ reco }: { reco: AiRecommendation }) {
  const cf = reco.contributing_factors
  return (
    <section className="analysisBlock">
      <div className="cardTitle">Décision assistée</div>
      <div className={decisionBadge(reco.decision)} style={{ marginTop: 12 }}>
        <div className="decisionLabel">{reco.decision_label}</div>
        <div className="decisionMeta">
          Confiance {(reco.confidence * 100).toFixed(0)}%
          {reco.monitoring_frequency && (
            <span className="badge" style={{ marginLeft: 8 }}>Suivi {reco.monitoring_frequency}</span>
          )}
          {reco.requires_manual_validation && (
            <span className="badge badge-amber" style={{ marginLeft: 8 }}>Validation comité</span>
          )}
        </div>
      </div>

      {(reco.suggested_montant != null || reco.suggested_dti_target != null) && (
        <div className="recoMetrics">
          {reco.suggested_montant != null && (
            <div className="recoMetric">
              <span className="recoMetricLabel">Montant suggéré</span>
              <strong>{reco.suggested_montant.toLocaleString()} TND</strong>
              {reco.montant_reduction_pct != null && (
                <span className="badge badge-amber">−{reco.montant_reduction_pct}%</span>
              )}
            </div>
          )}
          {reco.suggested_dti_target != null && (
            <div className="recoMetric">
              <span className="recoMetricLabel">DTI cible</span>
              <strong>{(reco.suggested_dti_target * 100).toFixed(0)}%</strong>
            </div>
          )}
        </div>
      )}

      {cf && (cf.rules.length > 0 || cf.ews.length > 0) && (
        <>
          <div className="sectionTitle">Facteurs contributifs</div>
          <div className="systemFactors">
            {cf.rules.map((r) => (
              <span key={r} className="badge badge-amber">Règle: {r}</span>
            ))}
            {cf.ews.map((e) => (
              <span key={e} className="badge badge-red">EWS: {e}</span>
            ))}
            <span className="badge">Conformité {cf.compliance_score}/100</span>
          </div>
        </>
      )}

      {reco.conditions && reco.conditions.length > 0 && (
        <>
          <div className="sectionTitle">Conditions</div>
          <ul className="actionList">
            {reco.conditions.map((c) => (
              <li key={c}>{c}</li>
            ))}
          </ul>
        </>
      )}

      <div className="sectionTitle">Justification</div>
      <pre className="pre">{reco.justification}</pre>

      <div className="sectionTitle">Plan d'action</div>
      <ul className="actionList actionListNumbered">
        {reco.recommended_actions.map((a, i) => (
          <li key={a}><span className="actionNum">{i + 1}</span>{a}</li>
        ))}
      </ul>
    </section>
  )
}
