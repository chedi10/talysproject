export type ShapFeatureImpact = {
  feature: string
  label: string
  impact: number
}

export type ShapDriverDetail = {
  feature: string
  label: string
  impact: number
  value?: number | null
  portfolio_median?: number | null
  vs_portfolio_pct?: number | null
}

export type ShapExplanation = {
  method: string
  model_used: string
  increases_risk: ShapFeatureImpact[]
  decreases_risk: ShapFeatureImpact[]
  summary: string
  base_prediction?: number | null
  driver_details?: ShapDriverDetail[]
  credit_context?: {
    montant?: number
    duree_mois?: number
    dti?: number
    objet?: string
    cycle?: string
  }
}

export type BusinessRuleResult = {
  rule_id: string
  name: string
  triggered: boolean
  severity: 'INFO' | 'WARNING' | 'CRITICAL'
  action: 'none' | 'alert' | 'manual_review' | 'block'
  message: string
  value?: unknown
  threshold?: unknown
  policy_ref?: string
}

export type BusinessRulesBlock = {
  rules: BusinessRuleResult[]
  triggered_count: number
  triggered_rule_ids?: string[]
  requires_manual_review: boolean
  highest_severity: 'INFO' | 'WARNING' | 'CRITICAL'
  compliance_score?: number
  summary: string
  credit_snapshot?: Record<string, unknown>
}

export type EarlyWarningAlert = {
  code: string
  severity: 'INFO' | 'WARNING' | 'CRITICAL'
  message: string
  metric: string
  current: number | string
  baseline?: number | string
}

export type TrendPoint = {
  credit_id: number
  date: string
  value: number
  is_current: boolean
}

export type TrendSeries = {
  metric: string
  label: string
  points: TrendPoint[]
}

export type EarlyWarningBlock = {
  alerts: EarlyWarningAlert[]
  alert_count: number
  critical_count: number
  degradation_detected: boolean
  summary: string
  watchlist_priority?: 'NONE' | 'LOW' | 'MEDIUM' | 'HIGH'
  trend_series?: TrendSeries[]
  n_credits_historique?: number
}

export type AiDecision =
  | 'ACCEPTER'
  | 'ACCEPTER_AVEC_GARANTIE'
  | 'REDUIRE_MONTANT'
  | 'DEMANDER_GARANT'
  | 'REFUSER'

export type ContributingFactors = {
  rules: string[]
  ews: string[]
  compliance_score: number
}

export type AiRecommendation = {
  decision: AiDecision
  decision_label: string
  confidence: number
  justification: string
  recommended_actions: string[]
  requires_manual_validation: boolean
  suggested_montant?: number | null
  montant_reduction_pct?: number | null
  suggested_dti_target?: number | null
  monitoring_frequency?: string
  conditions?: string[]
  contributing_factors?: ContributingFactors
}

export type ClientProfile = {
  cin: string
  client_id: number
  nom: string
  prenom: string
  age: number
  ville: string
  profession: string
  revenu_mensuel: number
  statut_kyc: string
}

export type CreditAnalysisBundle = {
  shap: ShapExplanation
  business_rules: BusinessRulesBlock
  early_warning: EarlyWarningBlock
  recommendation: AiRecommendation
}

function severityBadge(severity: string) {
  if (severity === 'CRITICAL') return 'badge badge-red'
  if (severity === 'WARNING') return 'badge badge-amber'
  return 'badge'
}

function decisionBadge(decision: AiDecision) {
  if (decision === 'ACCEPTER') return 'decisionCard decision-accept'
  if (decision === 'REFUSER') return 'decisionCard decision-refuse'
  return 'decisionCard decision-conditional'
}

function ShapBar({ label, impact, maxAbs }: { label: string; impact: number; maxAbs: number }) {
  const pct = maxAbs > 0 ? Math.min(100, (Math.abs(impact) / maxAbs) * 100) : 0
  const positive = impact > 0
  return (
    <div className="shapRow">
      <div className="shapLabel" title={label}>
        {label}
      </div>
      <div className="shapTrack">
        <div className={`shapFill ${positive ? 'shap-up' : 'shap-down'}`} style={{ width: `${pct}%` }} />
      </div>
      <div className={`shapValue ${positive ? 'shap-up-text' : 'shap-down-text'}`}>
        {impact > 0 ? '+' : ''}
        {impact.toFixed(3)}
      </div>
    </div>
  )
}

export function ShapPanel({ shap }: { shap: ShapExplanation }) {
  const allImpacts = [...shap.increases_risk, ...shap.decreases_risk]
  const maxAbs = Math.max(...allImpacts.map((x) => Math.abs(x.impact)), 0.001)
  return (
    <section className="analysisBlock">
      <div className="cardTitle">Explainable AI (SHAP)</div>
      <div className="muted" style={{ margin: '10px 0' }}>
        {shap.summary} <span className="badge">{shap.method}</span>
      </div>
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

export function RulesPanel({ rules }: { rules: BusinessRulesBlock }) {
  const triggered = rules.rules.filter((r) => r.triggered)
  return (
    <section className="analysisBlock">
      <div className="cardTitle">Business Rules Engine</div>
      <div className="muted" style={{ margin: '10px 0' }}>
        {rules.summary}
        {rules.requires_manual_review && <span className="badge badge-amber" style={{ marginLeft: 8 }}>Revue manuelle</span>}
      </div>
      {triggered.length === 0 ? (
        <div className="muted">Aucune règle déclenchée.</div>
      ) : (
        <ul className="alertList">
          {triggered.map((r) => (
            <li key={r.rule_id} className="alertItem">
              <span className={severityBadge(r.severity)}>{r.severity}</span>
              <strong>{r.name}</strong> — {r.message}
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}

export function EarlyWarningPanel({ ews }: { ews: EarlyWarningBlock }) {
  return (
    <section className="analysisBlock">
      <div className="cardTitle">Early Warning System</div>
      <div className="muted" style={{ margin: '10px 0' }}>
        {ews.summary}
        {ews.degradation_detected && <span className="badge badge-red" style={{ marginLeft: 8 }}>Dégradation</span>}
      </div>
      {ews.alerts.length === 0 ? (
        <div className="muted">Aucune alerte.</div>
      ) : (
        <ul className="alertList">
          {ews.alerts.map((a, i) => (
            <li key={`${a.code}-${i}`} className="alertItem">
              <span className={severityBadge(a.severity)}>{a.severity}</span>
              {a.message}
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}

export function RecommendationPanel({ reco }: { reco: AiRecommendation }) {
  return (
    <section className="analysisBlock">
      <div className="cardTitle">Recommandation IA</div>
      <div className={decisionBadge(reco.decision)} style={{ marginTop: 12 }}>
        <div className="decisionLabel">{reco.decision_label}</div>
        <div className="decisionMeta">
          Confiance {(reco.confidence * 100).toFixed(0)}%
          {reco.requires_manual_validation && <span className="badge badge-amber" style={{ marginLeft: 8 }}>Validation requise</span>}
        </div>
      </div>
      <pre className="pre" style={{ marginTop: 10 }}>
        {reco.justification}
      </pre>
      <ul className="actionList">
        {reco.recommended_actions.map((a) => (
          <li key={a}>{a}</li>
        ))}
      </ul>
    </section>
  )
}

export function AnalysisPanel({ analysis }: { analysis: CreditAnalysisBundle }) {
  return (
    <div style={{ display: 'grid', gap: 16 }}>
      <ShapPanel shap={analysis.shap} />
      <RulesPanel rules={analysis.business_rules} />
      <EarlyWarningPanel ews={analysis.early_warning} />
      <RecommendationPanel reco={analysis.recommendation} />
    </div>
  )
}
