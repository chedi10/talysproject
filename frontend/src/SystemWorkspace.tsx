import type { ReactNode } from 'react'
import {
  EarlyWarningSystemPanel,
  RecommendationSystemPanel,
  RulesSystemPanel,
  ShapSystemPanel,
  type SystemClientContext,
} from './SystemsPanel'
import type { ClientProfile } from './AnalysisPanel'

export type SystemKind = 'shap' | 'rules' | 'ews' | 'recommendation'

const META: Record<
  SystemKind,
  { title: string; subtitle: string; role: string; icon: string; accent: string }
> = {
  shap: {
    title: 'Explainable AI',
    subtitle: 'Transparence des décisions — facteurs explicatifs par client',
    role: 'Analyste risque · Conformité · Audit',
    icon: '🔍',
    accent: 'systemAccentCyan',
  },
  rules: {
    title: 'Business Rules Engine',
    subtitle: 'Politique crédit, KYC, DTI et règles réglementaires',
    role: 'Compliance Officer · Agent crédit',
    icon: '📜',
    accent: 'systemAccentOrange',
  },
  ews: {
    title: 'Early Warning System',
    subtitle: 'Surveillance proactive — détection de dégradation comportementale',
    role: 'Risk Manager · Collections',
    icon: '⚠️',
    accent: 'systemAccentAmber',
  },
  recommendation: {
    title: 'Recommandation IA',
    subtitle: 'Décision assistée avec justification et plan d\'action',
    role: 'Comité crédit · Agent décisionnel',
    icon: '⭐',
    accent: 'systemAccentGreen',
  },
}

type Props = {
  kind: SystemKind
  cin: string
  onCinChange: (v: string) => void
  onSubmit: (e: React.FormEvent) => void
  loading: boolean
  canSubmit: boolean
  error: string | null
  result: any | null
}

function riskBadge(level: string) {
  if (level === 'FAIBLE') return 'badge badge-green'
  if (level === 'MODERE') return 'badge badge-amber'
  return 'badge badge-red'
}

function ClientContextStrip({ ctx }: { ctx: SystemClientContext & { client_profile?: ClientProfile } }) {
  const p = ctx.client_profile
  return (
    <>
      {p && p.nom && (
        <div className="clientProfileCard">
          <div className="clientProfileName">{p.prenom} {p.nom}</div>
          <div className="clientProfileMeta">
            {p.age > 0 && <span>{p.age} ans</span>}
            {p.ville && <span>{p.ville}</span>}
            {p.profession && <span>{p.profession}</span>}
            {p.revenu_mensuel > 0 && <span>{p.revenu_mensuel.toLocaleString()} TND/mois</span>}
            {p.statut_kyc && <span className="badge">{p.statut_kyc}</span>}
          </div>
        </div>
      )}
      <div className="systemContextStrip">
      <div className="systemContextItem">
        <span className="systemContextLabel">CIN</span>
        <strong>{ctx.cin}</strong>
      </div>
      <div className="systemContextItem">
        <span className="systemContextLabel">Crédit</span>
        <strong>#{ctx.credit_id}</strong>
      </div>
      <div className="systemContextItem">
        <span className="systemContextLabel">KYC</span>
        <strong>{ctx.kyc_score}</strong>
      </div>
      <div className="systemContextItem">
        <span className="systemContextLabel">Score institutionnel</span>
        <strong>{(ctx.institutional_score * 100).toFixed(1)}%</strong>
      </div>
      <div className="systemContextItem">
        <span className="systemContextLabel">Risque</span>
        <span className={riskBadge(ctx.risk_level)}>{ctx.risk_level}</span>
      </div>
      </div>
    </>
  )
}

function EmptyState({ children }: { children: ReactNode }) {
  return (
    <div className="systemEmpty">
      <div className="systemEmptyIcon">◎</div>
      <p>{children}</p>
    </div>
  )
}

export default function SystemWorkspace({
  kind,
  cin,
  onCinChange,
  onSubmit,
  loading,
  canSubmit,
  error,
  result,
}: Props) {
  const meta = META[kind]

  return (
    <div className={`systemWorkspace ${meta.accent}`}>
      <header className="systemHero">
        <div className="systemHeroIcon">{meta.icon}</div>
        <div className="systemHeroText">
          <div className="systemHeroTag">Système autonome · indépendant des modèles ML</div>
          <h1 className="systemHeroTitle">{meta.title}</h1>
          <p className="systemHeroSubtitle">{meta.subtitle}</p>
          <div className="systemHeroRole">
            <span className="systemRoleLabel">Rôle métier</span>
            {meta.role}
          </div>
        </div>
      </header>

      <div className="systemLayout">
        <aside className="systemControl card">
          <div className="cardTitle">Identification client</div>
          <form className="form" onSubmit={onSubmit}>
            <label className="label">
              CIN client
              <input
                className="input"
                value={cin}
                onChange={(e) => onCinChange(e.target.value)}
                placeholder="88710263"
                autoComplete="off"
              />
            </label>
            <p className="systemHint">
              Analyse basée sur les données SQLite — KYC, retards, transactions et historique crédit.
            </p>
            <button className="button buttonPrimary" disabled={!canSubmit} type="submit">
              {loading ? 'Analyse en cours…' : 'Lancer l\'analyse'}
            </button>
          </form>
          {error && (
            <div className="error" style={{ marginTop: 12 }}>
              <div className="errorTitle">Erreur</div>
              <pre className="pre">{error}</pre>
            </div>
          )}
        </aside>

        <main className="systemResults card">
          {!result ? (
            <EmptyState>Saisissez un CIN et lancez l'analyse pour afficher les résultats.</EmptyState>
          ) : (
            <>
              <ClientContextStrip ctx={result} />
              {result.risk_factors?.length > 0 && (
                <div className="systemFactors">
                  {result.risk_factors.map((f: string) => (
                    <span key={f} className="badge badge-amber">
                      {f}
                    </span>
                  ))}
                </div>
              )}
              {kind === 'shap' && <ShapSystemPanel shap={result.shap} />}
              {kind === 'rules' && <RulesSystemPanel rules={result.business_rules} />}
              {kind === 'ews' && <EarlyWarningSystemPanel ews={result.early_warning} />}
              {kind === 'recommendation' && <RecommendationSystemPanel reco={result.recommendation} />}
            </>
          )}
        </main>
      </div>
    </div>
  )
}
