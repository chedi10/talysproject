import { useEffect, useState } from 'react'
import AppShell from './AppShell'
import { fetchClientProfile, logoutApi, type AuthUser, type ClientProfile } from './authApi'
import ClientChatWorkspace from './ClientChatWorkspace'

type Props = {
  user: AuthUser
  onLogout: () => void
}

type View = 'accueil' | 'credits' | 'alertes' | 'assistant' | 'contact'

function santeBadge(sante: string) {
  if (sante === 'EXCELLENT') return 'badge badge-green'
  if (sante === 'BON') return 'badge badge-green'
  if (sante === 'A_SURVEILLER') return 'badge badge-amber'
  return 'badge badge-red'
}

function santeLabel(sante: string) {
  const map: Record<string, string> = {
    EXCELLENT: 'Excellent',
    BON: 'Bon',
    A_SURVEILLER: 'À surveiller',
    FRAGILE: 'Fragile',
  }
  return map[sante] ?? sante
}

function alertClass(level: string) {
  if (level === 'danger') return 'alertCard alertDanger'
  if (level === 'warning') return 'alertCard alertWarning'
  return 'alertCard alertInfo'
}

export default function ClientPortal({ user, onLogout }: Props) {
  const [view, setView] = useState<View>('accueil')
  const [profile, setProfile] = useState<ClientProfile | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [chatOpen, setChatOpen] = useState(false)

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const p = await fetchClientProfile()
        if (!cancelled) setProfile(p)
      } catch (err: any) {
        if (!cancelled) setError(err?.message ?? String(err))
      } finally {
        if (!cancelled) setLoading(false)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [])

  async function handleLogout() {
    await logoutApi()
    onLogout()
  }

  const navItems = [
    { id: 'accueil', label: 'Accueil', icon: '🏠' },
    { id: 'assistant', label: 'Assistant', icon: '💬' },
    { id: 'credits', label: 'Mes crédits', icon: '💳' },
    { id: 'alertes', label: 'Alertes', icon: '🔔' },
    { id: 'contact', label: 'Contact', icon: '📞' },
  ]

  const summary = profile?.credit_summary
  const alertCount = profile?.alerts?.filter((a) => a.level !== 'info').length ?? 0

  return (
    <AppShell
      user={user}
      navItems={navItems}
      activeNav={view}
      onNavChange={(id) => setView(id as View)}
      onLogout={handleLogout}
      subtitle="Espace client — suivi de votre relation Talys Microfinance"
    >
      {loading && <div className="card muted">Chargement de votre dossier...</div>}
      {error && (
        <div className="card error">
          <div className="errorTitle">Erreur</div>
          <pre className="pre">{error}</pre>
        </div>
      )}

      {!loading && !error && profile && view === 'assistant' && (
        <ClientChatWorkspace profile={profile} userId={user.id} />
      )}

      {!loading && !error && profile && view === 'accueil' && (
        <div className="portalGrid">
          <section className="card portalHero clientHero">
            <div className="portalHeroContent">
              <div className="cardTitle">Bonjour, {profile.prenom} {profile.nom}</div>
              <p className="muted">CIN {profile.cin} · {profile.ville} · {profile.profession}</p>
            </div>
            <div className="portalHeroBadge">
              <span className="portalHeroBadgeLabel">Santé du dossier</span>
              <span className={santeBadge(profile.sante_dossier)}>{santeLabel(profile.sante_dossier)}</span>
              <span className="muted">Score KYC {profile.kyc_score}/100</span>
            </div>
          </section>

          <section className="card">
            <div className="cardTitle">Synthèse</div>
            <div className="kpis">
              <div className="kpi">
                <div className="kpiLabel">Crédits actifs</div>
                <div className="kpiValue">{summary?.actifs ?? 0}</div>
              </div>
              <div className="kpi">
                <div className="kpiLabel">Encours total</div>
                <div className="kpiValue">{summary?.montant_total?.toLocaleString() ?? 0} TND</div>
              </div>
              <div className="kpi">
                <div className="kpiLabel">DTI moyen</div>
                <div className="kpiValue">{((summary?.dti_moyen ?? 0) * 100).toFixed(0)}%</div>
              </div>
              <div className="kpi">
                <div className="kpiLabel">Taux retard</div>
                <div className={`kpiValue ${profile.taux_retard > 0.2 ? 'badge-red' : ''}`}>
                  {(profile.taux_retard * 100).toFixed(0)}%
                </div>
              </div>
            </div>
            <div className="infoGrid">
              <div><span className="muted">Revenu</span> {profile.revenu_mensuel.toLocaleString()} TND/mois</div>
              <div><span className="muted">Âge</span> {profile.age} ans</div>
              <div><span className="muted">Statut KYC</span> <span className="badge">{profile.statut_kyc}</span></div>
              {profile.prochaine_echeance && (
                <div><span className="muted">Prochaine échéance</span> {profile.prochaine_echeance}</div>
              )}
            </div>
          </section>

          {profile.alerts.length > 0 && (
            <section className="card">
              <div className="cardTitle">Alertes récentes ({alertCount})</div>
              <div className="alertList">
                {profile.alerts.slice(0, 3).map((a, i) => (
                  <div key={i} className={alertClass(a.level)}>
                    <strong>{a.title}</strong>
                    <p>{a.message}</p>
                  </div>
                ))}
              </div>
              {profile.alerts.length > 3 && (
                <button type="button" className="button buttonSmall buttonGhost" onClick={() => setView('alertes')}>
                  Voir toutes les alertes
                </button>
              )}
              <button type="button" className="button buttonSmall" style={{ marginLeft: 8 }} onClick={() => setView('assistant')}>
                Demander à l'assistant →
              </button>
            </section>
          )}
        </div>
      )}

      {!loading && !error && profile && view === 'credits' && (
        <section className="card">
          <div className="cardTitle">Mes crédits ({profile.credits.length})</div>
          {profile.credits.length ? (
            <div className="creditCardGrid">
              {profile.credits.map((c: any) => (
                <div key={c.credit_id} className={`creditCard ${c.en_defaut ? 'creditCardDefault' : ''}`}>
                  <div className="creditCardHead">
                    <strong>{c.objet}</strong>
                    <span className={c.en_defaut ? 'badge badge-red' : 'badge badge-green'}>
                      {c.en_defaut ? 'En défaut' : 'Actif'}
                    </span>
                  </div>
                  <div className="creditCardMeta">
                    <span>{Number(c.montant).toLocaleString()} TND</span>
                    <span>{c.duree_mois} mois</span>
                    <span>DTI {(Number(c.dti) * 100).toFixed(0)}%</span>
                  </div>
                  <div className="muted creditCardDate">Début {String(c.date_debut).slice(0, 10)} · #{c.credit_id}</div>
                </div>
              ))}
            </div>
          ) : (
            <div className="muted">Aucun crédit enregistré.</div>
          )}
        </section>
      )}

      {!loading && !error && profile && view === 'alertes' && (
        <section className="card">
          <div className="cardTitle">Centre d'alertes</div>
          <div className="alertList">
            {profile.alerts.map((a, i) => (
              <div key={i} className={alertClass(a.level)}>
                <strong>{a.title}</strong>
                <p>{a.message}</p>
              </div>
            ))}
          </div>
        </section>
      )}

      {!loading && !error && view === 'contact' && (
        <div className="grid">
          <section className="card">
            <div className="cardTitle">Votre conseiller Talys</div>
            <p className="muted">
              Pour toute demande de crédit, rééchelonnement ou mise à jour KYC, contactez votre agence ou votre agent référent.
            </p>
            <ul className="actionList">
              <li>Agence {profile?.ville ?? '—'} — Lun–Ven 8h30–17h</li>
              <li>Téléphone : 71 000 000 (simulation)</li>
              <li>Email : contact@talys.local</li>
              <li>Délai de réponse moyen : 24 h ouvrées</li>
            </ul>
          </section>
          <section className="card">
            <div className="cardTitle">Documents & informations</div>
            <ul className="docList">
              <li><span>Contrat de crédit</span><span className="badge">Disponible en agence</span></li>
              <li><span>Relevé de remboursements</span><span className="badge badge-green">À jour</span></li>
              <li><span>Attestation KYC</span><span className={`badge ${profile?.statut_kyc === 'OK' ? 'badge-green' : 'badge-amber'}`}>{profile?.statut_kyc}</span></li>
              <li><span>Données personnelles</span><span className="muted">Hébergées SQLite sécurisé</span></li>
            </ul>
          </section>
        </div>
      )}

      {!loading && !error && profile && (
        <>
          <button
            type="button"
            className="clientChatFab"
            onClick={() => {
              setChatOpen((o) => !o)
              if (!chatOpen) setView('assistant')
            }}
            aria-label="Ouvrir Talys Assistant"
            title="Talys Assistant — démarches & solutions"
          >
            {chatOpen ? '✕' : '💬'}
          </button>
          {chatOpen && view !== 'assistant' && (
            <div className="clientChatOverlay">
              <div className="clientChatOverlayHead">
                <strong>Talys Assistant</strong>
                <button type="button" className="button buttonSmall buttonGhost" onClick={() => setView('assistant')}>
                  Plein écran
                </button>
                <button type="button" className="button buttonSmall buttonGhost" onClick={() => setChatOpen(false)}>
                  Fermer
                </button>
              </div>
              <ClientChatWorkspace profile={profile} userId={user.id} compact />
            </div>
          )}
        </>
      )}
    </AppShell>
  )
}
