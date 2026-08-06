import type { ChatResponse, RagSource } from './api'
import { renderSimpleMarkdown } from './markdown'

type SystemsSummary = {
  rules?: { compliance_score?: number }
  recommendation?: { decision_label?: string; confidence?: number }
}

type Props = {
  chatHistory: Array<{ role: 'user' | 'assistant'; content: string }>
  chatMeta: ChatResponse | null
  reportData: { markdown?: string; sources?: RagSource[]; structured?: Record<string, unknown> } | null
  reportLoading: boolean
  reportDownloading: 'md' | 'pdf' | null
  activeCin: string
  onGenerateReport: () => void
  onDownloadReport: (format: 'md' | 'pdf') => void
}

function riskBadge(level: string) {
  if (level === 'FAIBLE') return 'badge badge-green'
  if (level === 'MODERE') return 'badge badge-amber'
  return 'badge badge-red'
}

function RagSourcesPanel({ sources }: { sources: RagSource[] }) {
  if (!sources?.length) return <div className="muted">Aucune source RAG récupérée.</div>
  return (
    <div className="sourceList">
      {sources.map((s, i) => (
        <div key={`${s.source}-${s.chunk_id}-${i}`} className="sourceItem">
          <div className="sourceHead">
            <code>{s.source}#{s.chunk_id}</code>
            <span className="badge">{(s.score * 100).toFixed(0)}%</span>
          </div>
          <pre className="pre sourceExcerpt">{s.text.length > 280 ? `${s.text.slice(0, 280)}…` : s.text}</pre>
        </div>
      ))}
    </div>
  )
}

export default function ChatWorkspace({
  chatHistory,
  chatMeta,
  reportData,
  reportLoading,
  reportDownloading,
  activeCin,
  onGenerateReport,
  onDownloadReport,
}: Props) {
  const structured = chatMeta?.structured
  const systems = chatMeta?.systems as SystemsSummary | null | undefined

  return (
    <div className="chatWorkspace">
      <div className="cardTitle">Assistant conversationnel Talys</div>
      <p className="muted">LangGraph + RAG — scoring ML, systèmes institutionnels et rapports comité crédit.</p>

      {chatHistory.length === 0 ? (
        <div className="systemEmpty">
          <p>Posez une question ou utilisez un raccourci ci-contre.</p>
        </div>
      ) : (
        <div className="chatThread">
          {chatHistory.map((m, i) => (
            <div key={i} className={`chatBubble chatBubble-${m.role}`}>
              <div className="chatBubbleHead">{m.role === 'user' ? 'Vous' : 'Agent Talys'}</div>
              {m.role === 'assistant' ? (
                <div className="chatMarkdown" dangerouslySetInnerHTML={{ __html: renderSimpleMarkdown(m.content) }} />
              ) : (
                <div className="chatUserText">{m.content}</div>
              )}
            </div>
          ))}
        </div>
      )}

      {chatMeta && (
        <div className="orchestrationMeta">
          <span className="badge">intent: {chatMeta.intent ?? '—'}</span>
          <span className="badge">modèle: {chatMeta.model_selected ?? '—'}</span>
          <span className="badge">CIN: {chatMeta.cin ?? '—'}</span>
          {chatMeta.report_available && <span className="badge badge-green">Rapport généré</span>}
        </div>
      )}

      {(structured || systems) && (
        <>
          <div className="sectionTitle">Synthèse</div>
          <div className="kpis">
            {structured?.kyc_score != null && (
              <div className="kpi">
                <div className="kpiLabel">KYC</div>
                <div className="kpiValue">{structured.kyc_score}</div>
              </div>
            )}
            {structured?.default_proba != null && (
              <div className="kpi">
                <div className="kpiLabel">Proba ML</div>
                <div className="kpiValue">{structured.default_proba}</div>
              </div>
            )}
            {structured?.institutional_score != null && (
              <div className="kpi">
                <div className="kpiLabel">Score institutionnel</div>
                <div className="kpiValue">{(structured.institutional_score * 100).toFixed(1)}%</div>
              </div>
            )}
            {(structured?.risk_level || structured?.institutional_risk) && (
              <div className="kpi">
                <div className="kpiLabel">Risque</div>
                <div className={`kpiValue ${riskBadge(String(structured.risk_level || structured.institutional_risk))}`}>
                  {structured.risk_level || structured.institutional_risk}
                </div>
              </div>
            )}
            {structured?.model_used && (
              <div className="kpi">
                <div className="kpiLabel">Modèle</div>
                <div className="kpiValue">{structured.model_used}</div>
              </div>
            )}
            {systems?.rules?.compliance_score != null && (
              <div className="kpi">
                <div className="kpiLabel">Conformité</div>
                <div className="kpiValue">{systems.rules.compliance_score}/100</div>
              </div>
            )}
          </div>
        </>
      )}

      {systems?.recommendation?.decision_label && (
        <>
          <div className="sectionTitle">Recommandation institutionnelle</div>
          <div className="decisionCard decision-conditional">
            <div className="decisionLabel">{systems.recommendation.decision_label}</div>
            <div className="decisionMeta">
              Confiance {((systems.recommendation.confidence ?? 0) * 100).toFixed(0)}%
            </div>
          </div>
        </>
      )}

      {chatMeta?.rag_sources && chatMeta.rag_sources.length > 0 && (
        <>
          <div className="sectionTitle">Sources RAG ({chatMeta.rag_sources.length})</div>
          <RagSourcesPanel sources={chatMeta.rag_sources} />
        </>
      )}

      {activeCin.length >= 6 && (
        <>
          <div className="sectionTitle">Rapport comité crédit</div>
          <div className="actions">
            <button className="button buttonPrimary" onClick={onGenerateReport} disabled={reportLoading}>
              {reportLoading ? 'Génération…' : 'Aperçu rapport RAG'}
            </button>
            <button className="button" type="button" onClick={() => onDownloadReport('md')} disabled={reportDownloading !== null}>
              {reportDownloading === 'md' ? '…' : 'Télécharger .md'}
            </button>
            <button className="button" type="button" onClick={() => onDownloadReport('pdf')} disabled={reportDownloading !== null}>
              {reportDownloading === 'pdf' ? '…' : 'Télécharger .pdf'}
            </button>
          </div>
          {reportData?.structured && (
            <div className="reportStructuredPreview">
              <span className="badge">PDF enrichi — systèmes institutionnels inclus</span>
              {(reportData.structured as any).systems?.recommendation?.decision_label && (
                <span className="badge badge-amber">
                  {(reportData.structured as any).systems.recommendation.decision_label}
                </span>
              )}
            </div>
          )}
          {reportData?.markdown && (
            <>
              <div className="sectionTitle">Aperçu Markdown</div>
              <div className="chatMarkdown reportPreview" dangerouslySetInnerHTML={{ __html: renderSimpleMarkdown(reportData.markdown.slice(0, 4000)) }} />
            </>
          )}
          {reportData?.sources && reportData.sources.length > 0 && (
            <>
              <div className="sectionTitle">Sources rapport</div>
              <RagSourcesPanel sources={reportData.sources} />
            </>
          )}
        </>
      )}
    </div>
  )
}
