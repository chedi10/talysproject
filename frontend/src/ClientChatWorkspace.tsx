import { useCallback, useRef, useState } from 'react'
import { clientChat, type ClientProfile } from './authApi'
import { renderSimpleMarkdown } from './markdown'

type Props = {
  profile: ClientProfile | null
  userId: string
  compact?: boolean
}

type Turn = { role: 'user' | 'assistant'; content: string }

const WELCOME = `Bonjour ! Je suis **Talys Assistant**, votre conseiller digital 24h/24.

Je peux vous aider sur :

- **Votre dossier** — crédits, KYC, échéances, alertes
- **Les démarches** — demander un crédit, mettre à jour le KYC, documents
- **Les solutions** — retard de paiement, rééchelonnement, crédit en défaut
- **Vos questions** — DTI, types de crédits, droits, réclamations

Posez votre question librement ou cliquez une suggestion ci-dessous.`

const DEFAULT_SUGGESTIONS = [
  'Résume mon dossier complet',
  'Comment demander un nouveau crédit ?',
  'Quels documents sont nécessaires ?',
  "J'ai un retard, que faire ?",
  'Comment mettre à jour mon KYC ?',
  'Comment contacter mon agent ?',
]

export default function ClientChatWorkspace({ profile, userId, compact = false }: Props) {
  const sessionId = `client-${userId.slice(0, 12)}`
  const [history, setHistory] = useState<Turn[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [suggestions, setSuggestions] = useState(DEFAULT_SUGGESTIONS)
  const threadRef = useRef<HTMLDivElement>(null)

  const scrollDown = () => {
    requestAnimationFrame(() => {
      threadRef.current?.scrollTo({ top: threadRef.current.scrollHeight, behavior: 'smooth' })
    })
  }

  const send = useCallback(
    async (text: string) => {
      const msg = text.trim()
      if (!msg || loading) return
      setError(null)
      setLoading(true)
      setInput('')
      setHistory((h) => [...h, { role: 'user', content: msg }])
      scrollDown()
      try {
        const res = await clientChat(sessionId, msg)
        setHistory((h) => [...h, { role: 'assistant', content: res.answer }])
        if (res.suggested_prompts?.length) setSuggestions(res.suggested_prompts)
      } catch (err: any) {
        setError(err?.message ?? String(err))
      } finally {
        setLoading(false)
        scrollDown()
      }
    },
    [loading, sessionId],
  )

  return (
    <div className={`clientChatWrap ${compact ? 'clientChatCompact' : ''}`}>
      <section className="card clientChatCard">
        {!compact && (
          <>
            <div className="cardTitle">Talys Assistant</div>
            <p className="muted">
              {profile
                ? `${profile.prenom} ${profile.nom} · CIN ${profile.cin} · Guide dossier & démarches`
                : 'Votre guide microfinance — dossier, démarches et solutions.'}
            </p>
          </>
        )}

        <div className="clientChatThread" ref={threadRef}>
          {history.length === 0 ? (
            <div className="chatBubble chatBubble-assistant">
              <div className="chatBubbleHead">Talys Assistant</div>
              <div className="chatMarkdown" dangerouslySetInnerHTML={{ __html: renderSimpleMarkdown(WELCOME) }} />
            </div>
          ) : (
            history.map((m, i) => (
              <div key={i} className={`chatBubble chatBubble-${m.role}`}>
                <div className="chatBubbleHead">{m.role === 'user' ? 'Vous' : 'Talys Assistant'}</div>
                {m.role === 'assistant' ? (
                  <div className="chatMarkdown" dangerouslySetInnerHTML={{ __html: renderSimpleMarkdown(m.content) }} />
                ) : (
                  <div className="chatUserText">{m.content}</div>
                )}
              </div>
            ))
          )}
          {loading && (
            <div className="chatBubble chatBubble-assistant">
              <div className="chatBubbleHead">Talys Assistant</div>
              <div className="muted">Je prépare une réponse adaptée à votre situation…</div>
            </div>
          )}
        </div>

        {error && (
          <div className="error" style={{ marginTop: 10 }}>
            <pre className="pre">{error}</pre>
          </div>
        )}

        <form
          className="clientChatForm"
          onSubmit={(e) => {
            e.preventDefault()
            send(input)
          }}
        >
          <input
            className="input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ex : Comment demander un crédit ? J'ai un retard…"
            disabled={loading}
          />
          <button className="button" type="submit" disabled={loading || !input.trim()}>
            Envoyer
          </button>
        </form>

        <div className="clientChatSuggestions">
          {suggestions.map((p) => (
            <button key={p} type="button" className="button buttonSmall buttonGhost" disabled={loading} onClick={() => send(p)}>
              {p.length > 44 ? `${p.slice(0, 44)}…` : p}
            </button>
          ))}
        </div>
      </section>
    </div>
  )
}
