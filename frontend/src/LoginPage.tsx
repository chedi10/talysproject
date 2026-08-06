import { useState } from 'react'
import { login, register } from './authApi'
import type { AuthUser } from './authStorage'

type Mode = 'login' | 'register'
type RegisterRole = 'agent' | 'client'

type Props = {
  onSuccess: (user: AuthUser) => void
}

export default function LoginPage({ onSuccess }: Props) {
  const [mode, setMode] = useState<Mode>('login')
  const [registerRole, setRegisterRole] = useState<RegisterRole>('agent')
  const [username, setUsername] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [password2, setPassword2] = useState('')
  const [cin, setCin] = useState('')
  const [remember, setRemember] = useState(true)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault()
    setError(null)

    const uname = username.trim()
    if (uname.length < 3) {
      setError("Nom d'utilisateur : minimum 3 caractères.")
      return
    }
    if (password.length < 6) {
      setError('Mot de passe : minimum 6 caractères.')
      return
    }
    if (mode === 'register') {
      if (!email.trim().includes('@')) {
        setError('Email invalide.')
        return
      }
      if (password !== password2) {
        setError('Les mots de passe ne correspondent pas.')
        return
      }
      if (registerRole === 'client' && cin.trim().length < 6) {
        setError('CIN requis pour créer un compte client (ex: 88710263).')
        return
      }
    }

    setLoading(true)
    try {
      const res =
        mode === 'login'
          ? await login(uname, password, remember)
          : await register(uname, email.trim(), password, remember, registerRole, cin.trim() || undefined)
      onSuccess(res.user)
    } catch (err: any) {
      setError(err?.message ?? String(err))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="authPage">
      <div className="authCard authCardWide">
        <div className="authBrand">
          <img src="/talys_logo.png" alt="Talys" className="authLogo" />
          <h1 className="authTitle">Talys Banque</h1>
          <p className="authSubtitle">Plateforme intégrée de scoring crédit — base SQLite</p>
        </div>

        <div className="authRoleHint">
          <div className="authRoleBox">
            <strong>Client</strong>
            <span>Consulte son dossier et ses crédits</span>
          </div>
          <div className="authRoleBox">
            <strong>Agent</strong>
            <span>Analyse CIN, chat IA, rapports</span>
          </div>
          <div className="authRoleBox">
            <strong>Admin</strong>
            <span>Gestion utilisateurs et vue globale</span>
          </div>
        </div>

        <div className="authTabs">
          <button
            type="button"
            className={`authTab ${mode === 'login' ? 'authTabActive' : ''}`}
            onClick={() => {
              setMode('login')
              setError(null)
            }}
          >
            Connexion
          </button>
          <button
            type="button"
            className={`authTab ${mode === 'register' ? 'authTabActive' : ''}`}
            onClick={() => {
              setMode('register')
              setError(null)
            }}
          >
            Créer un compte
          </button>
        </div>

        <form className="form authForm" onSubmit={onSubmit}>
          {mode === 'register' && (
            <label className="label">
              Type de compte
              <select className="input" value={registerRole} onChange={(e) => setRegisterRole(e.target.value as RegisterRole)}>
                <option value="agent">Agent crédit</option>
                <option value="client">Client (CIN requis)</option>
              </select>
            </label>
          )}

          <label className="label">
            Nom d'utilisateur
            <input
              className="input"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="ex: agent1"
              autoComplete="username"
              required
              minLength={3}
            />
          </label>

          {mode === 'register' && (
            <label className="label">
              Email
              <input
                className="input"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="agent1@talys.local"
                autoComplete="email"
                required
              />
            </label>
          )}

          {mode === 'register' && registerRole === 'client' && (
            <label className="label">
              CIN (carte d'identité)
              <input
                className="input"
                value={cin}
                onChange={(e) => setCin(e.target.value)}
                placeholder="88710263"
                required
                minLength={6}
              />
            </label>
          )}

          <label className="label">
            Mot de passe
            <input
              className="input"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="min. 6 caractères"
              autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
              required
              minLength={6}
            />
          </label>

          {mode === 'register' && (
            <label className="label">
              Confirmer le mot de passe
              <input
                className="input"
                type="password"
                value={password2}
                onChange={(e) => setPassword2(e.target.value)}
                placeholder="retapez le mot de passe"
                autoComplete="new-password"
                required
                minLength={6}
              />
            </label>
          )}

          <label className="authRemember">
            <input type="checkbox" checked={remember} onChange={(e) => setRemember(e.target.checked)} />
            Rester connecté (session 14 jours)
          </label>

          {mode === 'login' && (
            <p className="muted authHint">
              Admin : <code>admin</code> / <code>admin123</code>
              <br />
              Client démo : <code>client_demo</code> / <code>client123</code> (CIN 88710263)
              <br />
              Nouveau client : onglet « Créer un compte » avec votre CIN bancaire.
            </p>
          )}

          <button className="button authSubmit" type="submit" disabled={loading}>
            {loading ? 'Chargement...' : mode === 'login' ? 'Se connecter' : 'Créer mon compte'}
          </button>
        </form>

        {error && (
          <div className="error authError">
            <div className="errorTitle">Erreur</div>
            <pre className="pre">{error}</pre>
          </div>
        )}
      </div>
    </div>
  )
}
