import { useEffect, useState } from 'react'
import LoginPage from './LoginPage'
import ClientPortal from './ClientPortal'
import AdminDashboard from './AdminDashboard'
import DashboardApp from './DashboardApp'
import { clearAuth, fetchMe, loadStoredAuth, type AuthUser } from './authApi'

export default function App() {
  const [user, setUser] = useState<AuthUser | null>(() => loadStoredAuth()?.user ?? null)
  const [checking, setChecking] = useState(true)
  const [adminAgentMode, setAdminAgentMode] = useState(false)

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      const stored = loadStoredAuth()
      if (!stored) {
        if (!cancelled) {
          setUser(null)
          setChecking(false)
        }
        return
      }
      try {
        const me = await fetchMe()
        if (!cancelled) setUser(me)
      } catch {
        clearAuth()
        if (!cancelled) setUser(null)
      } finally {
        if (!cancelled) setChecking(false)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [])

  if (checking) {
    return (
      <div className="authPage">
        <div className="authCard authCardCompact">
          <p className="muted">Vérification de la session...</p>
        </div>
      </div>
    )
  }

  if (!user) {
    return <LoginPage onSuccess={setUser} />
  }

  if (user.role === 'client') {
    return <ClientPortal user={user} onLogout={() => setUser(null)} />
  }

  if (user.role === 'admin') {
    if (adminAgentMode) {
      return (
        <DashboardApp
          user={user}
          onLogout={() => setUser(null)}
          onBackToAdmin={() => setAdminAgentMode(false)}
        />
      )
    }
    return (
      <AdminDashboard
        user={user}
        onLogout={() => setUser(null)}
        onAgentMode={() => setAdminAgentMode(true)}
      />
    )
  }

  return <DashboardApp user={user} onLogout={() => setUser(null)} />
}
