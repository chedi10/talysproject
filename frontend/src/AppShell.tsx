import type { ReactNode } from 'react'
import type { AuthUser } from './authApi'

export type NavItem = {
  id: string
  label: string
  icon?: string
}

type Props = {
  user: AuthUser
  navItems: NavItem[]
  activeNav: string
  onNavChange: (id: string) => void
  onLogout: () => void
  children: ReactNode
  subtitle?: string
}

function roleBadge(role: string) {
  if (role === 'admin') return 'badge badge-amber'
  if (role === 'client') return 'badge badge-blue'
  return 'badge badge-green'
}

function roleLabel(role: string) {
  if (role === 'admin') return 'Administrateur'
  if (role === 'client') return 'Client'
  return 'Agent crédit'
}

export default function AppShell({ user, navItems, activeNav, onNavChange, onLogout, children, subtitle }: Props) {
  return (
    <div className="bankApp">
      <header className="bankHeader">
        <div className="bankHeaderBrand">
          <img src="/talys_logo.png" alt="Talys" className="bankLogo" />
          <div>
            <div className="bankTitle">Talys Banque — Scoring Crédit</div>
            <div className="bankSubtitle">{subtitle ?? 'Système intégré de gestion du risque'}</div>
          </div>
        </div>
        <div className="bankHeaderUser">
          <div className="bankUserMeta">
            <strong>{user.username}</strong>
            <span className={roleBadge(user.role)}>{roleLabel(user.role)}</span>
          </div>
          <button type="button" className="button buttonGhost" onClick={onLogout}>
            Déconnexion
          </button>
        </div>
      </header>

      <nav className="bankNav" aria-label="Navigation principale">
        {navItems.map((item) => (
          <button
            key={item.id}
            type="button"
            className={`bankNavItem ${activeNav === item.id ? 'bankNavItemActive' : ''}`}
            onClick={() => onNavChange(item.id)}
          >
            {item.icon && <span className="bankNavIcon">{item.icon}</span>}
            {item.label}
          </button>
        ))}
      </nav>

      <main className="bankMain">{children}</main>

      <footer className="bankFooter muted">
        Base de données SQLite · <code>data/local/talys.db</code>
      </footer>
    </div>
  )
}
