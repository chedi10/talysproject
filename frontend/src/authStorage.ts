const TOKEN_KEY = 'talys_auth_token'
const USER_KEY = 'talys_auth_user'

export type UserRole = 'client' | 'agent' | 'admin'

export type AuthUser = {
  id: string
  username: string
  email: string
  role: UserRole
  cin?: string | null
  client_id?: number | null
  created_at?: string | null
}

export type AuthResponse = {
  token: string
  user: AuthUser
}

export type ActivityRecord = {
  id: string
  user_id: string
  username: string
  role: string
  action: string
  cin: string | null
  model: string | null
  intent: string | null
  message_preview: string | null
  session_id: string | null
  created_at: string
}

export type ActivityListResponse = {
  items: ActivityRecord[]
  total: number
  scope: 'mine' | 'all'
}

function readStore(): Storage {
  if (sessionStorage.getItem(TOKEN_KEY)) return sessionStorage
  return localStorage
}

export function loadStoredAuth(): { token: string; user: AuthUser } | null {
  const store = readStore()
  const token = store.getItem(TOKEN_KEY) || localStorage.getItem(TOKEN_KEY)
  const raw = store.getItem(USER_KEY) || localStorage.getItem(USER_KEY)
  if (!token || !raw) return null
  try {
    return { token, user: JSON.parse(raw) as AuthUser }
  } catch {
    return null
  }
}

export function saveAuth(token: string, user: AuthUser, remember = true) {
  clearAuth()
  const store = remember ? localStorage : sessionStorage
  store.setItem(TOKEN_KEY, token)
  store.setItem(USER_KEY, JSON.stringify(user))
}

export function clearAuth() {
  for (const store of [localStorage, sessionStorage]) {
    store.removeItem(TOKEN_KEY)
    store.removeItem(USER_KEY)
  }
}

export function getStoredToken(): string | null {
  return sessionStorage.getItem(TOKEN_KEY) || localStorage.getItem(TOKEN_KEY)
}

export function getStoredUser(): AuthUser | null {
  return loadStoredAuth()?.user ?? null
}
