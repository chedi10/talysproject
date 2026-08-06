import type { ActivityListResponse, AuthResponse, AuthUser } from './authStorage'
import { clearAuth, getStoredToken, saveAuth } from './authStorage'

export type { ActivityListResponse, AuthResponse, AuthUser }
export { clearAuth, getStoredToken, loadStoredAuth, saveAuth } from './authStorage'

const API_BASE = (import.meta as any).env?.VITE_API_BASE_URL ?? 'http://localhost:8000'

export function authHeaders(extra: Record<string, string> = {}): Record<string, string> {
  const headers: Record<string, string> = { ...extra }
  const token = getStoredToken()
  if (token) headers.Authorization = `Bearer ${token}`
  return headers
}

async function parseError(res: Response): Promise<string> {
  const text = await res.text().catch(() => '')
  try {
    const j = JSON.parse(text)
    if (j.detail) return typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail)
  } catch {
    /* ignore */
  }
  return text || res.statusText
}

export async function register(
  username: string,
  email: string,
  password: string,
  remember = true,
  role: 'client' | 'agent' = 'agent',
  cin?: string,
): Promise<AuthResponse> {
  const res = await fetch(`${API_BASE}/auth/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, email, password, role, cin: cin || undefined }),
  })
  if (!res.ok) throw new Error(await parseError(res))
  const data = (await res.json()) as AuthResponse
  saveAuth(data.token, data.user, remember)
  return data
}

export async function login(username: string, password: string, remember = true): Promise<AuthResponse> {
  const res = await fetch(`${API_BASE}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  })
  if (!res.ok) throw new Error(await parseError(res))
  const data = (await res.json()) as AuthResponse
  saveAuth(data.token, data.user, remember)
  return data
}

export async function logoutApi(): Promise<void> {
  const token = getStoredToken()
  if (!token) {
    clearAuth()
    return
  }
  try {
    await fetch(`${API_BASE}/auth/logout`, {
      method: 'POST',
      headers: authHeaders(),
    })
  } finally {
    clearAuth()
  }
}

export async function fetchMe(): Promise<AuthUser> {
  const res = await fetch(`${API_BASE}/auth/me`, { headers: authHeaders() })
  if (!res.ok) throw new Error(await parseError(res))
  return (await res.json()) as AuthUser
}

export async function fetchHistory(limit = 100): Promise<ActivityListResponse> {
  const res = await fetch(`${API_BASE}/auth/history?limit=${limit}`, { headers: authHeaders() })
  if (!res.ok) throw new Error(await parseError(res))
  return (await res.json()) as ActivityListResponse
}

export async function fetchUsers(): Promise<AuthUser[]> {
  const res = await fetch(`${API_BASE}/auth/users`, { headers: authHeaders() })
  if (!res.ok) throw new Error(await parseError(res))
  return (await res.json()) as AuthUser[]
}

export type ChatSessionSummary = {
  user_id: string
  username: string
  session_id: string
  message_count: number
  updated_at: string | null
  created_at?: string | null
  title?: string | null
  last_cin?: string | null
  last_intent?: string | null
  last_preview: string | null
}

export async function fetchChatSessions(limit = 50): Promise<{ items: ChatSessionSummary[]; scope: 'mine' | 'all' }> {
  const res = await fetch(`${API_BASE}/auth/chat/sessions?limit=${limit}`, { headers: authHeaders() })
  if (!res.ok) throw new Error(await parseError(res))
  return (await res.json()) as { items: ChatSessionSummary[]; scope: 'mine' | 'all' }
}

export async function fetchChatMessages(sessionId: string): Promise<{ session_id: string; messages: Array<{ role: string; content: string }> }> {
  const res = await fetch(`${API_BASE}/auth/chat/sessions/${encodeURIComponent(sessionId)}`, { headers: authHeaders() })
  if (!res.ok) throw new Error(await parseError(res))
  return (await res.json()) as { session_id: string; messages: Array<{ role: string; content: string }> }
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
  kyc_score: number
  credits: Array<Record<string, unknown>>
  credit_summary: {
    total: number
    actifs: number
    en_defaut: number
    montant_total: number
    dti_moyen: number
  }
  alerts: Array<{ level: 'info' | 'warning' | 'danger'; title: string; message: string }>
  sante_dossier: 'EXCELLENT' | 'BON' | 'A_SURVEILLER' | 'FRAGILE'
  prochaine_echeance: string | null
  taux_retard: number
}

export async function fetchClientProfile(): Promise<ClientProfile> {
  const res = await fetch(`${API_BASE}/client/profile`, { headers: authHeaders() })
  if (!res.ok) throw new Error(await parseError(res))
  return (await res.json()) as ClientProfile
}

export type ClientChatResponse = {
  session_id: string
  intent: string
  answer: string
  rag_sources: Array<{ source: string; chunk_id: number; score: number; text: string }>
  suggested_prompts: string[]
}

export async function clientChat(sessionId: string, message: string): Promise<ClientChatResponse> {
  const res = await fetch(`${API_BASE}/client/chat`, {
    method: 'POST',
    headers: authHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ session_id: sessionId, message }),
  })
  if (!res.ok) throw new Error(await parseError(res))
  return (await res.json()) as ClientChatResponse
}

export type SystemStats = {
  database: string
  clients: number
  credits: number
  transactions: number
  users: number
  activity_log: number
  model_loaded: boolean
  model_name: string
  relations: number
  remboursements: number
  chat_sessions: number
  credits_en_defaut: number
  default_rate: number
  activity_last_7_days: number
  users_by_role: Record<string, number>
  kyc_breakdown: Record<string, number>
  activity_by_action: Record<string, number>
  graph_model: string | null
  graph_auc: number | null
}

export async function fetchAdminStats(): Promise<SystemStats> {
  const res = await fetch(`${API_BASE}/admin/stats`, { headers: authHeaders() })
  if (!res.ok) throw new Error(await parseError(res))
  return (await res.json()) as SystemStats
}

export async function createUser(payload: {
  username: string
  email: string
  password: string
  role: 'client' | 'agent' | 'admin'
  cin?: string
}): Promise<AuthUser> {
  const res = await fetch(`${API_BASE}/auth/users`, {
    method: 'POST',
    headers: authHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify(payload),
  })
  if (!res.ok) throw new Error(await parseError(res))
  return (await res.json()) as AuthUser
}
