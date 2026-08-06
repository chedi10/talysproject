export type RiskLevel = 'FAIBLE' | 'MODERE' | 'ELEVE'

export type CreditExplanationItem = {
  credit_id: number
  prediction: 0 | 1
  default_proba: number
  risk_level: RiskLevel
}

export type ExplainClassicResponse = {
  cin: string
  credit_id: number
  kyc_score: number
  prediction: 0 | 1
  default_proba: number
  risk_level: RiskLevel
  model_used: string
  message: string
}

export type ExplainSequentialResponse = {
  cin: string
  kyc_score: number
  prediction: 0 | 1
  default_proba: number
  risk_level: RiskLevel
  model_used: string
  message: string
  credits: CreditExplanationItem[]
  n_credits: number
}

export type ExplainGraphResponse = {
  cin: string
  kyc_score: number
  prediction: 0 | 1
  default_proba: number
  risk_level: RiskLevel
  model_used: string
  message: string
  network?: import('./GraphNetworkViz').GraphNetworkSnapshot | null
}

export type ModelKind = 'classic' | 'sequential' | 'graph' | 'ensemble'

export type EnsembleModelScore = {
  model_key: 'classic' | 'sequential' | 'graph'
  model_name: string
  weight: number
  available: boolean
  default_proba?: number | null
  risk_level?: RiskLevel | null
  prediction?: number | null
  error?: string | null
}

export type ExplainEnsembleResponse = {
  cin: string
  credit_id?: number | null
  kyc_score: number
  prediction: 0 | 1
  default_proba: number
  risk_level: RiskLevel
  model_used: string
  method: string
  models: EnsembleModelScore[]
  vote_default: number
  vote_non_default: number
  agreement: 'unanimous' | 'majority' | 'split'
  models_available: number
  models_total: number
  message: string
  network?: import('./GraphNetworkViz').GraphNetworkSnapshot | null
}

import { authHeaders } from './authApi'

const API_BASE = (import.meta as any).env?.VITE_API_BASE_URL ?? 'http://localhost:8000'

function endpointFor(model: ModelKind): string {
  if (model === 'classic') return '/explain/by-cin'
  if (model === 'sequential') return '/explain/sequential/by-cin'
  if (model === 'ensemble') return '/explain/ensemble/by-cin'
  return '/explain/graph/by-cin'
}

export async function explainByCin(model: ModelKind, cin: string) {
  const res = await fetch(`${API_BASE}${endpointFor(model)}`, {
    method: 'POST',
    headers: authHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ cin }),
  })
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`API error ${res.status}: ${text || res.statusText}`)
  }
  return (await res.json()) as unknown
}

async function callSystem(path: string, cin: string) {
  const res = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: authHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ cin }),
  })
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`API error ${res.status}: ${text || res.statusText}`)
  }
  return await res.json()
}

export const fetchShapSystem = (cin: string) => callSystem('/systems/shap/by-cin', cin)
export const fetchRulesSystem = (cin: string) => callSystem('/systems/rules/by-cin', cin)
export const fetchEarlyWarningSystem = (cin: string) => callSystem('/systems/early-warning/by-cin', cin)
export const fetchRecommendationSystem = (cin: string) => callSystem('/systems/recommendation/by-cin', cin)

export type ChatStructuredResult = {
  kyc_score?: number | null
  default_proba?: number | null
  risk_level?: string | null
  model_used?: string | null
  institutional_score?: number | null
  institutional_risk?: string | null
}

export type ChatResponse = {
  session_id: string
  model_selected: 'classic' | 'sequential' | 'graph' | 'ensemble' | null
  intent: 'classic_score' | 'sequential_score' | 'graph_score' | 'full_report' | 'compare_models' | 'institutional' | null
  cin: string | null
  answer: string
  rag_sources?: RagSource[]
  structured?: ChatStructuredResult | null
  systems?: Record<string, unknown> | null
  report_available?: boolean
  suggested_prompts?: string[]
}

export type RagSource = {
  source: string
  chunk_id: number
  score: number
  text: string
}

export type ReportResponse = {
  cin: string
  markdown: string
  sources: RagSource[]
  structured?: Record<string, unknown> | null
}

export async function chat(session_id: string, message: string): Promise<ChatResponse> {
  const res = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: authHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ session_id, message }),
  })
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`API error ${res.status}: ${text || res.statusText}`)
  }
  return (await res.json()) as ChatResponse
}

export async function downloadReport(cin: string, format: 'md' | 'pdf'): Promise<void> {
  const res = await fetch(`${API_BASE}/report/by-cin/download`, {
    method: 'POST',
    headers: authHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ cin, format }),
  })
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`API error ${res.status}: ${text || res.statusText}`)
  }

  const blob = await res.blob()
  const ext = format === 'pdf' ? 'pdf' : 'md'
  const filename = `report_${cin}.${ext}`
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  URL.revokeObjectURL(url)
}

export async function generateReport(cin: string): Promise<ReportResponse> {
  const res = await fetch(`${API_BASE}/report/by-cin`, {
    method: 'POST',
    headers: authHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ cin }),
  })
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`API error ${res.status}: ${text || res.statusText}`)
  }
  return (await res.json()) as ReportResponse
}

