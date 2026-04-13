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
}

export type ModelKind = 'classic' | 'sequential' | 'graph'

const API_BASE = (import.meta as any).env?.VITE_API_BASE_URL ?? 'http://localhost:8000'

function endpointFor(model: ModelKind): string {
  if (model === 'classic') return '/explain/by-cin'
  if (model === 'sequential') return '/explain/sequential/by-cin'
  return '/explain/graph/by-cin'
}

export async function explainByCin(model: ModelKind, cin: string) {
  const res = await fetch(`${API_BASE}${endpointFor(model)}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ cin }),
  })
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`API error ${res.status}: ${text || res.statusText}`)
  }
  return (await res.json()) as unknown
}

