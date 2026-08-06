import { useMemo, useState } from 'react'

export type GraphNetworkNode = {
  id: string
  client_id: number
  cin?: string
  label: string
  ville?: string
  is_center?: boolean
  en_defaut?: boolean | null
  default_proba?: number | null
  risk_level?: 'FAIBLE' | 'MODERE' | 'ELEVE' | null
}

export type GraphNetworkEdge = {
  id: string
  source: string
  target: string
  type_relation: string
  risk_relation: number
  color: string
  label?: string
  intra?: boolean
}

export type GraphNetworkSnapshot = {
  nodes: GraphNetworkNode[]
  edges: GraphNetworkEdge[]
  stats?: {
    degree?: number
    displayed_neighbors?: number
    relations_by_type?: Record<string, number>
  }
  legend?: Array<{ type: string; color: string; label: string }>
}

type LayoutNode = GraphNetworkNode & { x: number; y: number; r: number }

function riskColor(level?: string | null, isCenter?: boolean) {
  if (isCenter) return '#1E4078'
  if (level === 'FAIBLE') return '#8BC53F'
  if (level === 'MODERE') return '#F59E0B'
  if (level === 'ELEVE') return '#EF4444'
  return '#64748B'
}

function layoutNodes(nodes: GraphNetworkNode[], width: number, height: number): LayoutNode[] {
  const cx = width / 2
  const cy = height / 2
  const center = nodes.find((n) => n.is_center) ?? nodes[0]
  const others = nodes.filter((n) => n.id !== center?.id)
  const radius = Math.min(width, height) * 0.34

  const placed: LayoutNode[] = []
  if (center) {
    placed.push({ ...center, x: cx, y: cy, r: 28 })
  }
  others.forEach((n, i) => {
    const angle = (2 * Math.PI * i) / Math.max(others.length, 1) - Math.PI / 2
    placed.push({
      ...n,
      x: cx + radius * Math.cos(angle),
      y: cy + radius * Math.sin(angle),
      r: 20,
    })
  })
  return placed
}

type Props = {
  network: GraphNetworkSnapshot | null | undefined
}

export default function GraphNetworkViz({ network }: Props) {
  const [hovered, setHovered] = useState<string | null>(null)

  const width = 520
  const height = 400

  const layout = useMemo(
    () => layoutNodes(network?.nodes ?? [], width, height),
    [network?.nodes],
  )
  const byId = useMemo(() => Object.fromEntries(layout.map((n) => [n.id, n])), [layout])

  if (!network?.nodes?.length) {
    return <div className="muted">Réseau non disponible pour ce client.</div>
  }

  const stats = network.stats ?? {}
  const legend = network.legend ?? []

  return (
    <div className="graphVizWrap">
      <div className="graphVizStats">
        <span><b>{stats.displayed_neighbors ?? layout.length - 1}</b> voisins affichés</span>
        <span>Degré total : <b>{stats.degree ?? '—'}</b></span>
        {stats.relations_by_type && (
          <span>
            {Object.entries(stats.relations_by_type).map(([t, n]) => (
              <span key={t} className="graphVizTag">{t} ({n})</span>
            ))}
          </span>
        )}
      </div>

      <svg className="graphVizSvg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Réseau client GAT">
        <defs>
          <filter id="graphGlow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="2" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {network.edges.map((e) => {
          const s = byId[e.source]
          const t = byId[e.target]
          if (!s || !t) return null
          const active = hovered === e.id || hovered === s.id || hovered === t.id
          return (
            <g key={e.id}>
              <line
                x1={s.x}
                y1={s.y}
                x2={t.x}
                y2={t.y}
                stroke={e.color}
                strokeWidth={active ? 3 : e.intra ? 1 : 2}
                strokeOpacity={e.intra ? 0.35 : active ? 1 : 0.65}
                strokeDasharray={e.intra ? '4 3' : undefined}
              />
              {active && !e.intra && (
                <text
                  x={(s.x + t.x) / 2}
                  y={(s.y + t.y) / 2 - 4}
                  className="graphEdgeLabel"
                  textAnchor="middle"
                >
                  {e.label ?? e.type_relation}
                </text>
              )}
            </g>
          )
        })}

        {layout.map((n) => {
          const fill = riskColor(n.risk_level, n.is_center)
          const active = hovered === n.id
          return (
            <g
              key={n.id}
              transform={`translate(${n.x}, ${n.y})`}
              onMouseEnter={() => setHovered(n.id)}
              onMouseLeave={() => setHovered(null)}
              style={{ cursor: 'pointer' }}
            >
              {n.is_center && (
                <circle r={n.r + 8} fill="none" stroke="#F58220" strokeWidth={2} strokeOpacity={0.5} />
              )}
              <circle
                r={n.r}
                fill={fill}
                stroke={active ? '#0f172a' : '#fff'}
                strokeWidth={active ? 3 : 2}
                filter={n.is_center ? 'url(#graphGlow)' : undefined}
              />
              <text className="graphNodeLabel" textAnchor="middle" dy={4}>
                {n.is_center ? '★' : n.label.slice(0, 2).toUpperCase()}
              </text>
              {active && (
                <g transform={`translate(0, ${n.r + 14})`}>
                  <rect
                    x={-72}
                    y={-2}
                    width={144}
                    height={n.is_center ? 52 : 44}
                    rx={6}
                    fill="#0f172a"
                    fillOpacity={0.92}
                  />
                  <text className="graphTooltipTitle" textAnchor="middle" dy={12}>
                    {n.label}
                  </text>
                  {n.is_center && n.cin && (
                    <text className="graphTooltipMeta" textAnchor="middle" dy={26}>
                      CIN {n.cin}
                    </text>
                  )}
                  {n.ville && (
                    <text className="graphTooltipMeta" textAnchor="middle" dy={n.is_center ? 38 : 26}>
                      {n.ville}
                    </text>
                  )}
                  {n.default_proba != null && (
                    <text className="graphTooltipMeta" textAnchor="middle" dy={n.is_center ? 50 : 38}>
                      Proba {Math.round(n.default_proba * 100)}% · {n.risk_level ?? '—'}
                    </text>
                  )}
                </g>
              )}
            </g>
          )
        })}
      </svg>

      {legend.length > 0 && (
        <div className="graphLegend">
          {legend.map((item) => (
            <span key={item.type} className="graphLegendItem">
              <span className="graphLegendSwatch" style={{ background: item.color }} />
              {item.label}
            </span>
          ))}
          <span className="graphLegendItem graphLegendHint">— tirets = lien entre voisins</span>
        </div>
      )}
    </div>
  )
}
