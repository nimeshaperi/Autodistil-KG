import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import ForceGraph2D from 'react-force-graph-2d'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import type { TraversalProgressEvent, WsEvent } from '@/api/client'

/* ------------------------------------------------------------------ */
/*  Types for the force-directed graph                                 */
/* ------------------------------------------------------------------ */

interface GraphNode {
  id: string
  labels: string[]
  state: 'pending' | 'processing' | 'completed' | 'failed'
  depth: number
  /** Animation progress 0-1 for pulse effect */
  __pulse?: number
  /** Timestamp when node started processing */
  __startedAt?: number
  /** Number of conversations generated from this node */
  conversations?: number
  /** Number of paths analyzed */
  pathsAnalyzed?: number
  x?: number
  y?: number
  fx?: number
  fy?: number
}

interface GraphLink {
  source: string
  target: string
  label?: string
}

interface GraphData {
  nodes: GraphNode[]
  links: GraphLink[]
}

/* ------------------------------------------------------------------ */
/*  Derive graph data from traversal events                            */
/* ------------------------------------------------------------------ */

/** Maximum nodes/edges rendered to prevent browser crashes on large graphs */
const MAX_GRAPH_NODES = 200
const MAX_GRAPH_LINKS = 500

function deriveGraphData(events: TraversalProgressEvent[]): {
  graphData: GraphData
  strategy: string
  visited: number
  total: number
  datasetSize: number
  done: boolean
  currentNodeId: string | null
  currentStep: string | null
  totalNodesBeforeCap: number
  totalLinksBeforeCap: number
} {
  const nodeMap = new Map<string, GraphNode>()
  const linkSet = new Set<string>()
  const links: GraphLink[] = []
  let strategy = ''
  let visited = 0
  let total = 0
  let datasetSize = 0
  let done = false
  let currentNodeId: string | null = null
  let currentStep: string | null = null

  for (const ev of events) {
    // Track the latest step description
    if (ev.step) currentStep = ev.step

    switch (ev.type) {
      case 'traversal_start':
        strategy = ev.strategy ?? ''
        total = ev.max_nodes ?? 0
        break

      case 'node_start': {
        const id = ev.node_id ?? '?'
        currentNodeId = id
        if (!nodeMap.has(id)) {
          nodeMap.set(id, {
            id,
            labels: ev.labels ?? [],
            state: 'processing',
            depth: ev.depth ?? 0,
            __startedAt: Date.now(),
          })
        } else {
          const existing = nodeMap.get(id)!
          existing.state = 'processing'
          existing.labels = ev.labels ?? existing.labels
          existing.__startedAt = Date.now()
        }
        // Real edges are added by subgraph_loaded / neighbors_loaded events
        if (ev.total) total = ev.total
        break
      }

      case 'subgraph_loaded': {
        // Reasoning strategy — add center node, neighbor nodes, and real edges
        const id = ev.node_id
        if (id && nodeMap.has(id)) {
          const node = nodeMap.get(id)!
          node.labels = ev.labels ?? node.labels
        }
        // Add neighbor nodes from the subgraph
        if (ev.nodes) {
          for (const n of ev.nodes) {
            if (!nodeMap.has(n.id)) {
              nodeMap.set(n.id, {
                id: n.id,
                labels: n.labels ?? [],
                state: 'pending',
                depth: (id ? (nodeMap.get(id)?.depth ?? 0) : 0) + 1,
              })
            }
          }
        }
        // Add real relationship edges from the subgraph
        if (ev.edges) {
          for (const e of ev.edges) {
            const linkKey = `${e.source}→${e.target}`
            const reverseKey = `${e.target}→${e.source}`
            if (!linkSet.has(linkKey) && !linkSet.has(reverseKey)) {
              linkSet.add(linkKey)
              links.push({ source: e.source, target: e.target, label: e.type })
            }
          }
        }
        if (ev.visited) visited = ev.visited
        if (ev.total) total = ev.total
        break
      }

      case 'neighbors_loaded': {
        // Simple strategy — add neighbor nodes and edges to center
        const id = ev.node_id
        if (id) {
          if (nodeMap.has(id)) {
            const node = nodeMap.get(id)!
            node.labels = ev.labels ?? node.labels
          }
          if (ev.neighbors) {
            const neighbors = ev.neighbors
            for (const n of neighbors) {
              if (!nodeMap.has(n.id)) {
                nodeMap.set(n.id, {
                  id: n.id,
                  labels: n.labels ?? [],
                  state: 'pending',
                  depth: (nodeMap.get(id)?.depth ?? 0) + 1,
                })
              }
              const linkKey = `${id}→${n.id}`
              const reverseKey = `${n.id}→${id}`
              if (!linkSet.has(linkKey) && !linkSet.has(reverseKey)) {
                linkSet.add(linkKey)
                links.push({ source: id, target: n.id, label: n.relationship_type })
              }
            }
          }
        }
        break
      }

      case 'node_done': {
        const id = ev.node_id ?? '?'
        const isError = ev.status === 'error'
        if (nodeMap.has(id)) {
          const node = nodeMap.get(id)!
          node.state = isError ? 'failed' : 'completed'
          node.conversations = ev.conversations
          node.pathsAnalyzed = ev.paths_analyzed
        }
        visited = ev.visited ?? visited
        datasetSize = ev.dataset_size ?? datasetSize
        if (currentNodeId === id) currentNodeId = null
        break
      }

      case 'traversal_complete':
        visited = ev.visited ?? visited
        datasetSize = ev.dataset_size ?? datasetSize
        done = true
        currentNodeId = null
        break
    }
  }

  const allNodes = Array.from(nodeMap.values())
  const totalNodesBeforeCap = allNodes.length
  const totalLinksBeforeCap = links.length

  // Prioritise: processing > completed > pending; then by depth (lower first)
  const STATE_PRIORITY: Record<string, number> = { processing: 0, completed: 1, failed: 2, pending: 3 }
  const cappedNodes = allNodes.length > MAX_GRAPH_NODES
    ? allNodes
        .sort((a, b) => (STATE_PRIORITY[a.state] ?? 9) - (STATE_PRIORITY[b.state] ?? 9) || a.depth - b.depth)
        .slice(0, MAX_GRAPH_NODES)
    : allNodes

  const visibleIds = new Set(cappedNodes.map((n) => n.id))
  const cappedLinks = links
    .filter((l) => visibleIds.has(l.source) && visibleIds.has(l.target))
    .slice(0, MAX_GRAPH_LINKS)

  return {
    graphData: {
      nodes: cappedNodes,
      links: cappedLinks,
    },
    strategy,
    visited,
    total,
    datasetSize,
    done,
    currentNodeId,
    totalNodesBeforeCap,
    totalLinksBeforeCap,
    currentStep: done ? null : currentStep,
  }
}

/* ------------------------------------------------------------------ */
/*  Color schemes                                                      */
/* ------------------------------------------------------------------ */

const NODE_COLORS: Record<string, string> = {
  pending: '#94a3b8',    // slate-400
  processing: '#5b7fb5', // muted blue
  completed: '#22c55e',  // green-500
  failed: '#ef4444',     // red-500
}

const NODE_COLORS_GLOW: Record<string, string> = {
  processing: 'rgba(91, 127, 181, 0.4)',
  completed: 'rgba(34, 197, 94, 0.15)',
}

/* ------------------------------------------------------------------ */
/*  Main component                                                     */
/* ------------------------------------------------------------------ */

interface GraphVisualizationProps {
  wsEvents: WsEvent[]
}

export default function GraphVisualization({ wsEvents }: GraphVisualizationProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const graphRef = useRef<{ d3Force: (name: string) => unknown; zoomToFit: (ms?: number, px?: number) => void } | null>(null)
  const [dimensions, setDimensions] = useState<{ width: number; height: number } | null>(null)
  const animFrameRef = useRef<number>(0)
  const [, forceRender] = useState(0)

  // Extract traversal events
  const traversalEvents = useMemo(
    () => wsEvents.filter((e): e is TraversalProgressEvent => e.event === 'traversal_progress'),
    [wsEvents],
  )

  const { graphData, done, currentNodeId, totalNodesBeforeCap, totalLinksBeforeCap } = useMemo(
    () => deriveGraphData(traversalEvents),
    [traversalEvents],
  )

  const isCapped = totalNodesBeforeCap > MAX_GRAPH_NODES || totalLinksBeforeCap > MAX_GRAPH_LINKS

  // Resize observer — measure the container and pass exact dimensions to ForceGraph2D
  // Re-run when traversalEvents arrive (component may have been returning null before)
  const hasEvents = traversalEvents.length > 0
  useEffect(() => {
    if (!hasEvents) return
    const el = containerRef.current
    if (!el) return
    const measure = () => {
      const rect = el.getBoundingClientRect()
      if (rect.width > 0 && rect.height > 0) {
        setDimensions({ width: Math.floor(rect.width), height: Math.floor(rect.height) })
      }
    }
    // Initial measurement (with small delay for layout to settle)
    const raf = requestAnimationFrame(measure)
    const ro = new ResizeObserver(() => measure())
    ro.observe(el)
    return () => {
      cancelAnimationFrame(raf)
      ro.disconnect()
    }
  }, [hasEvents])

  // Center the graph whenever nodes change or dimensions become available
  const prevNodeCount = useRef(0)
  useEffect(() => {
    if (!dimensions || graphData.nodes.length === 0) return
    if (graphData.nodes.length !== prevNodeCount.current) {
      prevNodeCount.current = graphData.nodes.length
      const timer = setTimeout(() => {
        graphRef.current?.zoomToFit(400, 60)
      }, 600)
      return () => clearTimeout(timer)
    }
  }, [graphData.nodes.length, dimensions])

  // Animation loop for pulsing active nodes
  useEffect(() => {
    if (done) return
    let running = true
    const tick = () => {
      if (!running) return
      forceRender((n) => n + 1)
      animFrameRef.current = requestAnimationFrame(tick)
    }
    // Throttle to ~15fps for performance
    const interval = setInterval(() => {
      animFrameRef.current = requestAnimationFrame(tick)
    }, 66)
    return () => {
      running = false
      cancelAnimationFrame(animFrameRef.current)
      clearInterval(interval)
    }
  }, [done])

  const nodeCanvasObject = useCallback(
    (node: GraphNode, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const x = node.x ?? 0
      const y = node.y ?? 0
      const isActive = node.id === currentNodeId || node.state === 'processing'
      const baseRadius = isActive ? 6 : node.state === 'completed' ? 5 : 4
      const radius = baseRadius / Math.max(globalScale * 0.5, 0.5)

      // Glow effect for active nodes
      if (isActive) {
        const pulseT = ((Date.now() - (node.__startedAt ?? Date.now())) / 800) % 1
        const glowRadius = radius * (1.5 + pulseT * 1.5)
        const gradient = ctx.createRadialGradient(x, y, radius, x, y, glowRadius)
        gradient.addColorStop(0, NODE_COLORS_GLOW.processing!)
        gradient.addColorStop(1, 'rgba(91, 127, 181, 0)')
        ctx.beginPath()
        ctx.arc(x, y, glowRadius, 0, 2 * Math.PI)
        ctx.fillStyle = gradient
        ctx.fill()
      } else if (node.state === 'completed') {
        const gradient = ctx.createRadialGradient(x, y, radius, x, y, radius * 1.5)
        gradient.addColorStop(0, NODE_COLORS_GLOW.completed!)
        gradient.addColorStop(1, 'rgba(34, 197, 94, 0)')
        ctx.beginPath()
        ctx.arc(x, y, radius * 1.5, 0, 2 * Math.PI)
        ctx.fillStyle = gradient
        ctx.fill()
      }

      // Node circle
      ctx.beginPath()
      ctx.arc(x, y, radius, 0, 2 * Math.PI)
      ctx.fillStyle = NODE_COLORS[node.state] ?? NODE_COLORS.pending
      ctx.fill()

      // Border
      ctx.strokeStyle = isActive ? '#5b7fb5' : node.state === 'completed' ? '#16a34a' : '#64748b'
      ctx.lineWidth = isActive ? 1.5 / globalScale : 0.5 / globalScale
      ctx.stroke()
    },
    [currentNodeId],
  )

  const linkCanvasObject = useCallback(
    (link: GraphLink, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const src = link.source as unknown as GraphNode
      const tgt = link.target as unknown as GraphNode
      if (!src?.x || !tgt?.x) return

      ctx.beginPath()
      ctx.moveTo(src.x, src.y ?? 0)
      ctx.lineTo(tgt.x, tgt.y ?? 0)
      ctx.strokeStyle = 'rgba(148, 163, 184, 0.3)'
      ctx.lineWidth = 0.5 / globalScale
      ctx.stroke()

      // Arrow
      const dx = tgt.x - src.x
      const dy = (tgt.y ?? 0) - (src.y ?? 0)
      const dist = Math.sqrt(dx * dx + dy * dy)
      if (dist < 1) return
      const nx = dx / dist
      const ny = dy / dist
      const arrowLen = 4 / globalScale
      const ax = tgt.x - nx * 6 / globalScale
      const ay = (tgt.y ?? 0) - ny * 6 / globalScale
      ctx.beginPath()
      ctx.moveTo(ax, ay)
      ctx.lineTo(ax - nx * arrowLen + ny * arrowLen * 0.4, ay - ny * arrowLen - nx * arrowLen * 0.4)
      ctx.lineTo(ax - nx * arrowLen - ny * arrowLen * 0.4, ay - ny * arrowLen + nx * arrowLen * 0.4)
      ctx.closePath()
      ctx.fillStyle = 'rgba(148, 163, 184, 0.5)'
      ctx.fill()

      // Edge labels only on extreme zoom
      if (link.label && globalScale > 4) {
        const midX = (src.x + tgt.x) / 2
        const midY = ((src.y ?? 0) + (tgt.y ?? 0)) / 2
        const fontSize = Math.max(6 / globalScale, 1.5)
        ctx.font = `${fontSize}px sans-serif`
        ctx.textAlign = 'center'
        ctx.textBaseline = 'middle'
        ctx.fillStyle = 'rgba(148, 163, 184, 0.6)'
        ctx.fillText(link.label, midX, midY - 2 / globalScale)
      }
    },
    [],
  )

  // Don't render if no traversal events
  if (traversalEvents.length === 0) return null

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Graph Visualization</CardTitle>
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="text-xs tabular-nums">
              {totalNodesBeforeCap} nodes · {totalLinksBeforeCap} edges
            </Badge>
            {isCapped && (
              <Badge variant="secondary" className="text-xs text-amber-600 dark:text-amber-400">
                Showing {graphData.nodes.length}/{totalNodesBeforeCap}
              </Badge>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {/* Legend */}
        <div className="flex items-center gap-4 mb-3 text-xs text-muted-foreground">
          <div className="flex items-center gap-1.5">
            <span className="inline-block w-2.5 h-2.5 rounded-full bg-primary animate-pulse" />
            Processing
          </div>
          <div className="flex items-center gap-1.5">
            <span className="inline-block w-2.5 h-2.5 rounded-full bg-green-500/40" />
            Completed
          </div>
          <div className="flex items-center gap-1.5">
            <span className="inline-block w-2.5 h-2.5 rounded-full bg-slate-400" />
            Pending
          </div>
        </div>

        {/* Graph canvas */}
        <div
          ref={containerRef}
          className="rounded-lg border bg-slate-950/5 dark:bg-slate-50/5"
          style={{ height: 450, position: 'relative' }}
        >
          {dimensions && (
            <ForceGraph2D
              ref={graphRef as React.MutableRefObject<never>}
              graphData={graphData}
              width={dimensions.width}
              height={dimensions.height}
              nodeCanvasObject={nodeCanvasObject as never}
              linkCanvasObject={linkCanvasObject as never}
              nodeId="id"
              cooldownTicks={graphData.nodes.length > 50 ? 80 : 150}
              d3AlphaDecay={0.02}
              d3VelocityDecay={0.3}
              enableZoomInteraction={true}
              enablePanInteraction={true}
              onEngineStop={() => graphRef.current?.zoomToFit(400, 60)}
              backgroundColor="transparent"
              nodeLabel={(node: GraphNode) => {
                const parts = [node.id]
                if (node.labels.length) parts.push(node.labels.join(', '))
                parts.push(node.state)
                if (node.conversations) parts.push(`${node.conversations} conversations`)
                if (node.pathsAnalyzed) parts.push(`${node.pathsAnalyzed} paths`)
                return parts.join(' · ')
              }}
            />
          )}
        </div>
      </CardContent>
    </Card>
  )
}
