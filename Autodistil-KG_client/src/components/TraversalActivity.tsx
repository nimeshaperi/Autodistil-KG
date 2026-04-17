import { useMemo } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import type { TraversalProgressEvent, WsEvent } from '@/api/client'

/* ------------------------------------------------------------------ */
/*  Derived state from the stream of traversal_progress events        */
/* ------------------------------------------------------------------ */

interface TraversalState {
  strategy: string
  /** Currently active node being processed */
  currentNode: {
    id: string
    labels: string[]
    depth: number
    step: string
  } | null
  /** Subgraph info for the current node (reasoning strategy) */
  subgraph: {
    nodeCount: number
    edgeCount: number
    pathCount: number
    center: { id: string; labels: string[]; properties: Record<string, string> } | null
    nodes: { id: string; labels: string[]; properties?: Record<string, string> }[]
    edges: { source: string; target: string; type: string }[]
  } | null
  /** Current path reasoning progress */
  pathProgress: {
    index: number
    total: number
    description: string
  } | null
  /** Global counters */
  visited: number
  nodesStarted: number
  total: number
  datasetSize: number
  /** Recently completed nodes (last 6) */
  recentNodes: { id: string; labels: string[]; pathsAnalyzed?: number; conversations?: number; error?: string; qualityFiltered?: boolean; qualityScores?: Record<string, number> }[]
  /** Has traversal finished? */
  done: boolean
  /** Alignment summary (set on traversal_complete) */
  alignment?: { quality_filtered?: number; quality_threshold?: number; domain_focus?: string; has_reference_texts?: boolean }
}

function deriveTraversalState(events: TraversalProgressEvent[]): TraversalState {
  const state: TraversalState = {
    strategy: '',
    currentNode: null,
    subgraph: null,
    pathProgress: null,
    visited: 0,
    nodesStarted: 0,
    total: 0,
    datasetSize: 0,
    recentNodes: [],
    done: false,
  }
  for (const ev of events) {
    switch (ev.type) {
      case 'traversal_start':
        state.strategy = ev.strategy ?? ''
        state.total = ev.max_nodes ?? 0
        break
      case 'node_start':
        state.currentNode = {
          id: ev.node_id ?? '?',
          labels: ev.labels ?? [],
          depth: ev.depth ?? 0,
          step: ev.step ?? 'processing',
        }
        state.nodesStarted++
        state.subgraph = null
        state.pathProgress = null
        if (ev.total) state.total = ev.total
        break
      case 'subgraph_loaded':
        state.subgraph = {
          nodeCount: ev.node_count ?? 0,
          edgeCount: ev.edge_count ?? 0,
          pathCount: ev.path_count ?? 0,
          center: ev.center ?? null,
          nodes: ev.nodes ?? [],
          edges: ev.edges ?? [],
        }
        if (state.currentNode) {
          state.currentNode.labels = ev.labels ?? state.currentNode.labels
          state.currentNode.step = 'subgraph_loaded'
        }
        break
      case 'path_reasoning':
        state.pathProgress = {
          index: ev.path_index ?? 0,
          total: ev.total_paths ?? 0,
          description: ev.path_description ?? '',
        }
        if (state.currentNode) state.currentNode.step = 'path_reasoning'
        break
      case 'synthesis':
        state.pathProgress = null
        if (state.currentNode) state.currentNode.step = 'synthesizing'
        break
      case 'qa_generation':
        if (state.currentNode) state.currentNode.step = 'generating_qa'
        break
      case 'quality_scored':
        // Informational — the node_done event handles UI state
        break
      case 'node_done': {
        state.visited = ev.visited ?? state.visited
        state.datasetSize = ev.dataset_size ?? state.datasetSize
        const isError = ev.status === 'error'
        const isFiltered = ev.status === 'quality_filtered'
        state.recentNodes = [
          {
            id: ev.node_id ?? '?',
            labels: ev.labels ?? [],
            pathsAnalyzed: ev.paths_analyzed,
            conversations: ev.conversations,
            ...(isError ? { error: String((ev as unknown as Record<string, unknown>).error ?? 'failed') } : {}),
            ...(isFiltered ? { qualityFiltered: true, qualityScores: ev.quality_scores } : {}),
          },
          ...state.recentNodes,
        ].slice(0, 6)
        state.currentNode = null
        state.subgraph = null
        state.pathProgress = null
        break
      }
      case 'traversal_complete':
        state.visited = ev.visited ?? state.visited
        state.datasetSize = ev.dataset_size ?? state.datasetSize
        state.currentNode = null
        state.done = true
        if (ev.alignment) state.alignment = ev.alignment
        break
    }
  }
  return state
}

/* ------------------------------------------------------------------ */
/*  Step label helper                                                 */
/* ------------------------------------------------------------------ */

const STEP_LABELS: Record<string, { label: string; color: string }> = {
  querying: { label: 'Querying graph', color: 'bg-primary' },
  querying_subgraph: { label: 'Loading subgraph', color: 'bg-primary' },
  subgraph_loaded: { label: 'Subgraph loaded', color: 'bg-primary' },
  llm_generating: { label: 'LLM generating', color: 'bg-primary/80' },
  path_reasoning: { label: 'Reasoning over paths', color: 'bg-primary/70' },
  llm_reasoning: { label: 'LLM reasoning', color: 'bg-primary/70' },
  llm_synthesizing: { label: 'Synthesizing', color: 'bg-primary/60' },
  synthesizing: { label: 'Synthesizing insights', color: 'bg-primary/60' },
  generating_qa: { label: 'Generating QA pair', color: 'bg-green-500/40' },
  llm_generating_qa: { label: 'Generating QA pair', color: 'bg-green-500/40' },
  processing: { label: 'Processing', color: 'bg-gray-500' },
}

function StepBadge({ step }: { step: string }) {
  const info = STEP_LABELS[step] ?? { label: step, color: 'bg-gray-500' }
  return (
    <span className={`inline-flex items-center gap-1.5 text-xs px-2 py-0.5 rounded-full text-white ${info.color}`}>
      <span className="relative flex h-2 w-2">
        <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${info.color}`} />
        <span className={`relative inline-flex rounded-full h-2 w-2 ${info.color}`} />
      </span>
      {info.label}
    </span>
  )
}

/* ------------------------------------------------------------------ */
/*  Subgraph node cards (key-value properties)                         */
/* ------------------------------------------------------------------ */

interface NodeCardProps {
  id: string
  labels: string[]
  properties: Record<string, string>
  isCenter?: boolean
  relationshipType?: string
}

function NodeCard({ id, labels, properties, isCenter, relationshipType }: NodeCardProps) {
  const entries = Object.entries(properties)
  return (
    <div className={`rounded-md border p-2 text-xs space-y-1.5 ${isCenter ? 'border-primary/40 bg-primary/5' : 'bg-muted/20'}`}>
      <div className="flex items-center gap-1.5 min-w-0">
        <span className={`h-2 w-2 rounded-full shrink-0 ${isCenter ? 'bg-primary' : 'bg-primary/60'}`} />
        <span className="font-mono font-medium truncate">{id}</span>
        {relationshipType && (
          <Badge variant="secondary" className="text-[9px] py-0 ml-auto shrink-0">{relationshipType}</Badge>
        )}
      </div>
      {labels.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {labels.map((l) => (
            <Badge key={l} variant="outline" className="text-[9px] py-0">{l}</Badge>
          ))}
        </div>
      )}
      {entries.length > 0 && (
        <div className="grid grid-cols-[auto_1fr] gap-x-2 gap-y-0.5">
          {entries.map(([key, val]) => (
            <div key={key} className="contents">
              <span className="text-muted-foreground truncate">{key}</span>
              <span className="truncate">{val}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

interface SubgraphNodeCardsProps {
  center: { id: string; labels: string[]; properties: Record<string, string> } | null
  nodes: { id: string; labels: string[]; properties?: Record<string, string> }[]
  edges: { source: string; target: string; type: string }[]
}

function SubgraphNodeCards({ center, nodes, edges }: SubgraphNodeCardsProps) {
  // Build a map of node_id → relationship types connecting to/from center
  const relationshipMap = useMemo(() => {
    const m = new Map<string, string>()
    for (const e of edges) {
      if (center && e.source === center.id) m.set(e.target, e.type)
      else if (center && e.target === center.id) m.set(e.source, e.type)
    }
    return m
  }, [center, edges])

  if (!center && nodes.length === 0) return null

  // Only show first 8 neighbors to keep it compact
  const displayNodes = nodes.slice(0, 8)
  const remaining = nodes.length - displayNodes.length

  return (
    <div className="space-y-2 animate-in fade-in duration-300">
      <p className="text-xs font-medium text-muted-foreground">Subgraph Nodes</p>

      {/* Center node */}
      {center && (
        <NodeCard
          id={center.id}
          labels={center.labels}
          properties={center.properties}
          isCenter
        />
      )}

      {/* Neighbor nodes */}
      {displayNodes.length > 0 && (
        <div className="grid grid-cols-2 gap-1.5">
          {displayNodes.map((node) => (
            <NodeCard
              key={node.id}
              id={node.id}
              labels={node.labels}
              properties={node.properties ?? {}}
              relationshipType={relationshipMap.get(node.id)}
            />
          ))}
        </div>
      )}
      {remaining > 0 && (
        <p className="text-[10px] text-muted-foreground text-center">+{remaining} more nodes</p>
      )}
    </div>
  )
}

/* ------------------------------------------------------------------ */
/*  Main component                                                    */
/* ------------------------------------------------------------------ */

interface TraversalActivityProps {
  wsEvents: WsEvent[]
}

export default function TraversalActivity({ wsEvents }: TraversalActivityProps) {
  const traversalEvents = useMemo(
    () => wsEvents.filter((e): e is TraversalProgressEvent => e.event === 'traversal_progress'),
    [wsEvents],
  )

  const state = useMemo(() => deriveTraversalState(traversalEvents), [traversalEvents])

  // Don't render if no traversal events yet
  if (traversalEvents.length === 0) return null

  // Use nodesStarted for progress (updates on every node_start, not just node_done)
  const progressCount = Math.max(state.visited, state.nodesStarted)
  const progressPct = state.total > 0 ? Math.min(100, Math.round((progressCount / state.total) * 100)) : 0

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Graph Traversal</CardTitle>
          <div className="flex items-center gap-2">
            {state.strategy && (
              <Badge variant="secondary" className="text-xs capitalize">
                {state.strategy}
              </Badge>
            )}
            {state.done ? (
              <Badge variant="success" className="text-xs">Complete</Badge>
            ) : state.currentNode ? (
              <StepBadge step={state.currentNode.step} />
            ) : null}
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* ── Counters ── */}
        <div className="grid grid-cols-3 gap-3 text-center">
          <div className="rounded-lg border bg-muted/30 p-2">
            <div className="text-lg font-bold tabular-nums">{state.visited}<span className="text-sm font-normal text-muted-foreground">/{state.nodesStarted}</span></div>
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider">Done / Started</div>
          </div>
          <div className="rounded-lg border bg-muted/30 p-2">
            <div className="text-lg font-bold tabular-nums">{state.datasetSize}</div>
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider">Conversations</div>
          </div>
          <div className="rounded-lg border bg-muted/30 p-2">
            <div className="text-lg font-bold tabular-nums">{state.total > 0 ? `${progressPct}%` : '--'}</div>
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider">Progress</div>
          </div>
        </div>

        {/* ── Progress bar ── */}
        {state.total > 0 && (
          <div className="flex items-center gap-2">
            <Progress value={progressPct} className="flex-1 h-1.5" />
            <span className="text-xs text-muted-foreground tabular-nums w-16 text-right">
              {progressCount}/{state.total}
            </span>
          </div>
        )}

        {/* ── Current node activity ── */}
        {state.currentNode && (
          <div className="rounded-lg border bg-muted/20 p-3 space-y-2 animate-in fade-in duration-300">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 min-w-0">
                <div className="h-3 w-3 rounded-full bg-primary animate-pulse shrink-0" />
                <span className="text-sm font-mono truncate">{state.currentNode.id}</span>
              </div>
              <span className="text-xs text-muted-foreground shrink-0 ml-2">depth {state.currentNode.depth}</span>
            </div>

            {state.currentNode.labels.length > 0 && (
              <div className="flex flex-wrap gap-1">
                {state.currentNode.labels.map((l) => (
                  <Badge key={l} variant="outline" className="text-[10px] py-0">{l}</Badge>
                ))}
              </div>
            )}

            {/* Subgraph info */}
            {state.subgraph && (
              <div className="flex gap-3 text-xs text-muted-foreground">
                <span>{state.subgraph.nodeCount} nodes</span>
                <span>{state.subgraph.edgeCount} edges</span>
                <span>{state.subgraph.pathCount} paths</span>
              </div>
            )}

            {/* Subgraph node properties */}
            {state.subgraph && (state.subgraph.center || state.subgraph.nodes.length > 0) && (
              <SubgraphNodeCards
                center={state.subgraph.center}
                nodes={state.subgraph.nodes}
                edges={state.subgraph.edges}
              />
            )}

            {/* Path reasoning progress */}
            {state.pathProgress && (
              <div className="space-y-1">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-muted-foreground">
                    Path {state.pathProgress.index}/{state.pathProgress.total}
                  </span>
                  <span className="text-muted-foreground tabular-nums">
                    {Math.round((state.pathProgress.index / state.pathProgress.total) * 100)}%
                  </span>
                </div>
                <Progress
                  value={(state.pathProgress.index / state.pathProgress.total) * 100}
                  className="h-1"
                />
                {state.pathProgress.description && (
                  <p className="text-[11px] text-muted-foreground font-mono truncate leading-tight">
                    {state.pathProgress.description}
                  </p>
                )}
              </div>
            )}
          </div>
        )}

        {/* ── Recent nodes ── */}
        {state.recentNodes.length > 0 && (
          <div className="space-y-1.5">
            <p className="text-xs font-medium text-muted-foreground">Recently processed</p>
            <div className="space-y-1">
              {state.recentNodes.map((node, i) => (
                <div
                  key={`${node.id}-${i}`}
                  className="flex items-center justify-between text-xs rounded px-2 py-1 bg-muted/30 animate-in slide-in-from-top-1 duration-200"
                  style={{ animationDelay: `${i * 50}ms` }}
                >
                  <div className="flex items-center gap-2 min-w-0">
                    <div className={`h-1.5 w-1.5 rounded-full shrink-0 ${node.error ? 'bg-red-500' : node.qualityFiltered ? 'bg-amber-500' : 'bg-green-500/40'}`} />
                    <span className="font-mono truncate">{node.id}</span>
                    {node.labels.map((l) => (
                      <Badge key={l} variant="outline" className="text-[9px] py-0 hidden sm:inline-flex">{l}</Badge>
                    ))}
                  </div>
                  <div className="flex gap-2 text-muted-foreground shrink-0 ml-2">
                    {node.error ? (
                      <span className="text-red-500 truncate max-w-[150px]">failed</span>
                    ) : node.qualityFiltered ? (
                      <span className="text-amber-500 truncate max-w-[180px]" title={node.qualityScores ? `rel=${node.qualityScores.relevance?.toFixed(2)} gnd=${node.qualityScores.groundedness?.toFixed(2)} cmp=${node.qualityScores.completeness?.toFixed(2)}` : ''}>
                        filtered (avg {node.qualityScores?.avg?.toFixed(2) ?? '?'})
                      </span>
                    ) : (
                      <>
                        {node.pathsAnalyzed != null && <span>{node.pathsAnalyzed} paths</span>}
                        {node.conversations != null && <span>{node.conversations} conv</span>}
                      </>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── Done message ── */}
        {state.done && (
          <div className="text-center py-2 text-sm text-green-600 dark:text-green-400 font-medium animate-in fade-in duration-500">
            <p>Traversal complete — {state.datasetSize} conversations generated</p>
            {state.alignment?.quality_filtered != null && state.alignment.quality_filtered > 0 && (
              <p className="text-amber-500 text-xs font-normal mt-1">
                {state.alignment.quality_filtered} Q&amp;A pairs filtered by quality gate (threshold {state.alignment.quality_threshold})
              </p>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
