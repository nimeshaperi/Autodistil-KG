import { useEffect, useState } from 'react'
import { Trash2 } from 'lucide-react'
import { listRuns, deleteRun, type RunSummary } from '@/api/client'

interface RunHistoryProps {
  currentRunId: string | null
  onSelectRun: (runId: string) => void
  onRunDeleted?: (runId: string) => void
}

const STATUS_COLORS: Record<string, string> = {
  running: 'bg-primary',
  completed: 'bg-green-500/40',
  failed: 'bg-red-500',
  cancelled: 'bg-muted-foreground',
  queued: 'bg-yellow-500',
}

export default function RunHistory({ currentRunId, onSelectRun, onRunDeleted }: RunHistoryProps) {
  const [runs, setRuns] = useState<RunSummary[]>([])
  const [loading, setLoading] = useState(false)
  const [expanded, setExpanded] = useState(false)
  const [deleting, setDeleting] = useState<string | null>(null)

  const fetchRuns = () => {
    setLoading(true)
    listRuns()
      .then(setRuns)
      .catch(() => {})
      .finally(() => setLoading(false))
  }

  const handleDelete = async (e: React.MouseEvent, runId: string) => {
    e.stopPropagation()
    if (!confirm(`Delete run ${runId.slice(0, 8)}...? This removes all output files.`)) return
    setDeleting(runId)
    try {
      await deleteRun(runId)
      setRuns((prev) => prev.filter((r) => r.run_id !== runId))
      onRunDeleted?.(runId)
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Failed to delete run')
    } finally {
      setDeleting(null)
    }
  }

  useEffect(() => {
    fetchRuns()
    const interval = setInterval(fetchRuns, 10_000)
    return () => clearInterval(interval)
  }, [])

  if (!expanded) {
    return (
      <button
        onClick={() => { setExpanded(true); fetchRuns() }}
        className="mb-4 text-sm text-muted-foreground hover:text-foreground flex items-center gap-1.5 transition-colors"
      >
        <span className="text-xs">&#9654;</span>
        Session History
        {runs.length > 0 && (
          <span className="bg-muted text-muted-foreground text-xs px-1.5 py-0.5 rounded-full">
            {runs.length}
          </span>
        )}
      </button>
    )
  }

  return (
    <div className="mb-4 border border-border rounded-lg bg-card">
      <div className="flex items-center justify-between px-3 py-2 border-b border-border">
        <button
          onClick={() => setExpanded(false)}
          className="text-sm font-medium text-foreground flex items-center gap-1.5"
        >
          <span className="text-xs">&#9660;</span>
          Session History
        </button>
        <button
          onClick={fetchRuns}
          disabled={loading}
          className="text-xs text-muted-foreground hover:text-foreground disabled:opacity-50"
        >
          {loading ? 'Refreshing...' : 'Refresh'}
        </button>
      </div>
      {runs.length === 0 ? (
        <p className="px-3 py-3 text-sm text-muted-foreground">No pipeline runs yet.</p>
      ) : (
        <ul className="divide-y divide-border max-h-48 overflow-y-auto">
          {runs.map((run) => {
            const isActive = run.run_id === currentRunId
            const isDeleting = deleting === run.run_id
            const canDelete = run.status !== 'running'
            return (
              <li key={run.run_id}>
                <div
                  className={`flex items-center gap-2 px-3 py-2 text-sm hover:bg-muted/50 transition-colors ${
                    isActive ? 'bg-primary/5 border-l-2 border-primary' : ''
                  }`}
                >
                  <button
                    onClick={() => onSelectRun(run.run_id)}
                    className="flex items-center gap-2 flex-1 min-w-0 text-left"
                  >
                    <span
                      className={`inline-block w-2 h-2 rounded-full flex-shrink-0 ${
                        STATUS_COLORS[run.status] || 'bg-gray-400'
                      }`}
                    />
                    <span className="font-mono text-xs truncate flex-1">{run.run_id}</span>
                    <span className="text-xs text-muted-foreground capitalize">{run.status}</span>
                  </button>
                  {canDelete && (
                    <button
                      onClick={(e) => handleDelete(e, run.run_id)}
                      disabled={isDeleting}
                      className="p-1 rounded hover:bg-destructive/10 text-muted-foreground hover:text-destructive transition-colors disabled:opacity-50 shrink-0"
                      title="Delete run"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  )}
                </div>
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}
