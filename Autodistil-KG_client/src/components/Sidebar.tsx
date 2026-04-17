import { Settings, Activity, FileOutput, Terminal, History, Circle, Trash2, X } from 'lucide-react'
import type { RunSummary } from '../api/client'
import { listRuns, deleteRun } from '../api/client'
import { useEffect, useState, useCallback } from 'react'

export type PageId = 'configure' | 'monitor' | 'results' | 'inference'

const NAV_ITEMS: { id: PageId; label: string; icon: typeof Settings }[] = [
  { id: 'configure', label: 'Configure', icon: Settings },
  { id: 'monitor', label: 'Monitor', icon: Activity },
  { id: 'results', label: 'Results', icon: FileOutput },
  { id: 'inference', label: 'Inference', icon: Terminal },
]

interface SidebarProps {
  activePage: PageId
  onNavigate: (page: PageId) => void
  currentRunId: string | null
  onSelectRun: (runId: string) => void
  onRunDeleted?: (runId: string) => void
  isOpen?: boolean
  onClose?: () => void
}

const STATUS_COLORS: Record<string, string> = {
  running: 'text-primary',
  completed: 'text-green-600/60',
  failed: 'text-red-500',
  queued: 'text-yellow-500',
}

export default function Sidebar({ activePage, onNavigate, currentRunId, onSelectRun, onRunDeleted, isOpen = false, onClose }: SidebarProps) {
  const [runs, setRuns] = useState<RunSummary[]>([])
  const [deleting, setDeleting] = useState<string | null>(null)

  const fetchRuns = useCallback(async () => {
    try {
      const data = await listRuns()
      setRuns(data)
    } catch {
      // ignore
    }
  }, [])

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
  }, [fetchRuns])

  return (
    <>
      {/* Mobile backdrop — closes sidebar on tap outside */}
      {isOpen && (
        <div
          className="fixed inset-0 z-30 bg-black/50 sm:hidden"
          onClick={onClose}
          aria-hidden="true"
        />
      )}
      <aside className={`
        fixed inset-y-0 left-0 z-40 w-56 h-full
        border-r border-border/50 bg-sidebar/70 backdrop-blur-sm
        flex flex-col shrink-0 overflow-hidden
        transition-transform duration-200
        ${isOpen ? 'translate-x-0' : '-translate-x-full'}
        sm:relative sm:translate-x-0
      `}>
      <div className="flex items-center justify-between p-4 border-b border-border/50">
        <div>
          <h1 className="text-lg font-bold text-foreground">Autodistil-KG</h1>
          <p className="text-xs text-muted-foreground">Pipeline Manager</p>
        </div>
        <button
          onClick={onClose}
          className="sm:hidden p-1 rounded-md hover:bg-accent text-muted-foreground hover:text-foreground transition-colors"
          aria-label="Close menu"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      <nav className="flex-1 py-2">
        {NAV_ITEMS.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => { onNavigate(id); onClose?.() }}
            className={`w-full flex items-center gap-3 px-4 py-2.5 text-sm transition-colors ${
              activePage === id
                ? 'bg-primary/10 text-primary border-r-2 border-primary font-medium'
                : 'text-muted-foreground hover:text-foreground hover:bg-accent'
            }`}
          >
            <Icon className="w-4 h-4" />
            {label}
          </button>
        ))}
      </nav>

      <div className="border-t border-border/50">
        <div className="flex items-center gap-2 px-4 py-2 text-xs font-medium text-muted-foreground uppercase tracking-wider">
          <History className="w-3 h-3" />
          Run History
        </div>
        <div className="max-h-48 overflow-y-auto pb-2">
          {runs.length === 0 && (
            <p className="px-4 py-2 text-xs text-muted-foreground">No runs yet</p>
          )}
          {runs.slice(0, 20).map((run) => (
            <div
              key={run.run_id}
              className={`group flex items-center gap-1 px-4 py-1.5 text-xs transition-colors ${
                currentRunId === run.run_id
                  ? 'bg-accent text-foreground'
                  : 'text-muted-foreground hover:text-foreground hover:bg-accent/50'
              }`}
            >
              <button
                onClick={() => onSelectRun(run.run_id)}
                className="flex items-center gap-2 flex-1 min-w-0 text-left"
              >
                <Circle className={`w-2 h-2 fill-current shrink-0 ${STATUS_COLORS[run.status] ?? 'text-gray-400'}`} />
                <span className="truncate font-mono">{run.run_id.slice(0, 8)}</span>
                <span className="ml-auto text-[10px] capitalize">{run.status}</span>
              </button>
              {run.status !== 'running' && (
                <button
                  onClick={(e) => handleDelete(e, run.run_id)}
                  disabled={deleting === run.run_id}
                  className="p-0.5 rounded opacity-0 group-hover:opacity-100 hover:bg-destructive/10 hover:text-destructive transition-all disabled:opacity-50 shrink-0"
                  title="Delete run"
                >
                  <Trash2 className="h-3 w-3" />
                </button>
              )}
            </div>
          ))}
        </div>
      </div>
      </aside>
    </>
  )
}
