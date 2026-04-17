import React, { useState, useCallback, useRef } from 'react'
import Sidebar from './components/Sidebar'
import TopBar from './components/TopBar'
import { ErrorBoundary } from './components/ErrorBoundary'
import ConfigurePipeline from './components/ConfigurePipeline'
import MonitorProgress from './components/MonitorProgress'
import ResultsOutput from './components/ResultsOutput'
import InferencePlayground from './components/InferencePlayground'
import type { PageId } from './components/Sidebar'
import type { PipelineConfigPayload } from './types/config'
import type { WsEvent, WsRunHandle } from './api/client'
import { stopRun } from './api/client'


interface RunSession {
  runId: string
  config: PipelineConfigPayload | null
  wsEvents: WsEvent[]
  result: unknown
  isWsConnected: boolean | undefined
}

function emptySession(runId: string, config: PipelineConfigPayload | null = null): RunSession {
  return { runId, config, wsEvents: [], result: null, isWsConnected: undefined }
}

export default function App() {
  const [activePage, setActivePage] = useState<PageId>('configure')
  const [sessions, setSessions] = useState<Map<string, RunSession>>(new Map())
  const [activeRunId, setActiveRunId] = useState<string | null>(null)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const activeSession = activeRunId ? sessions.get(activeRunId) : undefined
  const wsHandleRef = useRef<WsRunHandle | null>(null)

  const handleStop = useCallback(async () => {
    // Try WebSocket stop first (for live connections)
    if (wsHandleRef.current) {
      wsHandleRef.current.stop()
    }
    // Also try REST stop (for async/polling runs)
    if (activeRunId) {
      try { await stopRun(activeRunId) } catch { /* may not be running */ }
    }
  }, [activeRunId])

  const setWsHandle = useCallback((handle: WsRunHandle | null) => {
    wsHandleRef.current = handle
  }, [])

  const handleRun = useCallback((runId: string, cfg: PipelineConfigPayload) => {
    setSessions((prev) => {
      const next = new Map(prev)
      next.set(runId, emptySession(runId, cfg))
      return next
    })
    setActiveRunId(runId)
    setActivePage('monitor')
  }, [])

  const handleDone = useCallback((result: unknown) => {
    setActiveRunId((currentId) => {
      if (currentId) {
        setSessions((prev) => {
          const next = new Map(prev)
          const s = next.get(currentId)
          if (s) next.set(currentId, { ...s, result })
          return next
        })
      }
      return currentId
    })
    // Only navigate to results if we're currently on the monitor page
    setActivePage((page) => page === 'monitor' ? 'results' : page)
  }, [])

  const setWsEvents = useCallback((updater: WsEvent[] | ((prev: WsEvent[]) => WsEvent[])) => {
    setActiveRunId((currentId) => {
      if (currentId) {
        setSessions((prev) => {
          const next = new Map(prev)
          const s = next.get(currentId)
          if (s) {
            const newEvents = typeof updater === 'function' ? updater(s.wsEvents) : updater
            next.set(currentId, { ...s, wsEvents: newEvents })
          } 
          return next
        })
      }
      return currentId
    })
  }, [])

  const setIsConnected: React.Dispatch<React.SetStateAction<boolean | undefined>> = useCallback(
    (value: React.SetStateAction<boolean | undefined>) => {
      setActiveRunId((currentId) => {
        if (currentId) {
          setSessions((prev) => {
            const next = new Map(prev)
            const s = next.get(currentId)
            if (s) {
              const resolved = typeof value === 'function' ? value(s.isWsConnected) : value
              next.set(currentId, { ...s, isWsConnected: resolved })
            }
            return next
          })
        }
        return currentId
      })
    }, []
  )

  const [resumeFromRunId, setResumeFromRunId] = useState<string | null>(null)

  const handleResume = useCallback((oldRunId: string) => {
    // Navigate to Configure page with resume_from context.
    // The user can verify/update their config (esp. API keys) and click Run.
    setResumeFromRunId(oldRunId)
    setActivePage('configure')
  }, [])

  const handleSelectRun = useCallback((runId: string) => {
    setSessions((prev) => {
      if (prev.has(runId)) return prev
      const next = new Map(prev)
      next.set(runId, emptySession(runId))
      return next
    })
    setActiveRunId(runId)
    setActivePage('monitor')
  }, [])

  return (
    <div className="h-screen flex overflow-hidden bg-background text-foreground">
      <Sidebar
        activePage={activePage}
        onNavigate={setActivePage}
        currentRunId={activeRunId}
        onSelectRun={handleSelectRun}
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />
      <div className="flex-1 flex flex-col min-w-0">
        <TopBar onMenuToggle={() => setSidebarOpen((o) => !o)} />
        <main className="flex-1 overflow-y-auto p-3 sm:p-6">
          <div className="max-w-5xl mx-auto">
            <ErrorBoundary>
              {activePage === 'configure' && (
                <ConfigurePipeline
                  onRun={handleRun}
                  onExportConfig={() => {}}
                  setWsEvents={setWsEvents}
                  setIsConnected={setIsConnected}
                  onDone={handleDone}
                  setWsHandle={setWsHandle}
                  resumeFromRunId={resumeFromRunId}
                  onResumeConsumed={() => setResumeFromRunId(null)}
                />
              )}
              {activePage === 'monitor' && (
                <ErrorBoundary>
                  <MonitorProgress
                    runId={activeRunId}
                    config={activeSession?.config ?? null}
                    wsEvents={activeSession?.wsEvents ?? []}
                    isConnected={activeSession?.isWsConnected}
                    onDone={handleDone}
                    onStop={handleStop}
                    onResume={handleResume}
                  />
                </ErrorBoundary>
              )}
              {activePage === 'results' && (
                <ErrorBoundary>
                  <ResultsOutput
                    runId={activeRunId}
                    result={activeSession?.result ?? null}
                    onRefresh={() => setActivePage('monitor')}
                  />
                </ErrorBoundary>
              )}
              {activePage === 'inference' && <InferencePlayground />}
            </ErrorBoundary>
          </div>
        </main>
      </div>
    </div>
  )
}
