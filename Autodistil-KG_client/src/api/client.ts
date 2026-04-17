/**
 * API client — strictly typed REST + WebSocket calls.
 */
import type {
  PipelineConfigPayload,
  PipelineRunResultResponse,
  InferenceLLMRequest,
  InferenceLLMResponse,
  InferenceGraphRAGRequest,
  InferenceGraphRAGResponse,
  InferenceFinetunedRequest,
  InferenceFinetunedResponse,
  AvailableModel,
  RegisteredModel,
  RegisterModelRequest,
} from '../types/config'

export type {
  InferenceLLMRequest,
  InferenceLLMResponse,
  InferenceGraphRAGRequest,
  InferenceGraphRAGResponse,
  InferenceFinetunedRequest,
  InferenceFinetunedResponse,
  AvailableModel,
  RegisteredModel,
  RegisterModelRequest,
}

// ===== Base URL =====

const getBaseUrl = (): string => {
  const env = import.meta.env?.VITE_API_URL
  if (env && typeof env === 'string' && env.trim() !== '') {
    return env.replace(/\/$/, '')
  }
  return '/api'
}

export const apiBase = getBaseUrl()

// ===== Pipeline endpoints =====

export interface RunResponse {
  run_id: string
  status: string
  message?: string
}

export async function runPipeline(config: PipelineConfigPayload, asyncRun = true): Promise<RunResponse> {
  const url = `${apiBase}/pipelines/run${asyncRun ? '?async=true' : ''}`
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error((err as { detail?: string }).detail || res.statusText)
  }
  return res.json()
}

export type RunResultResponse = PipelineRunResultResponse

export async function getRunStatus(runId: string): Promise<RunResultResponse> {
  const res = await fetch(`${apiBase}/pipelines/runs/${runId}`)
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error((body as { detail?: string }).detail || res.statusText)
  }
  return res.json()
}

export interface RunSummary {
  run_id: string
  status: string
  error?: string
  stages?: string[]
  resumable?: boolean
  checkpoint_conversations?: number | null
}

export async function listRuns(): Promise<RunSummary[]> {
  const res = await fetch(`${apiBase}/pipelines/runs`)
  if (!res.ok) throw new Error(res.statusText)
  const data = (await res.json()) as { runs: RunSummary[] }
  return data.runs
}

export interface RunEventsResponse {
  run_id: string
  events: WsEvent[]
  total: number
}

export async function getRunEvents(runId: string, since = 0): Promise<RunEventsResponse> {
  const res = await fetch(`${apiBase}/pipelines/runs/${runId}/events?since=${since}`)
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error((body as { detail?: string }).detail || res.statusText)
  }
  return res.json()
}

export async function stopRun(runId: string): Promise<{ run_id: string; message: string }> {
  const res = await fetch(`${apiBase}/pipelines/runs/${runId}/stop`, { method: 'POST' })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error((body as { detail?: string }).detail || res.statusText)
  }
  return res.json()
}

export async function deleteRun(runId: string): Promise<{ status: string; run_id: string }> {
  const res = await fetch(`${apiBase}/pipelines/runs/${runId}`, { method: 'DELETE' })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error((body as { detail?: string }).detail || res.statusText)
  }
  return res.json()
}

export async function getRunConfig(runId: string): Promise<PipelineConfigPayload> {
  const res = await fetch(`${apiBase}/pipelines/runs/${runId}/config`)
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error((body as { detail?: string }).detail || res.statusText)
  }
  return res.json()
}

export function getRunArtifactUrl(runId: string, artifactKey: 'chatml' | 'prepared' | 'eval_report'): string {
  return `${apiBase}/pipelines/runs/${runId}/artifacts/${artifactKey}`
}

export async function downloadRunArtifact(runId: string, artifactKey: 'chatml' | 'prepared' | 'eval_report'): Promise<void> {
  const url = getRunArtifactUrl(runId, artifactKey)
  const res = await fetch(url)
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error((body as { detail?: string }).detail || res.statusText)
  }
  const blob = await res.blob()
  const disposition = res.headers.get('Content-Disposition')
  const match = disposition?.match(/filename="?([^";]+)"?/)
  const filename = match?.[1]?.trim() ?? (artifactKey === 'chatml' ? 'dataset.jsonl' : artifactKey === 'prepared' ? 'prepared.jsonl' : 'eval_report.json')
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = filename
  a.click()
  URL.revokeObjectURL(a.href)
}

// ===== File uploads =====

export interface FileUploadResponse {
  path: string
}

export async function uploadFile(file: File): Promise<FileUploadResponse> {
  const formData = new FormData()
  formData.append('file', file)
  const res = await fetch(`${apiBase}/pipelines/uploads`, {
    method: 'POST',
    body: formData,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error((err as { detail?: string }).detail || res.statusText)
  }
  return res.json()
}

// ===== Health =====

export async function healthCheck(): Promise<{ status: string }> {
  const res = await fetch(`${apiBase}/health`)
  if (!res.ok) throw new Error(res.statusText)
  return res.json()
}

// ===== Inference endpoints =====

export async function inferenceLLM(body: InferenceLLMRequest): Promise<InferenceLLMResponse> {
  const res = await fetch(`${apiBase}/inference/llm`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error((err as { detail?: string }).detail || res.statusText)
  }
  return res.json()
}

export async function inferenceGraphRAG(body: InferenceGraphRAGRequest): Promise<InferenceGraphRAGResponse> {
  const res = await fetch(`${apiBase}/inference/graphrag`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error((err as { detail?: string }).detail || res.statusText)
  }
  return res.json()
}

export async function listModels(): Promise<AvailableModel[]> {
  const res = await fetch(`${apiBase}/inference/models`)
  if (!res.ok) throw new Error(res.statusText)
  const data = (await res.json()) as { models: AvailableModel[] }
  return data.models
}

export async function listRegisteredModels(): Promise<RegisteredModel[]> {
  const res = await fetch(`${apiBase}/inference/models/registered`)
  if (!res.ok) throw new Error(res.statusText)
  const data = (await res.json()) as { models: RegisteredModel[] }
  return data.models
}

export async function registerModel(body: RegisterModelRequest): Promise<RegisteredModel> {
  const res = await fetch(`${apiBase}/inference/models/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error((err as { detail?: string }).detail || res.statusText)
  }
  return res.json()
}

export async function unregisterModel(modelId: string): Promise<void> {
  const res = await fetch(`${apiBase}/inference/models/registered/${encodeURIComponent(modelId)}`, {
    method: 'DELETE',
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error((err as { detail?: string }).detail || res.statusText)
  }
}

export async function inferenceFinetuned(body: InferenceFinetunedRequest): Promise<InferenceFinetunedResponse> {
  const res = await fetch(`${apiBase}/inference/finetuned`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error((err as { detail?: string }).detail || res.statusText)
  }
  return res.json()
}

// ===== HuggingFace Hub upload =====

export interface UploadToHubRequest {
  model_path: string
  repo_id: string
  hf_token: string
  private?: boolean
  base_model?: string
  commit_message?: string
  merge_before_upload?: boolean
}

export interface UploadToHubResponse {
  status: string
  repo_url: string
  repo_id: string
  merged: boolean
  base_model?: string
}

export async function uploadModelToHub(body: UploadToHubRequest): Promise<UploadToHubResponse> {
  const res = await fetch(`${apiBase}/inference/models/upload-to-hub`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error((err as { detail?: string }).detail || res.statusText)
  }
  return res.json()
}

// ===== WebSocket types & client =====

export interface TraversalProgressEvent {
  event: 'traversal_progress'
  type: string
  node_id?: string
  labels?: string[]
  depth?: number
  visited?: number
  total?: number
  step?: string
  node_count?: number
  edge_count?: number
  path_count?: number
  center?: { id: string; labels: string[]; properties: Record<string, string> }
  nodes?: { id: string; labels: string[]; properties?: Record<string, string> }[]
  edges?: { source: string; target: string; type: string }[]
  path_index?: number
  total_paths?: number
  path_description?: string
  dataset_size?: number
  paths_analyzed?: number
  conversations?: number
  strategy?: string
  seed_nodes?: number
  max_nodes?: number
  max_depth?: number
  path_analyses_count?: number
  reasoning_depth?: number
  neighbors?: { id: string; labels: string[]; relationship_type?: string }[]
  status?: string
  // Alignment fields
  quality_scores?: Record<string, number>
  alignment?: { quality_filtered?: number; quality_threshold?: number; domain_focus?: string; has_reference_texts?: boolean }
}

export interface EvalProgressEvent {
  event: 'eval_progress'
  type: string
  system_id?: string
  label?: string
  kind?: string
  index?: number
  total?: number
  sample_index?: number
  total_samples?: number
  total_predictions?: number
  latency_sec?: number
  tokens_per_sec?: number
  scorers?: string[]
  scores?: Record<string, Record<string, number | null>>
  num_samples?: number
  active_systems?: string[]
  aggregate_metrics?: Record<string, Record<string, number>>
  status?: string
}

export interface FinetunerProgressEvent {
  event: 'finetuner_progress'
  type: string
  model_name?: string
  train_samples?: number
  eval_samples?: number
  step?: number
  total_steps?: number
  epoch?: number
  total_epochs?: number
  loss?: number
  learning_rate?: number
  grad_norm?: number
  train_loss?: number
  batch_size?: number
  output_dir?: string
}

export type WsEvent =
  | { event: 'run_start'; run_id: string }
  | { event: 'pipeline_start'; stages: string[] }
  | { event: 'stage_start'; stage: string }
  | { event: 'stage_end'; stage: string; success: boolean; error?: string; metadata?: Record<string, unknown> }
  | { event: 'done'; success: boolean; cancelled?: boolean; context?: Record<string, unknown>; results?: unknown[] }
  | { event: 'error'; message: string }
  | { event: 'log'; level: string; logger: string; message: string }
  | { event: 'stop_acknowledged'; run_id: string }
  | { event: 'stop_requested' }
  | TraversalProgressEvent
  | EvalProgressEvent
  | FinetunerProgressEvent

export interface WsRunCallbacks {
  onRunId: (runId: string) => void
  onEvent: (event: WsEvent) => void
  onDone: (result: RunResultResponse) => void
  onError: (message: string) => void
  onConnectionChange?: (connected: boolean) => void
}

export function getWebSocketUrl(): string {
  const base = apiBase || (typeof location !== 'undefined' ? `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}` : 'ws://localhost:8000')
  const wsPath = base.replace(/^http/, 'ws')
  return `${wsPath}/ws`
}

export interface WsRunHandle {
  close: () => void
  stop: () => void
}

export function runPipelineViaWebSocket(config: PipelineConfigPayload, callbacks: WsRunCallbacks, resumeFrom?: string): WsRunHandle {
  const wsUrl = getWebSocketUrl()
  const ws = new WebSocket(wsUrl)
  let closed = false
  let runId = ''

  const close = () => {
    if (closed) return
    closed = true
    try { ws.close() } catch { /* ignore */ }
  }

  const stop = () => {
    if (closed || !runId) return
    try { ws.send(JSON.stringify({ action: 'stop', run_id: runId })) } catch { /* ignore */ }
  }

  ws.onopen = () => {
    callbacks.onConnectionChange?.(true)
    ws.send(JSON.stringify({ action: 'run', config, ...(resumeFrom ? { resume_from: resumeFrom } : {}) }))
  }

  ws.onmessage = (ev) => {
    if (closed) return
    try {
      const payload = JSON.parse(ev.data as string) as WsEvent & { event: string }
      callbacks.onEvent(payload as WsEvent)
      switch (payload.event) {
        case 'run_start':
          runId = (payload as { event: 'run_start'; run_id: string }).run_id
          callbacks.onRunId(runId)
          break
        case 'done': {
          const d = payload as { event: 'done'; success: boolean; context?: Record<string, unknown>; results?: unknown[] }
          callbacks.onDone({
            run_id: runId,
            status: d.success ? 'completed' : 'failed',
            success: d.success,
            context: d.context,
            results: d.results as RunResultResponse['results'],
          })
          // Don't close WebSocket — keep alive for viewing results
          break
        }
        case 'error':
          callbacks.onError((payload as { event: 'error'; message: string }).message)
          // Don't close WebSocket — keep alive for retry/viewing
          break
      }
    } catch (e) {
      callbacks.onError(e instanceof Error ? e.message : 'Invalid message')
      close()
    }
  }

  ws.onerror = () => { if (!closed) { callbacks.onError('WebSocket error'); close() } }
  ws.onclose = () => { callbacks.onConnectionChange?.(false); if (!closed) close() }

  return { close, stop }
}
