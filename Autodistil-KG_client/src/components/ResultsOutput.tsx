import { useState, useEffect } from 'react'
import { FolderInput, RefreshCw, Download, ChevronDown, Upload, ExternalLink, Loader2 } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible'
import { downloadRunArtifact, uploadModelToHub, getRunStatus, type RunResultResponse } from '@/api/client'

interface ResultsOutputProps {
  runId: string | null
  result: unknown
  onRefresh: () => void
  stageFilter?: string
}

interface PerfStats {
  num_predictions?: number
  total_time_sec?: number
  avg_latency_sec?: number
  p50_latency_sec?: number
  p95_latency_sec?: number
  p99_latency_sec?: number
  min_latency_sec?: number
  max_latency_sec?: number
  total_input_tokens?: number
  total_output_tokens?: number
  avg_output_tokens?: number
  throughput_tokens_per_sec?: number
}

interface EvalReport {
  evalg_mode?: string
  num_samples?: number
  metrics_used?: string[]
  timestamp?: string
  systems?: Record<string, {
    label?: string
    kind?: string
    aggregate_metrics?: Record<string, number>
    performance?: PerfStats
  }>
  per_question?: Array<{
    index: number
    question: string
    reference: string
    predictions: Record<string, string>
    scores: Record<string, Record<string, number>>
    performance?: Record<string, { latency_sec?: number; output_tokens?: number; tokens_per_sec?: number }>
  }>
}

const METRIC_DISPLAY: Record<string, string> = {
  rouge1: 'ROUGE-1',
  rouge2: 'ROUGE-2',
  rougeL: 'ROUGE-L',
  rouge_avg: 'ROUGE avg',
  bleu1: 'BLEU-1',
  bleu2: 'BLEU-2',
  bleu4: 'BLEU-4',
  answer_relevancy: 'Answer Relevancy',
  faithfulness: 'Faithfulness',
  judge_accuracy: 'Accuracy (judge)',
  judge_completeness: 'Completeness (judge)',
  judge_relevance: 'Relevance (judge)',
  judge_avg: 'Judge avg',
  entity_grounding: 'Entity Grounding',
  claim_grounding: 'Claim Grounding',
  scope_grounding: 'Scope Grounding',
  grounding_avg: 'Grounding avg',
}

const REFERENCE_METRICS = new Set(['rouge1', 'rouge2', 'rougeL', 'rouge_avg', 'bleu1', 'bleu2', 'bleu4'])

function MetricsComparisonTable({ report }: { report: EvalReport }) {
  const systems = report.systems ?? {}
  const systemIds = Object.keys(systems)
  if (systemIds.length === 0) return null

  const allMetricNames = new Set<string>()
  for (const sys of Object.values(systems)) {
    for (const key of Object.keys(sys.aggregate_metrics ?? {})) {
      if (!key.endsWith('_fail_count') && !key.endsWith('_failed')) {
        allMetricNames.add(key)
      }
    }
  }

  // Sort: reference-based first, then LLM-judge; alphabetically within each group
  const refMetrics = Array.from(allMetricNames).filter((m) => REFERENCE_METRICS.has(m)).sort()
  const llmMetrics = Array.from(allMetricNames).filter((m) => !REFERENCE_METRICS.has(m)).sort()

  const findBest = (metric: string): string | null => {
    let best: string | null = null
    let bestVal = -Infinity
    for (const sysId of systemIds) {
      const val = systems[sysId]?.aggregate_metrics?.[metric]
      if (val != null && val > bestVal) {
        bestVal = val
        best = sysId
      }
    }
    return best
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-max text-sm border-collapse">
        <thead>
          <tr className="border-b">
            <th className="text-left p-2 font-medium text-muted-foreground">Metric</th>
            {systemIds.map((id) => (
              <th key={id} className="text-right p-2 font-medium">
                <div>{systems[id]?.label ?? id}</div>
                <div className="text-xs font-normal text-muted-foreground">{systems[id]?.kind}</div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {refMetrics.length > 0 && llmMetrics.length > 0 && (
            <tr>
              <td colSpan={systemIds.length + 1} className="px-2 pt-3 pb-1 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
                Reference-based
              </td>
            </tr>
          )}
          {refMetrics.map((metric) => {
            const bestId = findBest(metric)
            return (
              <tr key={metric} className="border-b last:border-0">
                <td className="p-2 text-xs">{METRIC_DISPLAY[metric] ?? metric}</td>
                {systemIds.map((sysId) => {
                  const val = systems[sysId]?.aggregate_metrics?.[metric]
                  const isBest = sysId === bestId
                  return (
                    <td
                      key={sysId}
                      className={`text-right p-2 font-mono text-xs ${isBest ? 'font-bold text-green-600 dark:text-green-400' : ''}`}
                    >
                      {val != null ? val.toFixed(4) : '—'}
                    </td>
                  )
                })}
              </tr>
            )
          })}
          {refMetrics.length > 0 && llmMetrics.length > 0 && (
            <tr>
              <td colSpan={systemIds.length + 1} className="px-2 pt-3 pb-1 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
                LLM Judge
              </td>
            </tr>
          )}
          {llmMetrics.map((metric) => {
            const bestId = findBest(metric)
            return (
              <tr key={metric} className="border-b last:border-0">
                <td className="p-2 text-xs">{METRIC_DISPLAY[metric] ?? metric}</td>
                {systemIds.map((sysId) => {
                  const val = systems[sysId]?.aggregate_metrics?.[metric]
                  const isBest = sysId === bestId
                  return (
                    <td
                      key={sysId}
                      className={`text-right p-2 font-mono text-xs ${isBest ? 'font-bold text-green-600 dark:text-green-400' : ''}`}
                    >
                      {val != null ? val.toFixed(4) : '—'}
                    </td>
                  )
                })}
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function PerformanceTable({ report }: { report: EvalReport }) {
  const systems = report.systems ?? {}
  const systemIds = Object.keys(systems)
  if (systemIds.length === 0) return null

  // Check if any system has performance data
  const hasPerf = systemIds.some((id) => {
    const p = systems[id]?.performance
    return p && p.num_predictions && p.num_predictions > 0
  })
  if (!hasPerf) return null

  const perfRows: Array<{ label: string; key: keyof PerfStats; unit: string; format: (v: number) => string; lowerIsBetter?: boolean }> = [
    { label: 'Avg Latency', key: 'avg_latency_sec', unit: 's', format: (v) => v.toFixed(3) },
    { label: 'P50 Latency', key: 'p50_latency_sec', unit: 's', format: (v) => v.toFixed(3) },
    { label: 'P95 Latency', key: 'p95_latency_sec', unit: 's', format: (v) => v.toFixed(3), lowerIsBetter: true },
    { label: 'Min Latency', key: 'min_latency_sec', unit: 's', format: (v) => v.toFixed(3) },
    { label: 'Max Latency', key: 'max_latency_sec', unit: 's', format: (v) => v.toFixed(3) },
    { label: 'Throughput', key: 'throughput_tokens_per_sec', unit: ' tok/s', format: (v) => v.toFixed(1) },
    { label: 'Avg Output Tokens', key: 'avg_output_tokens', unit: '', format: (v) => v.toFixed(0) },
    { label: 'Total Time', key: 'total_time_sec', unit: 's', format: (v) => v.toFixed(1) },
    { label: 'Total Output Tokens', key: 'total_output_tokens', unit: '', format: (v) => v.toFixed(0) },
  ]

  const findBest = (key: keyof PerfStats, lowerIsBetter = false): string | null => {
    let best: string | null = null
    let bestVal = lowerIsBetter ? Infinity : -Infinity
    for (const sysId of systemIds) {
      const val = systems[sysId]?.performance?.[key]
      if (val == null) continue
      if (lowerIsBetter ? val < bestVal : val > bestVal) {
        bestVal = val
        best = sysId
      }
    }
    return best
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-max text-sm border-collapse">
        <thead>
          <tr className="border-b">
            <th className="text-left p-2 font-medium text-muted-foreground">Performance</th>
            {systemIds.map((id) => (
              <th key={id} className="text-right p-2 font-medium">
                {systems[id]?.label ?? id}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {perfRows.map(({ label, key, unit, format, lowerIsBetter }) => {
            const latencyMetric = key.includes('latency') || key === 'total_time_sec'
            const isLower = lowerIsBetter ?? latencyMetric
            const bestId = key === 'throughput_tokens_per_sec' ? findBest(key, false) : findBest(key, isLower)
            return (
              <tr key={key} className="border-b last:border-0">
                <td className="p-2 font-mono text-xs">{label}</td>
                {systemIds.map((sysId) => {
                  const val = systems[sysId]?.performance?.[key]
                  const isBest = sysId === bestId && systemIds.length > 1
                  return (
                    <td
                      key={sysId}
                      className={`text-right p-2 font-mono text-xs ${isBest ? 'font-bold text-green-600 dark:text-green-400' : ''}`}
                    >
                      {val != null ? `${format(val)}${unit}` : '—'}
                    </td>
                  )
                })}
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function PerQuestionDetails({ report }: { report: EvalReport }) {
  const questions = report.per_question ?? []
  const systemIds = Object.keys(report.systems ?? {})
  if (questions.length === 0) return null

  return (
    <div className="space-y-2">
      {questions.map((q, idx) => (
        <Collapsible key={idx}>
          <CollapsibleTrigger className="flex w-full items-center gap-2 py-2 px-3 text-left text-sm font-medium rounded-md border hover:bg-muted/50 [&[data-state=open]>svg]:rotate-180">
            <span className="text-xs text-muted-foreground font-mono w-8">#{q.index + 1}</span>
            <span className="flex-1 truncate">{q.question}</span>
            <ChevronDown className="h-4 w-4 ml-auto transition-transform shrink-0" />
          </CollapsibleTrigger>
          <CollapsibleContent className="px-3 pb-3">
            <div className="space-y-3 mt-2">
              <div>
                <span className="text-xs font-medium text-muted-foreground">Reference:</span>
                <p className="text-xs mt-1 p-2 rounded bg-muted/50 border whitespace-pre-wrap">{q.reference}</p>
              </div>
              {systemIds.map((sysId) => {
                const perf = q.performance?.[sysId]
                return (
                  <div key={sysId}>
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="text-xs font-medium">{report.systems?.[sysId]?.label ?? sysId}</span>
                      {q.scores?.[sysId] && (
                        <span className="text-xs text-muted-foreground">
                          [{Object.entries(q.scores[sysId]).filter(([k]) => !k.endsWith('_failed')).map(([k, v]) => `${k}: ${v != null ? (v as number).toFixed(2) : '—'}`).join(', ')}]
                        </span>
                      )}
                      {perf && perf.latency_sec != null && perf.latency_sec > 0 && (
                        <span className="text-xs text-primary font-mono">
                          {perf.latency_sec.toFixed(2)}s
                          {perf.tokens_per_sec ? ` | ${perf.tokens_per_sec.toFixed(0)} tok/s` : ''}
                          {perf.output_tokens ? ` | ${perf.output_tokens} tokens` : ''}
                        </span>
                      )}
                    </div>
                    <p className="text-xs mt-1 p-2 rounded bg-muted/30 border whitespace-pre-wrap">
                      {q.predictions?.[sysId] ?? '(no prediction)'}
                    </p>
                  </div>
                )
              })}
            </div>
          </CollapsibleContent>
        </Collapsible>
      ))}
    </div>
  )
}

function HuggingFaceUpload({ modelPath }: { modelPath: string }) {
  const [repoId, setRepoId] = useState('')
  const [hfToken, setHfToken] = useState('')
  const [isPrivate, setIsPrivate] = useState(true)
  const [mergeWeights, setMergeWeights] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [uploadResult, setUploadResult] = useState<{ repo_url: string } | null>(null)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [expanded, setExpanded] = useState(false)

  const handleUpload = async () => {
    if (!repoId.trim() || !hfToken.trim()) return
    setUploading(true)
    setUploadError(null)
    setUploadResult(null)
    try {
      const res = await uploadModelToHub({
        model_path: modelPath,
        repo_id: repoId.trim(),
        hf_token: hfToken.trim(),
        private: isPrivate,
        merge_before_upload: mergeWeights,
      })
      setUploadResult(res)
    } catch (e) {
      setUploadError(e instanceof Error ? e.message : 'Upload failed')
    } finally {
      setUploading(false)
    }
  }

  if (uploadResult) {
    return (
      <div className="rounded-lg border border-green-500/30 bg-green-500/5 p-4">
        <div className="flex items-center gap-2 mb-2">
          <Upload className="h-4 w-4 text-green-600" />
          <span className="text-sm font-medium text-green-700 dark:text-green-400">Uploaded to HuggingFace</span>
        </div>
        <a
          href={uploadResult.repo_url}
          target="_blank"
          rel="noopener noreferrer"
          className="text-sm text-primary underline flex items-center gap-1"
        >
          {uploadResult.repo_url}
          <ExternalLink className="h-3 w-3" />
        </a>
      </div>
    )
  }

  return (
    <div className="rounded-lg border border-border p-4 space-y-3">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-2 text-sm font-medium w-full text-left"
      >
        <Upload className="h-4 w-4" />
        Upload to HuggingFace
        <ChevronDown className={`h-4 w-4 ml-auto transition-transform ${expanded ? 'rotate-180' : ''}`} />
      </button>

      {expanded && (
        <div className="space-y-3 pt-1">
          <div>
            <label className="text-xs font-medium text-muted-foreground block mb-1">Repository ID</label>
            <input
              type="text"
              value={repoId}
              onChange={(e) => setRepoId(e.target.value)}
              placeholder="username/model-name"
              className="w-full rounded-md border border-input bg-background px-3 py-1.5 text-sm"
            />
          </div>
          <div>
            <label className="text-xs font-medium text-muted-foreground block mb-1">HuggingFace Token</label>
            <input
              type="password"
              value={hfToken}
              onChange={(e) => setHfToken(e.target.value)}
              placeholder="hf_..."
              className="w-full rounded-md border border-input bg-background px-3 py-1.5 text-sm"
            />
          </div>
          <div className="flex items-center gap-4">
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={isPrivate}
                onChange={(e) => setIsPrivate(e.target.checked)}
                className="rounded"
              />
              Private repo
            </label>
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={mergeWeights}
                onChange={(e) => setMergeWeights(e.target.checked)}
                className="rounded"
              />
              Merge LoRA weights
            </label>
          </div>
          {uploadError && (
            <p className="text-sm text-destructive" role="alert">{uploadError}</p>
          )}
          <Button
            onClick={handleUpload}
            disabled={uploading || !repoId.trim() || !hfToken.trim()}
            size="sm"
          >
            {uploading ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Uploading...
              </>
            ) : (
              <>
                <Upload className="h-4 w-4" />
                Upload Model
              </>
            )}
          </Button>
        </div>
      )}
    </div>
  )
}


export default function ResultsOutput({ runId, result, onRefresh, stageFilter }: ResultsOutputProps) {
  const [downloadError, setDownloadError] = useState<string | null>(null)
  const [fetchedResult, setFetchedResult] = useState<RunResultResponse | null>(null)
  const [fetching, setFetching] = useState(false)

  // When result isn't provided (e.g. viewing a historical run), fetch from the API.
  useEffect(() => {
    if (result != null || !runId) {
      setFetchedResult(null)
      return
    }
    setFetching(true)
    getRunStatus(runId)
      .then((r) => setFetchedResult(r))
      .catch(() => setFetchedResult(null))
      .finally(() => setFetching(false))
  }, [runId, result])

  const runResult = (result as RunResultResponse | null) ?? fetchedResult
  const hasResult = runResult != null
  const success = runResult?.success ?? false
  const results = runResult?.results ?? []
  const context = runResult?.context ?? {}
  const error = runResult?.error
  const chatmlPath = (!stageFilter || stageFilter === 'chatml_converter' || stageFilter === 'graph_traverser')
    ? context?.chatml_dataset_path as string | undefined : undefined
  const preparedPath = (!stageFilter || stageFilter === 'chatml_converter' || stageFilter === 'finetuner')
    ? context?.prepared_dataset_path as string | undefined : undefined
  const evalReportPath = (!stageFilter || stageFilter === 'evaluator')
    ? context?.eval_report_path as string | undefined : undefined
  const modelOutputPath = (!stageFilter || stageFilter === 'finetuner')
    ? context?.model_output_path as string | undefined : undefined

  // Extract the full eval report from the last result metadata
  const lastResult = results.length > 0 ? results[results.length - 1] : null
  const rawReport = (lastResult?.metadata as { raw_report?: EvalReport } | undefined)?.raw_report
  const evalMetrics = (lastResult?.metadata as { metrics?: Record<string, unknown> } | undefined)?.metrics

  const handleDownload = async (artifactKey: 'chatml' | 'prepared' | 'eval_report') => {
    if (!runId) return
    setDownloadError(null)
    try {
      await downloadRunArtifact(runId, artifactKey)
    } catch (e) {
      setDownloadError(e instanceof Error ? e.message : 'Download failed')
    }
  }

  return (
    <div className="space-y-6">
      <Card className="min-h-[320px] flex flex-col items-center justify-center">
        <CardContent className="p-8 flex flex-col items-center justify-center w-full">
          {!hasResult ? (
            <>
              {fetching ? (
                <>
                  <Loader2 className="h-12 w-12 text-muted-foreground animate-spin" aria-hidden />
                  <p className="text-sm text-muted-foreground mt-4">Loading results…</p>
                </>
              ) : (
                <>
                  <FolderInput className="h-14 w-14 text-muted-foreground" aria-hidden />
                  <h2 className="text-lg font-semibold mt-4">No Results Yet</h2>
                  <p className="text-sm text-muted-foreground mt-1 text-center max-w-sm">
                    Run the pipeline to generate outputs and view results here
                  </p>
                  <Button variant="outline" className="mt-6" onClick={onRefresh}>
                    <RefreshCw className="h-4 w-4" />
                    Refresh Page
                  </Button>
                </>
              )}
            </>
          ) : (
            <>
              <div className="w-full text-left">
                <div className="flex items-center gap-2 mb-4">
                  <Badge variant={success ? 'default' : 'destructive'}>
                    {success ? 'Completed' : 'Failed'}
                  </Badge>
                  {runId && (
                    <span className="text-xs text-muted-foreground font-mono">Run ID: {runId}</span>
                  )}
                </div>
                {error && (
                  <p className="text-sm text-destructive mb-4" role="alert">
                    {error}
                  </p>
                )}
                {results.length > 0 && (
                  <div className="mb-4">
                    <h3 className="text-sm font-medium mb-2">Stage results</h3>
                    <ul className="list-disc list-inside text-sm text-muted-foreground space-y-1">
                      {results.map((r, i) => (
                        <li key={i}>
                          Stage {i + 1}: {r.success ? 'Success' : 'Failed'}
                          {r.error && ` — ${r.error}`}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                {Object.keys(context).length > 0 && (
                  <details className="mt-2">
                    <summary className="text-sm font-medium cursor-pointer">
                      Downloads &amp; output paths
                    </summary>
                    <div className="mt-2 space-y-3">
                      {chatmlPath && runId && (
                        <div className="flex items-center gap-2 flex-wrap">
                          <span className="text-xs text-muted-foreground font-mono break-all flex-1 min-w-0">{chatmlPath}</span>
                          <Button type="button" variant="outline" size="sm" onClick={() => handleDownload('chatml')} className="shrink-0">
                            <Download className="h-3.5 w-3.5" />
                            Download JSONL
                          </Button>
                        </div>
                      )}
                      {preparedPath && runId && (
                        <div className="flex items-center gap-2 flex-wrap">
                          <span className="text-xs text-muted-foreground font-mono break-all flex-1 min-w-0">{preparedPath}</span>
                          <Button type="button" variant="outline" size="sm" onClick={() => handleDownload('prepared')} className="shrink-0">
                            <Download className="h-3.5 w-3.5" />
                            Download prepared
                          </Button>
                        </div>
                      )}
                      {evalReportPath && runId && (
                        <div className="flex items-center gap-2 flex-wrap">
                          <span className="text-xs text-muted-foreground font-mono break-all flex-1 min-w-0">{evalReportPath}</span>
                          <Button type="button" variant="outline" size="sm" onClick={() => handleDownload('eval_report')} className="shrink-0">
                            <Download className="h-3.5 w-3.5" />
                            Download eval report
                          </Button>
                        </div>
                      )}
                      <pre className="p-3 rounded-md border bg-muted/50 text-xs overflow-x-auto">
                        {JSON.stringify(context, null, 2)}
                      </pre>
                    </div>
                  </details>
                )}
                {downloadError && (
                  <p className="text-sm text-destructive mt-2" role="alert">
                    {downloadError}
                  </p>
                )}
                {modelOutputPath && (
                  <div className="mt-4">
                    <HuggingFaceUpload modelPath={modelOutputPath} />
                  </div>
                )}
              </div>
              <Button variant="outline" className="mt-6" onClick={onRefresh}>
                <RefreshCw className="h-4 w-4" />
                Refresh Page
              </Button>
            </>
          )}
        </CardContent>
      </Card>

      {(!stageFilter || stageFilter === 'evaluator') && rawReport && rawReport.systems && Object.keys(rawReport.systems).length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Evaluation Comparison</CardTitle>
            <div className="flex items-center gap-3 text-xs text-muted-foreground">
              <span>Mode: {rawReport.evalg_mode ?? 'unknown'}</span>
              <span>{rawReport.num_samples ?? 0} samples</span>
              {rawReport.metrics_used && <span>Metrics: {rawReport.metrics_used.join(', ')}</span>}
            </div>
          </CardHeader>
          <CardContent className="space-y-6">
            <MetricsComparisonTable report={rawReport} />
            <PerformanceTable report={rawReport} />
          </CardContent>
        </Card>
      )}

      {(!stageFilter || stageFilter === 'evaluator') && rawReport && (rawReport.per_question ?? []).length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Per-Question Breakdown</CardTitle>
            <p className="text-xs text-muted-foreground">
              Click a question to see predictions and scores from each system.
            </p>
          </CardHeader>
          <CardContent>
            <PerQuestionDetails report={rawReport} />
          </CardContent>
        </Card>
      )}

      {(!stageFilter || stageFilter === 'evaluator') && !rawReport && evalMetrics && (
        <Card>
          <CardHeader>
            <CardTitle>Evaluation Metrics</CardTitle>
          </CardHeader>
          <CardContent>
            <pre className="p-3 rounded-md border bg-muted/50 text-xs overflow-x-auto">
              {JSON.stringify(evalMetrics, null, 2)}
            </pre>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
