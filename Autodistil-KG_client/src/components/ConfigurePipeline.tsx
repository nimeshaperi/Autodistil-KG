/**
 * Pipeline configuration orchestrator.
 *
 * Renders the stage enable/disable grid, delegates to per-stage form
 * components, and provides run / import / export / log-level controls.
 */
import { useState, useCallback, useRef } from 'react'
import { Database, FileText, Cpu, BarChart3, Upload, Download, Play, FolderOpen } from 'lucide-react'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import type { PipelineConfigPayload, StageId, GraphTraverserConfig, ChatMLConverterConfig, FineTunerConfig, EvaluatorConfig, TraversalConfig, LLMConfig, RedisConfig, GraphRAGConfigRequest, LogLevel } from '@/types/config'
import { STAGE_ORDER, STAGE_LABELS, STAGE_DESCRIPTIONS } from '@/types/config'
import { runPipeline, runPipelineViaWebSocket } from '@/api/client'
import type { RunResultResponse, WsRunHandle } from '@/api/client'
import type { WsEvent } from '@/api/client'

import {
  GraphTraverserForm,
  ChatMLConverterForm,
  FineTunerForm,
  EvaluatorForm,
  StandaloneHint,
  ALL_MODELS,
  inferModelType,
} from './stages'

/* ------------------------------------------------------------------ */
/*  Stage icon map                                                      */
/* ------------------------------------------------------------------ */

const STAGE_ICONS: Record<StageId, typeof Database> = {
  graph_traverser: Database,
  chatml_converter: FileText,
  finetuner: Cpu,
  evaluator: BarChart3,
}

/* ------------------------------------------------------------------ */
/*  Defaults                                                            */
/* ------------------------------------------------------------------ */

const defaultTraversal: TraversalConfig = {
  strategy: 'bfs',
  max_nodes: 500,
  max_depth: 5,
  reasoning_depth: 2,
  max_paths_per_node: 15,
  path_batch_size: 5,
  num_workers: 1,
}

const defaultDataset = {
  seed_prompts: ['What can you tell me about this node? Describe: {properties}'],
  include_metadata: true,
}

const defaultRedis: RedisConfig = {
  host: 'localhost',
  port: 6379,
  db: 0,
  password: '',
  key_prefix: 'graph_traverser:',
}

const defaultLLM: LLMConfig = {
  provider: 'openai',
  api_key: '',
  model: 'gpt-4',
  base_url: '',
}

const defaultAlignment: GraphTraverserConfig['alignment'] = {
  quality_filter: false,
  quality_threshold: 0.7,
}

const defaultGraphTraverser: GraphTraverserConfig = {
  output_path: 'output/dataset.jsonl',
  traversal: defaultTraversal,
  dataset: defaultDataset,
  alignment: defaultAlignment,
  neo4j: { uri: 'bolt://localhost:7687', database: '', username: 'neo4j', password: '' },
  redis: defaultRedis,
  llm: defaultLLM,
}

const defaultChatML: ChatMLConverterConfig = {
  input_path: 'output/dataset.jsonl',
  output_path: 'output/prepared.jsonl',
  prepare_for_finetuning: true,
  chat_template: 'auto',
}

const defaultFineTuner: FineTunerConfig = {
  model_name: ALL_MODELS[0].value,
  model_type: ALL_MODELS[0].type,
  train_data_path: 'output/prepared.jsonl',
  output_dir: 'output/finetuned',
  max_seq_length: 2048,
  num_train_epochs: 1,
  per_device_train_batch_size: 2,
  learning_rate: 2e-4,
}

const defaultEvaluator: EvaluatorConfig = {
  eval_dataset_path: 'output/prepared.jsonl',
  output_report_path: 'output/eval_report.json',
  metrics: ['rouge', 'bleu', 'answer_relevancy'],
  evalg_mode: 'internal',
  graph_rag_enabled: false,
  use_vllm: false,
  vllm_gpu_memory_utilization: 0.9,
}

/* ------------------------------------------------------------------ */
/*  Props & helpers                                                     */
/* ------------------------------------------------------------------ */

interface ConfigurePipelineProps {
  onRun: (runId: string, config: PipelineConfigPayload) => void
  onExportConfig?: (config: PipelineConfigPayload) => void
  setWsEvents?: React.Dispatch<React.SetStateAction<WsEvent[]>>
  setIsConnected?: React.Dispatch<React.SetStateAction<boolean | undefined>>
  onDone?: (result: RunResultResponse) => void
  setWsHandle?: (handle: WsRunHandle | null) => void
  resumeFromRunId?: string | null
  onResumeConsumed?: () => void
}

const VALID_STAGES: StageId[] = ['graph_traverser', 'chatml_converter', 'finetuner', 'evaluator']

function parseImportedConfig(data: unknown): PipelineConfigPayload | null {
  if (!data || typeof data !== 'object') return null
  const o = data as Record<string, unknown>
  const run_stages = (Array.isArray(o.run_stages)
    ? (o.run_stages as string[]).filter((id) => VALID_STAGES.includes(id as StageId))
    : []) as StageId[]
  if (run_stages.length === 0) return null
  return {
    output_dir: typeof o.output_dir === 'string' ? o.output_dir : undefined,
    run_stages,
    graph_traverser: o.graph_traverser && typeof o.graph_traverser === 'object'
      ? (o.graph_traverser as PipelineConfigPayload['graph_traverser'])
      : undefined,
    chatml_converter: o.chatml_converter && typeof o.chatml_converter === 'object'
      ? (o.chatml_converter as ChatMLConverterConfig)
      : undefined,
    finetuner: o.finetuner && typeof o.finetuner === 'object'
      ? (o.finetuner as FineTunerConfig)
      : undefined,
    evaluator: o.evaluator && typeof o.evaluator === 'object'
      ? (o.evaluator as PipelineConfigPayload['evaluator'])
      : undefined,
  }
}

/* ------------------------------------------------------------------ */
/*  Main component                                                      */
/* ------------------------------------------------------------------ */

/** Top-level pipeline configuration view: stage selector, per-stage forms, and run controls. */
export default function ConfigurePipeline({ onRun, onExportConfig, setWsEvents, setIsConnected, onDone, setWsHandle, resumeFromRunId, onResumeConsumed }: ConfigurePipelineProps) {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [selectedStages, setSelectedStages] = useState<StageId[]>(['graph_traverser', 'chatml_converter'])
  const [outputDir, setOutputDir] = useState('output')
  const [graphTraverser, setGraphTraverser] = useState<GraphTraverserConfig>(defaultGraphTraverser)
  const [chatml, setChatml] = useState<ChatMLConverterConfig>(defaultChatML)
  const [finetuner, setFinetuner] = useState<FineTunerConfig>(defaultFineTuner)
  const [evaluator, setEvaluator] = useState<EvaluatorConfig>(defaultEvaluator)
  const [logLevel, setLogLevel] = useState<LogLevel>('INFO')
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [importError, setImportError] = useState<string | null>(null)

  const buildConfig = useCallback((): PipelineConfigPayload => {
    const run_stages = [...selectedStages].sort(
      (a, b) => STAGE_ORDER.indexOf(a) - STAGE_ORDER.indexOf(b)
    )
    const payload: PipelineConfigPayload = {
      output_dir: outputDir || 'output',
      run_stages,
      log_level: logLevel,
    }
    if (selectedStages.includes('graph_traverser')) {
      const csvToArray = (v?: string) => v ? v.split(',').map((s) => s.trim()).filter(Boolean) : undefined
      const t = graphTraverser.traversal
      const a = graphTraverser.alignment
      const hasAlignment = a && (a.domain_focus || a.domain_keywords || a.style_guide || a.target_audience || a.quality_filter || a.reference_texts_path)
      payload.graph_traverser = {
        ...graphTraverser,
        output_path: graphTraverser.output_path || 'output/dataset.jsonl',
        traversal: {
          strategy: t.strategy,
          max_nodes: t.max_nodes,
          max_depth: t.max_depth,
          reasoning_depth: t.reasoning_depth ?? 2,
          max_paths_per_node: t.max_paths_per_node ?? 15,
          path_batch_size: t.path_batch_size ?? 5,
          num_workers: t.num_workers ?? 1,
          relationship_types: csvToArray(t.relationship_types),
          node_labels: csvToArray(t.node_labels),
          seed_node_ids: csvToArray(t.seed_node_ids),
        },
        dataset: {
          ...graphTraverser.dataset,
          seed_prompts: graphTraverser.dataset.seed_prompts.filter(Boolean),
          output_format: graphTraverser.dataset.output_format ?? 'jsonl',
        },
        alignment: hasAlignment ? {
          domain_focus: a!.domain_focus || undefined,
          domain_keywords: a!.domain_keywords ? a!.domain_keywords.split(',').map(s => s.trim()).filter(Boolean) : undefined,
          style_guide: a!.style_guide || undefined,
          target_audience: a!.target_audience || undefined,
          max_answer_length: a!.max_answer_length || undefined,
          min_answer_length: a!.min_answer_length || undefined,
          quality_filter: a!.quality_filter,
          quality_threshold: a!.quality_filter ? a!.quality_threshold : undefined,
          reference_texts_path: a!.reference_texts_path || undefined,
        } : undefined,
      }
    }
    if (selectedStages.includes('chatml_converter')) {
      payload.chatml_converter = { ...chatml }
    }
    if (selectedStages.includes('finetuner')) {
      const dir = outputDir || 'output'
      const saveFolder = finetuner.save_as ? finetuner.save_as.trim() : 'finetuned'
      const ftOutputDir = finetuner.output_dir || `${dir}/${saveFolder}`
      payload.finetuner = {
        ...finetuner,
        output_dir: ftOutputDir,
        save_as: undefined,
        model_type: inferModelType(finetuner.model_name),
      }
    }
    if (selectedStages.includes('evaluator')) {
      const baseProvider = (!evaluator.base_model_provider || evaluator.base_model_provider === 'none') ? undefined : evaluator.base_model_provider
      // base_model_name is needed by both base model comparison AND vLLM serving
      const needsBaseName = baseProvider || evaluator.use_vllm
      payload.evaluator = {
        ...evaluator,
        base_model_provider: baseProvider,
        base_model_name: needsBaseName ? evaluator.base_model_name : undefined,
        base_model_api_key: baseProvider ? evaluator.base_model_api_key : undefined,
        base_model_base_url: baseProvider ? evaluator.base_model_base_url : undefined,
        graph_rag_config: (evaluator.graph_rag_enabled && evaluator.eval_graph_rag !== false) ? evaluator.graph_rag_config as GraphRAGConfigRequest : undefined,
        eval_distilled: evaluator.eval_distilled === false ? false : undefined,
        eval_base: evaluator.eval_base === false ? false : undefined,
        eval_graph_rag: evaluator.eval_graph_rag === false ? false : undefined,
        use_vllm: evaluator.use_vllm,
        vllm_gpu_memory_utilization: evaluator.use_vllm ? evaluator.vllm_gpu_memory_utilization : undefined,
        vllm_max_model_len: evaluator.use_vllm ? evaluator.vllm_max_model_len : undefined,
      }
    }
    return payload
  }, [selectedStages, graphTraverser, chatml, finetuner, evaluator])

  const handleRun = () => {
    setError(null)
    setRunning(true)
    const config = buildConfig()
    if (setWsEvents && onDone) {
      setWsEvents([])
      setIsConnected?.(undefined) // Reset connection status
      const handle = runPipelineViaWebSocket(config, {
        onRunId: (id) => onRun(id, config),
        onEvent: (e) => setWsEvents((prev) => [...prev, e]),
        onDone: (r) => {
          setIsConnected?.(false) // Connection closed after done
          setWsHandle?.(null)
          onDone(r)
          setRunning(false)
        },
        onError: (msg) => {
          setIsConnected?.(false)
          setWsHandle?.(null)
          setError(msg)
          setRunning(false)
        },
        onConnectionChange: (connected) => setIsConnected?.(connected),
      }, resumeFromRunId ?? undefined)
      setWsHandle?.(handle)
      onResumeConsumed?.()
      return
    }
    runPipeline(config, true)
      .then((res) => {
        if (res.run_id) {
          onRun(res.run_id, config)
        } else {
          setError(res.message || 'Run failed')
        }
      })
      .catch((e) => {
        setError(e instanceof Error ? e.message : 'Failed to run pipeline')
      })
      .finally(() => setRunning(false))
  }

  const handleExport = () => {
    setImportError(null)
    const config = buildConfig()
    const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = 'pipeline-config.json'
    a.click()
    URL.revokeObjectURL(a.href)
    onExportConfig?.(config)
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    e.target.value = ''
    if (!file) return
    const reader = new FileReader()
    reader.onload = () => {
      try {
        const text = reader.result as string
        const data = JSON.parse(text) as unknown
        const config = parseImportedConfig(data)
        if (!config) {
          setImportError('Invalid config: need run_stages array with at least one valid stage')
          return
        }
        setSelectedStages(config.run_stages)
        if (config.graph_traverser) {
          const gt = config.graph_traverser
          const def = defaultGraphTraverser
          setGraphTraverser({
            ...def,
            ...gt,
            traversal: {
              ...defaultTraversal,
              ...gt.traversal,
              // Imported JSON has string[] but form state uses CSV strings
              relationship_types: Array.isArray(gt.traversal?.relationship_types) ? gt.traversal.relationship_types.join(', ') : gt.traversal?.relationship_types,
              node_labels: Array.isArray(gt.traversal?.node_labels) ? gt.traversal.node_labels.join(', ') : gt.traversal?.node_labels,
              seed_node_ids: Array.isArray(gt.traversal?.seed_node_ids) ? gt.traversal.seed_node_ids.join(', ') : gt.traversal?.seed_node_ids,
            },
            dataset: { ...defaultDataset, ...gt.dataset },
            alignment: gt.alignment ? {
              ...gt.alignment,
              quality_filter: gt.alignment.quality_filter ?? false,
              quality_threshold: gt.alignment.quality_threshold ?? 0.7,
              domain_keywords: Array.isArray(gt.alignment.domain_keywords) ? gt.alignment.domain_keywords.join(', ') : gt.alignment.domain_keywords,
            } : undefined,
            neo4j: {
              uri: gt.neo4j?.uri ?? def.neo4j!.uri,
              database: gt.neo4j?.database ?? def.neo4j!.database,
              username: gt.neo4j?.username ?? def.neo4j!.username,
              password: gt.neo4j?.password ?? def.neo4j!.password,
            },
            redis: { ...defaultRedis, ...gt.redis },
            llm: { ...defaultLLM, ...gt.llm },
          })
        }
        if (config.chatml_converter) {
          setChatml({ ...defaultChatML, ...config.chatml_converter })
        }
        if (config.finetuner) {
          const modelName = config.finetuner.model_name || defaultFineTuner.model_name
          setFinetuner({
            ...defaultFineTuner,
            ...config.finetuner,
            model_name: modelName,
            model_type: inferModelType(modelName),
          })
        }
        if (config.evaluator) {
          setEvaluator({ ...defaultEvaluator, ...config.evaluator })
        }
        setError(null)
        setImportError(null)
      } catch (err) {
        setImportError(err instanceof Error ? err.message : 'Failed to parse config JSON')
      }
    }
    reader.readAsText(file, 'utf-8')
  }

  const handleImportClick = () => fileInputRef.current?.click()

  const handleOutputDirChange = (newDir: string) => {
    setOutputDir(newDir)
    const dir = newDir || 'output'
    setGraphTraverser((prev) => ({ ...prev, output_path: `${dir}/dataset.jsonl` }))
    setChatml((prev) => ({ ...prev, input_path: `${dir}/dataset.jsonl`, output_path: `${dir}/prepared.jsonl` }))
    setFinetuner((prev) => ({ ...prev, train_data_path: `${dir}/prepared.jsonl` }))
    setEvaluator((prev) => ({
      ...prev,
      eval_dataset_path: `${dir}/prepared.jsonl`,
      output_report_path: `${dir}/eval_report.json`,
    }))
  }

  const toggleStage = (id: StageId) => {
    setSelectedStages((prev) =>
      prev.includes(id) ? prev.filter((s) => s !== id) : [...prev, id]
    )
  }

  return (
    <div className="space-y-6">
      {/* Hidden file input for import */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".json,application/json"
        className="hidden"
        aria-hidden
        onChange={handleFileChange}
      />

      {/* Output Folder */}
      <div className="flex items-end gap-3 rounded-lg border p-3 bg-muted/30">
        <FolderOpen className="h-4 w-4 text-muted-foreground shrink-0 mb-1" />
        <div className="flex-1 space-y-1">
          <Label className="text-sm font-medium">Output Folder</Label>
          <Input
            value={outputDir}
            onChange={(e) => handleOutputDirChange(e.target.value)}
            placeholder="output"
            className="h-8 text-sm font-mono"
          />
        </div>
        <p className="text-[11px] text-muted-foreground mb-1 hidden sm:block">All stage outputs are saved under this folder.</p>
      </div>

      {/* Stage Selection */}
      <div className="space-y-3">
        <h2 className="text-lg font-semibold">Select Pipeline Stages</h2>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {STAGE_ORDER.map((id) => {
            const active = selectedStages.includes(id)
            const Icon = STAGE_ICONS[id]
            return (
              <button
                key={id}
                type="button"
                onClick={() => toggleStage(id)}
                className={`relative flex flex-col items-center gap-2 rounded-xl border-2 p-4 text-center transition-all ${
                  active
                    ? 'border-primary bg-primary/10 shadow-sm'
                    : 'border-border/50 bg-card hover:border-muted-foreground/40 hover:bg-accent/50'
                }`}
              >
                <div className={`flex h-10 w-10 items-center justify-center rounded-lg ${
                  active ? 'bg-primary text-primary-foreground' : 'bg-muted text-muted-foreground'
                }`}>
                  <Icon className="h-5 w-5" />
                </div>
                <div>
                  <p className="text-sm font-semibold">{STAGE_LABELS[id]}</p>
                  <p className="text-[11px] leading-tight text-muted-foreground mt-0.5">{STAGE_DESCRIPTIONS[id]}</p>
                </div>
                {active && (
                  <div className="absolute top-2 right-2 h-2 w-2 rounded-full bg-primary" />
                )}
              </button>
            )
          })}
        </div>
      </div>

      {/* Stage-specific configuration */}
      {selectedStages.includes('graph_traverser') && (
        <GraphTraverserForm value={graphTraverser} onChange={setGraphTraverser} />
      )}

      {selectedStages.includes('chatml_converter') && (
        <>
          {!selectedStages.includes('graph_traverser') && (
            <StandaloneHint stage="ChatML Converter" inputLabel="Input Path" inputField="input_path" />
          )}
          <ChatMLConverterForm value={chatml} onChange={setChatml} />
        </>
      )}

      {selectedStages.includes('finetuner') && (
        <>
          {!selectedStages.includes('chatml_converter') && (
            <StandaloneHint stage="FineTuner" inputLabel="Train Data Path" inputField="train_data_path" />
          )}
          <FineTunerForm value={finetuner} onChange={setFinetuner} />
        </>
      )}

      {selectedStages.includes('evaluator') && (
        <>
          {!selectedStages.includes('finetuner') && (
            <StandaloneHint stage="Evaluator" inputLabel="Model Path & Eval Dataset Path" inputField="model_path" />
          )}
          <EvaluatorForm
            value={evaluator}
            onChange={setEvaluator}
            inferredModelPath={selectedStages.includes('finetuner') ? (finetuner.output_dir || 'output/finetuned') : undefined}
            derivedEvalPath={
              selectedStages.includes('graph_traverser')
                ? (graphTraverser.output_path || 'output/dataset.jsonl').replace(/(\.\w+)$/, '_eval$1')
                : selectedStages.includes('chatml_converter')
                  ? (chatml.output_path || 'output/prepared.jsonl').replace(/(\.\w+)$/, '_eval$1')
                  : undefined
            }
          />
        </>
      )}

      {/* Pipeline Controls */}
      <Card>
        <CardHeader>
          <CardTitle>Pipeline Controls</CardTitle>
          <CardDescription>{selectedStages.length} stage{selectedStages.length !== 1 ? 's' : ''} selected</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap justify-end gap-2 items-center">
            <div className="flex items-center gap-1.5 mr-auto">
              <Label className="text-xs text-muted-foreground">Log Level</Label>
              <Select value={logLevel} onValueChange={(v) => setLogLevel(v as LogLevel)}>
                <SelectTrigger className="h-8 w-28 text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="DEBUG">Debug</SelectItem>
                  <SelectItem value="INFO">Info</SelectItem>
                  <SelectItem value="WARNING">Warning</SelectItem>
                  <SelectItem value="ERROR">Error</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <Button type="button" variant="outline" size="sm" onClick={handleImportClick}>
              <Upload className="h-4 w-4 mr-1" />
              Import
            </Button>
            <Button type="button" variant="outline" size="sm" onClick={handleExport}>
              <Download className="h-4 w-4 mr-1" />
              Export
            </Button>
            <Button type="button" size="sm" onClick={handleRun} disabled={running || selectedStages.length === 0}>
              <Play className="h-4 w-4 mr-1" />
              {running ? 'Running…' : resumeFromRunId ? 'Resume Pipeline' : 'Run Pipeline'}
            </Button>
            {resumeFromRunId && !running && (
              <span className="text-xs text-amber-600 dark:text-amber-400">
                Resuming from {resumeFromRunId.slice(0, 8)}…
              </span>
            )}
          </div>
          {importError && (
            <p className="text-sm text-destructive" role="alert">
              {importError}
            </p>
          )}
          {error && (
            <p className="text-sm text-destructive" role="alert">
              {error}
            </p>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
