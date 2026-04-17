/**
 * Evaluator stage configuration form.
 *
 * Covers evaluation systems (finetuned model, base model, Graph RAG),
 * vLLM serving toggle, inference settings, scoring metrics with LLM judge,
 * and dataset / evaluation-mode settings.
 */
import { BarChart3, FileText, Cpu } from 'lucide-react'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import type { EvaluatorConfig, LLMProviderType, EvalMode } from '@/types/config'
import {
  ConfigCollapsible,
  LabelInput,
  LLMProviderFields,
  ModelSelector,
  PathInputWithUpload,
  LLM_PROVIDERS,
} from './shared'

/* ------------------------------------------------------------------ */
/*  Component                                                           */
/* ------------------------------------------------------------------ */

/** Configuration card for the Evaluator pipeline stage. */
export function EvaluatorForm({
  value,
  onChange,
  inferredModelPath,
  derivedEvalPath,
}: {
  value: EvaluatorConfig
  onChange: (v: EvaluatorConfig) => void
  inferredModelPath?: string
  /** Auto-derived eval dataset path from the graph traverser / chatml converter. */
  derivedEvalPath?: string
}) {
  const update = (part: Partial<EvaluatorConfig>) => onChange({ ...value, ...part })
  const toggleMetric = (metric: string) => {
    const current = value.metrics || []
    if (current.includes(metric)) {
      update({ metrics: current.filter((m) => m !== metric) })
    } else {
      update({ metrics: [...current, metric] })
    }
  }

  const distilledEnabled = value.eval_distilled !== false
  const hasFinetuned = distilledEnabled && !!(value.model_path || inferredModelPath)
  const baseEnabled = value.eval_base !== false
  const hasBaseModel = baseEnabled && !!(value.base_model_provider && value.base_model_provider !== 'none')
  const hasGraphRag = value.graph_rag_enabled && value.eval_graph_rag !== false
  const systemCount = (hasFinetuned ? 1 : 0) + (hasBaseModel ? 1 : 0) + (hasGraphRag ? 1 : 0)

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <BarChart3 className="h-5 w-5 text-primary" />
          <div>
            <CardTitle>Evaluator Configuration</CardTitle>
            <CardDescription>
              Select which systems to evaluate and configure scoring metrics. {systemCount} system{systemCount !== 1 ? 's' : ''} selected.
            </CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-2">

        {/* --- SYSTEMS TO EVALUATE --- */}
        <ConfigCollapsible title={`Evaluation Systems (${systemCount})`} defaultOpen>
          <div className="space-y-4">
            <p className="text-xs text-muted-foreground">
              Each enabled system generates predictions for every eval sample. Results are compared side-by-side in the report.
            </p>

            {/* vLLM banner */}
            <div className={`rounded-lg border px-3 py-2.5 space-y-2.5 ${value.use_vllm ? 'border-primary/30 bg-primary/5' : ''}`}>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Cpu className="h-4 w-4 text-muted-foreground" />
                  <div>
                    <span className="text-sm font-medium">vLLM Serving</span>
                    <p className="text-[11px] text-muted-foreground">
                      {value.use_vllm
                        ? 'Base model + LoRA adapter served on a single vLLM server'
                        : 'Off — models load/unload sequentially via Unsloth'}
                    </p>
                  </div>
                </div>
                <Switch checked={value.use_vllm} onCheckedChange={(v) => update({ use_vllm: v })} />
              </div>
              {value.use_vllm && (
                <ModelSelector
                  value={value.base_model_name ?? ''}
                  onChange={(v) => update({ base_model_name: v })}
                  label="Base Model"
                  help="The model vLLM will load. LoRA adapter from finetuner runs on top of this."
                />
              )}
            </div>

            {/* System 1: Finetuned */}
            <div className={`rounded-lg border p-3 space-y-3 ${distilledEnabled && (value.model_path || inferredModelPath) ? '' : 'opacity-60'}`}>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className={`h-2.5 w-2.5 rounded-full ${hasFinetuned ? 'bg-primary' : 'bg-muted-foreground/30'}`} />
                  <span className="text-sm font-medium">Finetuned Model</span>
                  {distilledEnabled && inferredModelPath && !value.model_path && (
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-primary/10 text-primary font-medium">Auto from finetuner</span>
                  )}
                  {distilledEnabled && !inferredModelPath && !value.model_path && (
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground font-medium">No model — will skip</span>
                  )}
                  {!distilledEnabled && (
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground font-medium">Disabled</span>
                  )}
                </div>
                <Switch
                  checked={distilledEnabled}
                  onCheckedChange={(v) => update({ eval_distilled: v ? undefined : false })}
                />
              </div>
              <LabelInput
                label="Model Path"
                value={value.model_path ?? ''}
                onChange={(v) => update({ model_path: v })}
                placeholder={inferredModelPath ?? 'output/finetuned'}
                help={inferredModelPath ? `Will use finetuner output: ${inferredModelPath}` : 'Leave empty to skip finetuned model evaluation'}
              />
            </div>

            {/* System 2: Base Model */}
            <div className={`rounded-lg border p-3 space-y-3 ${hasBaseModel ? '' : 'opacity-60'}`}>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className={`h-2.5 w-2.5 rounded-full ${hasBaseModel ? 'bg-blue-500' : 'bg-muted-foreground/30'}`} />
                  <span className="text-sm font-medium">Base Model</span>
                  {!baseEnabled && (
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground font-medium">Disabled</span>
                  )}
                </div>
                <Switch
                  checked={baseEnabled && !!(value.base_model_provider && value.base_model_provider !== 'none')}
                  onCheckedChange={(v) => {
                    if (!v) {
                      update({ eval_base: false })
                    } else {
                      update({
                        eval_base: undefined,
                        base_model_provider: (value.base_model_provider && value.base_model_provider !== 'none')
                          ? value.base_model_provider
                          : 'openai_compatible',
                      })
                    }
                  }}
                />
              </div>
              {baseEnabled && !!(value.base_model_provider && value.base_model_provider !== 'none') && (
                <div className="space-y-3">
                  <div className="flex items-center gap-2">
                    <Label className="text-xs text-muted-foreground shrink-0">Type</Label>
                    <Select
                      value={value.base_model_provider === 'local' ? 'local' : 'api'}
                      onValueChange={(v) => update({
                        base_model_provider: v === 'local' ? 'local' : 'openai_compatible',
                      })}
                    >
                      <SelectTrigger className="h-7 text-xs">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="local">Local (same GPU)</SelectItem>
                        <SelectItem value="api">API / Remote</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  {value.base_model_provider === 'local' ? (
                    <ModelSelector
                      value={value.base_model_name ?? ''}
                      onChange={(v) => update({ base_model_name: v })}
                      label="Model"
                      help={value.use_vllm ? 'Will be served via vLLM alongside the LoRA adapter' : 'Loaded via Unsloth after finetuned model is unloaded'}
                    />
                  ) : (
                    <LLMProviderFields
                      value={{
                        provider: (value.base_model_provider as LLMProviderType) ?? 'openai_compatible',
                        model: value.base_model_name,
                        api_key: value.base_model_api_key,
                        base_url: value.base_model_base_url,
                      }}
                      onChange={(llm) => update({
                        base_model_provider: llm.provider as LLMProviderType,
                        base_model_name: llm.model,
                        base_model_api_key: llm.api_key,
                        base_model_base_url: llm.base_url,
                      })}
                    />
                  )}
                </div>
              )}
            </div>

            {/* System 3: Graph RAG */}
            <div className={`rounded-lg border p-3 space-y-3 ${hasGraphRag ? '' : 'opacity-60'}`}>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className={`h-2.5 w-2.5 rounded-full ${hasGraphRag ? 'bg-emerald-500' : 'bg-muted-foreground/30'}`} />
                  <span className="text-sm font-medium">Graph RAG</span>
                </div>
                <Switch
                  checked={value.graph_rag_enabled && value.eval_graph_rag !== false}
                  onCheckedChange={(checked) => update({ graph_rag_enabled: checked, eval_graph_rag: checked ? undefined : false })}
                />
              </div>
              {hasGraphRag && (
                <div className="space-y-4">
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    <LabelInput label="Neo4j URI" value={value.graph_rag_config?.neo4j_uri ?? ''} onChange={(v) => update({ graph_rag_config: { ...value.graph_rag_config, neo4j_uri: v } })} placeholder="bolt://localhost:7688" />
                    <LabelInput label="Neo4j User" value={value.graph_rag_config?.neo4j_user ?? ''} onChange={(v) => update({ graph_rag_config: { ...value.graph_rag_config, neo4j_user: v } })} placeholder="neo4j" />
                    <LabelInput label="Neo4j Password" type="password" value={value.graph_rag_config?.neo4j_password ?? ''} onChange={(v) => update({ graph_rag_config: { ...value.graph_rag_config, neo4j_password: v } })} placeholder="" />
                    <LabelInput label="Neo4j Database" value={value.graph_rag_config?.neo4j_database ?? ''} onChange={(v) => update({ graph_rag_config: { ...value.graph_rag_config, neo4j_database: v } })} placeholder="neo4j" />
                  </div>
                  {value.use_vllm ? (
                    <div className="rounded border border-primary/20 bg-primary/5 px-3 py-2">
                      <p className="text-xs font-medium text-primary">LLM: Using local vLLM server</p>
                      <p className="text-[11px] text-muted-foreground">
                        Graph RAG will use the same vLLM-hosted model ({value.base_model_name || 'base model'}) for entity extraction and answer synthesis.
                      </p>
                    </div>
                  ) : (
                    <div className="space-y-1">
                      <Label className="text-sm font-medium">LLM</Label>
                      <LLMProviderFields
                        value={{
                          provider: value.graph_rag_config?.llm_provider ?? 'openai',
                          model: value.graph_rag_config?.llm_model,
                          api_key: value.graph_rag_config?.llm_api_key,
                          base_url: value.graph_rag_config?.llm_base_url,
                        }}
                        onChange={(llm) => update({ graph_rag_config: { ...value.graph_rag_config, llm_provider: llm.provider as LLMProviderType, llm_model: llm.model, llm_api_key: llm.api_key, llm_base_url: llm.base_url } })}
                      />
                    </div>
                  )}
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    <LabelInput label="Embedding API Key" type="password" value={value.graph_rag_config?.embedding_api_key ?? ''} onChange={(v) => update({ graph_rag_config: { ...value.graph_rag_config, embedding_api_key: v } })} placeholder="(defaults to LLM key)" />
                    <LabelInput label="Embedding Model" value={value.graph_rag_config?.embedding_model ?? ''} onChange={(v) => update({ graph_rag_config: { ...value.graph_rag_config, embedding_model: v } })} placeholder="text-embedding-3-small" />
                    <LabelInput label="Embedding Base URL" value={value.graph_rag_config?.embedding_base_url ?? ''} onChange={(v) => update({ graph_rag_config: { ...value.graph_rag_config, embedding_base_url: v } })} placeholder="(defaults to OpenAI)" help="Custom endpoint for embeddings" />
                  </div>
                </div>
              )}
            </div>
          </div>
        </ConfigCollapsible>

        {/* --- INFERENCE SETTINGS --- */}
        <ConfigCollapsible title="Inference Settings" icon={<Cpu className="h-5 w-5 text-muted-foreground" />} defaultOpen={value.use_vllm}>
          <div className="space-y-3">
            {value.use_vllm && (
              <div className="space-y-3">
                <p className="text-xs text-muted-foreground">
                  vLLM will auto-start with the base model + LoRA adapter before evaluation and stop after.
                </p>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  <LabelInput
                    label="GPU Memory Utilization"
                    value={String(value.vllm_gpu_memory_utilization ?? 0.9)}
                    onChange={(v) => update({ vllm_gpu_memory_utilization: parseFloat(v) || 0.9 })}
                    placeholder="0.9"
                    help="Fraction of GPU memory (0.0-1.0)"
                  />
                  <LabelInput
                    label="Max Model Length"
                    value={value.vllm_max_model_len != null ? String(value.vllm_max_model_len) : ''}
                    onChange={(v) => update({ vllm_max_model_len: v ? parseInt(v) : undefined })}
                    placeholder="Auto-detect"
                    help="Override context length (leave empty for auto)"
                  />
                </div>
              </div>
            )}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <LabelInput
                label="Max New Tokens"
                type="number"
                value={String(value.max_new_tokens ?? 2048)}
                onChange={(v) => update({ max_new_tokens: Math.max(1, parseInt(v, 10) || 2048) })}
                placeholder="2048"
                help="Max tokens per response"
              />
              <LabelInput
                label="Max Seq Length"
                type="number"
                value={String(value.max_seq_length ?? 4096)}
                onChange={(v) => update({ max_seq_length: Math.max(1, parseInt(v, 10) || 4096) })}
                placeholder="4096"
                help="Context window size"
              />
            </div>
          </div>
        </ConfigCollapsible>

        {/* --- SCORING --- */}
        <ConfigCollapsible title="Scoring Metrics" defaultOpen>
          <div className="space-y-4">
            {/* Reference-based (no LLM required) */}
            <div className="space-y-1.5">
              <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">Reference-based (no LLM judge)</p>
              <div className="flex flex-wrap gap-3">
                {([
                  { id: 'rouge', label: 'ROUGE (1 / 2 / L)' },
                  { id: 'bleu', label: 'BLEU (1 / 2 / 4)' },
                ] as const).map(({ id, label }) => (
                  <div key={id} className="flex items-center gap-2">
                    <Switch
                      id={`metric_${id}`}
                      checked={(value.metrics || []).includes(id)}
                      onCheckedChange={() => toggleMetric(id)}
                    />
                    <Label htmlFor={`metric_${id}`} className="font-normal cursor-pointer text-sm">
                      {label}
                    </Label>
                  </div>
                ))}
              </div>
            </div>

            {/* LLM-judge metrics */}
            <div className="space-y-1.5">
              <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">LLM judge metrics</p>
              <div className="flex flex-wrap gap-3">
                {([
                  { id: 'answer_relevancy', label: 'Answer Relevancy' },
                  { id: 'correctness', label: 'Correctness (G-Eval)' },
                  { id: 'faithfulness', label: 'Faithfulness' },
                  { id: 'hallucination', label: 'Hallucination' },
                  { id: 'grounding', label: 'Grounding (G-Eval)' },
                ] as const).map(({ id, label }) => (
                  <div key={id} className="flex items-center gap-2">
                    <Switch
                      id={`metric_${id}`}
                      checked={(value.metrics || []).includes(id)}
                      onCheckedChange={() => toggleMetric(id)}
                    />
                    <Label htmlFor={`metric_${id}`} className="font-normal cursor-pointer text-sm">
                      {label}
                    </Label>
                  </div>
                ))}
              </div>
            </div>

            {/* LLM Judge config — only shown when at least one LLM-judge metric is active */}
            {(value.metrics || []).some((m) => !['rouge', 'bleu'].includes(m)) && (
              <div className="space-y-2">
                <Label className="text-sm font-medium">LLM Judge</Label>
                <p className="text-xs text-muted-foreground">The LLM used to score predictions against references.</p>
                <LLMProviderFields
                  value={{
                    provider: value.judge_provider ?? 'openai',
                    model: value.judge_model,
                    api_key: value.judge_api_key,
                    base_url: value.judge_base_url,
                  }}
                  onChange={(llm) => update({
                    judge_provider: llm.provider as LLMProviderType,
                    judge_model: llm.model,
                    judge_api_key: llm.api_key,
                    judge_base_url: llm.base_url,
                  })}
                />
                <LabelInput
                  label="Max Tokens (judge response)"
                  type="number"
                  value={value.judge_max_tokens != null ? String(value.judge_max_tokens) : ''}
                  onChange={(v) => update({ judge_max_tokens: v ? parseInt(v, 10) : undefined })}
                  placeholder="4096"
                  help="Maximum tokens the judge LLM may output per scoring call. Increase if verdicts are truncated."
                />
              </div>
            )}
          </div>
        </ConfigCollapsible>

        {/* --- DATASET & SETTINGS --- */}
        <ConfigCollapsible title="Dataset & Settings" icon={<FileText className="h-5 w-5 text-muted-foreground" />} defaultOpen={false}>
          <div className="space-y-4">
            <div className="space-y-2">
              <Label>Evaluation Mode</Label>
              <Select value={value.evalg_mode} onValueChange={(v) => update({ evalg_mode: v as EvalMode })}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="internal">Internal (in-process)</SelectItem>
                  <SelectItem value="cli">CLI (external command)</SelectItem>
                  <SelectItem value="noop">No-op (stub report)</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="space-y-1">
                <PathInputWithUpload
                  label="Eval Dataset Path"
                  value={value.eval_dataset_path ?? ''}
                  onChange={(v) => update({ eval_dataset_path: v })}
                  placeholder={derivedEvalPath || 'output/prepared.jsonl'}
                  help="JSONL with messages — type a path or upload"
                />
                {derivedEvalPath && !value.eval_dataset_path && (
                  <p className="text-[11px] text-muted-foreground">
                    Auto-resolved: <span className="font-mono">{derivedEvalPath}</span>
                  </p>
                )}
                {derivedEvalPath && value.eval_dataset_path && value.eval_dataset_path !== derivedEvalPath && (
                  <p className="text-[11px] text-amber-600 dark:text-amber-400">
                    Overriding auto-resolved <span className="font-mono">{derivedEvalPath}</span>
                  </p>
                )}
              </div>
              <LabelInput
                label="Output Report Path"
                value={value.output_report_path ?? ''}
                onChange={(v) => update({ output_report_path: v })}
                placeholder="output/eval_report.json"
              />
            </div>
            <LabelInput
              label="Max Eval Samples"
              type="number"
              value={value.max_eval_samples != null ? String(value.max_eval_samples) : ''}
              onChange={(v) => update({ max_eval_samples: v ? parseInt(v, 10) || undefined : undefined })}
              placeholder="All samples"
              help="Limit the number of samples to evaluate"
            />
          </div>
        </ConfigCollapsible>
      </CardContent>
    </Card>
  )
}
