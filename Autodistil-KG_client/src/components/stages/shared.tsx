/**
 * Shared helper components used across multiple stage-configuration forms.
 *
 * Includes collapsible sections, labelled inputs, LLM provider fields,
 * model selectors, file-upload path inputs, and standalone-mode hints.
 */
import { useState, useRef } from 'react'
import { ChevronDown, Globe, Upload } from 'lucide-react'
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
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible'
import type { LLMConfig, LLMProviderType } from '@/types/config'
import { uploadFile } from '@/api/client'

/* ------------------------------------------------------------------ */
/*  Constants                                                           */
/* ------------------------------------------------------------------ */

export const LLM_PROVIDERS = [
  { value: 'openai', label: 'OpenAI' },
  { value: 'openai_compatible', label: 'OpenAI Compatible' },
  { value: 'claude', label: 'Claude' },
  { value: 'gemini', label: 'Gemini' },
  { value: 'ollama', label: 'Ollama' },
  { value: 'vllm', label: 'vLLM' },
] as const

/** Models available for vLLM serving / base-model evaluation (HuggingFace IDs, Qwen + Gemma only, <=10B). */
export const LOCAL_SERVING_MODELS: { family: string; models: { value: string; label: string }[] }[] = [
  {
    family: 'Qwen 3.5',
    models: [
      { value: 'Qwen/Qwen3.5-0.8B', label: 'Qwen 3.5 0.8B' },
      { value: 'Qwen/Qwen3.5-2B', label: 'Qwen 3.5 2B' },
      { value: 'Qwen/Qwen3.5-4B', label: 'Qwen 3.5 4B' },
      { value: 'Qwen/Qwen3.5-9B', label: 'Qwen 3.5 9B' },
    ],
  },
  {
    family: 'Qwen 3',
    models: [
      { value: 'Qwen/Qwen3-0.6B', label: 'Qwen 3 0.6B' },
      { value: 'Qwen/Qwen3-1.7B', label: 'Qwen 3 1.7B' },
      { value: 'Qwen/Qwen3-4B', label: 'Qwen 3 4B' },
      { value: 'Qwen/Qwen3-8B', label: 'Qwen 3 8B' },
    ],
  },
  {
    family: 'Gemma 3',
    models: [
      { value: 'google/gemma-3-1b-it', label: 'Gemma 3 1B' },
      { value: 'google/gemma-3-4b-it', label: 'Gemma 3 4B' },
    ],
  },
]

/* ------------------------------------------------------------------ */
/*  ConfigCollapsible                                                    */
/* ------------------------------------------------------------------ */

/** Collapsible section wrapper with an optional icon and title. */
export function ConfigCollapsible({
  title,
  icon = null,
  children,
  defaultOpen = true,
}: {
  title: string
  icon?: React.ReactNode
  children: React.ReactNode
  defaultOpen?: boolean
}) {
  return (
    <Collapsible defaultOpen={defaultOpen} className="border-b border-border last:border-0">
      <CollapsibleTrigger className="flex w-full items-center gap-2 py-3 text-left font-medium hover:underline [&[data-state=open]>svg]:rotate-180">
        {icon}
        {title}
        <ChevronDown className="h-4 w-4 ml-auto transition-transform" />
      </CollapsibleTrigger>
      <CollapsibleContent className="pb-4">
        {children}
      </CollapsibleContent>
    </Collapsible>
  )
}

/* ------------------------------------------------------------------ */
/*  LabelInput                                                          */
/* ------------------------------------------------------------------ */

/** Simple labelled text/number input with optional help text. */
export function LabelInput({
  label,
  value,
  onChange,
  type = 'text',
  placeholder,
  help,
}: {
  label: string
  value: string
  onChange: (v: string) => void
  type?: string
  placeholder?: string
  help?: string
}) {
  return (
    <div className="space-y-2">
      <Label>{label}</Label>
      <Input type={type} value={value} onChange={(e) => onChange(e.target.value)} placeholder={placeholder} />
      {help && <p className="text-xs text-muted-foreground">{help}</p>}
    </div>
  )
}

/* ------------------------------------------------------------------ */
/*  LLMProviderFields                                                   */
/* ------------------------------------------------------------------ */

/** Provider-specific fields for configuring an LLM (API key, model, base URL, etc.). */
export function LLMProviderFields({
  value,
  onChange,
}: {
  value: LLMConfig
  onChange: (v: LLMConfig) => void
}) {
  const update = (part: Partial<LLMConfig>) => onChange({ ...value, ...part })
  return (
    <div className="space-y-3">
      <div className="space-y-2">
        <Label>Provider</Label>
        <Select value={value.provider ?? 'openai'} onValueChange={(v) => update({ provider: v as LLMProviderType })}>
          <SelectTrigger>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {LLM_PROVIDERS.map((p) => (
              <SelectItem key={p.value} value={p.value}>{p.label}</SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      {(value.provider === 'openai' || value.provider === 'openai_compatible' || value.provider === 'claude') && (
        <>
          <LabelInput label="API Key" type="password" value={value.api_key ?? ''} onChange={(v) => update({ api_key: v })} placeholder="sk-..." />
          <LabelInput label="Model" value={value.model ?? ''} onChange={(v) => update({ model: v })} placeholder={value.provider === 'claude' ? 'claude-3-opus-20240229' : 'gpt-4'} />
          {(value.provider === 'openai' || value.provider === 'openai_compatible') && (
            <LabelInput label="Base URL" value={value.base_url ?? ''} onChange={(v) => update({ base_url: v })} placeholder={value.provider === 'openai_compatible' ? 'https://openrouter.ai/api/v1' : 'https://api.openai.com/v1'} help={value.provider === 'openai_compatible' ? 'Required — endpoint for OpenAI-compatible API (e.g. OpenRouter, LiteLLM)' : 'Optional — override default OpenAI endpoint'} />
          )}
        </>
      )}
      {value.provider === 'gemini' && (
        <>
          <LabelInput label="Project ID" value={value.project_id ?? ''} onChange={(v) => update({ project_id: v })} placeholder="my-project" />
          <LabelInput label="Location" value={value.location ?? ''} onChange={(v) => update({ location: v })} placeholder="us-central1" />
          <LabelInput label="Model" value={value.model ?? ''} onChange={(v) => update({ model: v })} placeholder="gemini-pro" />
          <LabelInput label="Credentials path (optional)" value={value.credentials_path ?? ''} onChange={(v) => update({ credentials_path: v })} placeholder="/path/to/key.json" />
        </>
      )}
      {(value.provider === 'ollama' || value.provider === 'vllm') && (
        <>
          <LabelInput label="Base URL" value={value.base_url ?? ''} onChange={(v) => update({ base_url: v })} placeholder={value.provider === 'ollama' ? 'http://localhost:11434' : 'http://localhost:8000'} />
          <LabelInput label="Model" value={value.model ?? ''} onChange={(v) => update({ model: v })} placeholder={value.provider === 'ollama' ? 'llama2' : ''} />
        </>
      )}
    </div>
  )
}

/* ------------------------------------------------------------------ */
/*  ModelSelector                                                        */
/* ------------------------------------------------------------------ */

/** Reusable dropdown for selecting a local model (Qwen/Gemma, <=10B). */
export function ModelSelector({
  value,
  onChange,
  models = LOCAL_SERVING_MODELS,
  label = 'Model',
  placeholder = 'Select a model',
  help,
}: {
  value: string
  onChange: (v: string) => void
  models?: typeof LOCAL_SERVING_MODELS
  label?: string
  placeholder?: string
  help?: string
}) {
  return (
    <div className="space-y-1.5">
      <Label>{label}</Label>
      <Select value={value} onValueChange={onChange}>
        <SelectTrigger>
          <SelectValue placeholder={placeholder} />
        </SelectTrigger>
        <SelectContent>
          {models.map((family) => (
            <div key={family.family}>
              <div className="px-2 py-1.5 text-xs font-semibold text-muted-foreground">{family.family}</div>
              {family.models.map((m) => (
                <SelectItem key={m.value} value={m.value}>{m.label}</SelectItem>
              ))}
            </div>
          ))}
        </SelectContent>
      </Select>
      {help && <p className="text-xs text-muted-foreground">{help}</p>}
    </div>
  )
}

/* ------------------------------------------------------------------ */
/*  PathInputWithUpload                                                  */
/* ------------------------------------------------------------------ */

/** Text input for a file path combined with an upload button that POSTs the file to the API. */
export function PathInputWithUpload({
  label,
  value,
  onChange,
  placeholder,
  help,
  accept = '.jsonl,.json,.csv',
}: {
  label: string
  value: string
  onChange: (v: string) => void
  placeholder?: string
  help?: string
  accept?: string
}) {
  const fileRef = useRef<HTMLInputElement>(null)
  const [uploading, setUploading] = useState(false)

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setUploading(true)
    try {
      const result = await uploadFile(file)
      onChange(result.path)
    } catch (err) {
      console.error('Upload failed:', err)
    } finally {
      setUploading(false)
      if (fileRef.current) fileRef.current.value = ''
    }
  }

  return (
    <div className="space-y-2">
      <Label>{label}</Label>
      <div className="flex gap-2">
        <Input
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          className="flex-1"
        />
        <Button
          type="button"
          variant="outline"
          size="icon"
          disabled={uploading}
          onClick={() => fileRef.current?.click()}
          title="Upload file"
        >
          <Upload className="h-4 w-4" />
        </Button>
        <input
          ref={fileRef}
          type="file"
          accept={accept}
          className="hidden"
          onChange={handleUpload}
        />
      </div>
      {help && <p className="text-xs text-muted-foreground">{help}</p>}
    </div>
  )
}

/* ------------------------------------------------------------------ */
/*  StandaloneHint                                                       */
/* ------------------------------------------------------------------ */

/** Amber banner shown when a stage runs without its predecessor, prompting the user to set input paths. */
export function StandaloneHint({ stage, inputLabel }: { stage: string; inputLabel: string; inputField?: string }) {
  return (
    <div className="rounded-lg border border-primary/30 bg-primary/5 dark:bg-primary/10 dark:border-primary/20 px-4 py-3 text-sm text-foreground">
      <span className="font-medium">Standalone mode:</span> {stage} is running without its predecessor.
      Make sure <span className="font-medium">{inputLabel}</span> is set to an existing file from a previous run.
    </div>
  )
}

/* Re-export the Globe icon so stage forms can use it without adding lucide-react as a direct dependency concern. */
export { Globe }
