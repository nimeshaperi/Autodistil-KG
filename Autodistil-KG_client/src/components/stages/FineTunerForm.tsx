/**
 * FineTuner stage configuration form.
 *
 * Includes model family/variant selection grid, training data paths,
 * and hyperparameter fields (seq length, epochs, batch size, LR).
 */
import { Cpu } from 'lucide-react'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Label } from '@/components/ui/label'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import type { FineTunerConfig } from '@/types/config'
import { LabelInput, PathInputWithUpload } from './shared'

/* ------------------------------------------------------------------ */
/*  Constants                                                           */
/* ------------------------------------------------------------------ */

/** Supported model families for Unsloth finetuning. */
export const FINETUNE_MODELS: { family: string; type: string; models: { value: string; label: string }[] }[] = [
  {
    family: 'Gemma 4',
    type: 'gemma4',
    models: [
      { value: 'unsloth/gemma-4-E2B-it', label: 'Gemma 4 E2B (2.3B)' },
      { value: 'unsloth/gemma-4-E4B-it', label: 'Gemma 4 E4B (4.5B)' },
      { value: 'unsloth/gemma-4-31B-it', label: 'Gemma 4 31B' },
      { value: 'unsloth/gemma-4-26B-A4B-it', label: 'Gemma 4 26B-A4B (MoE)' },
    ],
  },
  {
    family: 'Gemma 3',
    type: 'gemma3',
    models: [
      { value: 'unsloth/gemma-3-270m-it', label: 'Gemma 3 270M' },
      { value: 'unsloth/gemma-3-1b-it', label: 'Gemma 3 1B' },
      { value: 'unsloth/gemma-3-4b-it', label: 'Gemma 3 4B' },
    ],
  },
  {
    family: 'Qwen 3',
    type: 'qwen3',
    models: [
      { value: 'unsloth/Qwen3-0.6B', label: 'Qwen 3 0.6B' },
      { value: 'unsloth/Qwen3-1.7B', label: 'Qwen 3 1.7B' },
      { value: 'unsloth/Qwen3-4B', label: 'Qwen 3 4B' },
      { value: 'unsloth/Qwen3-8B', label: 'Qwen 3 8B' },
    ],
  },
  {
    family: 'Qwen 3.5',
    type: 'qwen3_5',
    models: [
      { value: 'unsloth/Qwen3.5-0.8B', label: 'Qwen 3.5 0.8B' },
      { value: 'unsloth/Qwen3.5-2B', label: 'Qwen 3.5 2B' },
      { value: 'unsloth/Qwen3.5-4B', label: 'Qwen 3.5 4B' },
      { value: 'unsloth/Qwen3.5-9B', label: 'Qwen 3.5 9B' },
    ],
  },
  {
    family: 'Llama 3',
    type: 'llama3',
    models: [
      { value: 'unsloth/Llama-3.2-1B-Instruct', label: 'Llama 3.2 1B' },
      { value: 'unsloth/Llama-3.2-3B-Instruct', label: 'Llama 3.2 3B' },
    ],
  },
]

export const ALL_MODELS = FINETUNE_MODELS.flatMap((f) => f.models.map((m) => ({ ...m, type: f.type, family: f.family })))

/** Infer the model type string from a model name (for Unsloth config). */
export function inferModelType(modelName: string): string {
  const entry = ALL_MODELS.find((m) => m.value === modelName)
  if (entry) return entry.type
  const n = modelName.toLowerCase()
  if (n.includes('qwen3.5') || n.includes('qwen3_5')) return 'qwen3_5'
  if (n.includes('qwen3')) return 'qwen3'
  if (n.includes('qwen')) return 'qwen2'
  if (n.includes('gemma-4') || n.includes('gemma4')) return 'gemma4'
  if (n.includes('gemma')) return 'gemma3'
  if (n.includes('llama')) return 'llama3'
  return 'chatml'
}

/* ------------------------------------------------------------------ */
/*  Component                                                           */
/* ------------------------------------------------------------------ */

/** Configuration card for the FineTuner pipeline stage. */
export function FineTunerForm({
  value,
  onChange,
}: {
  value: FineTunerConfig
  onChange: (v: FineTunerConfig) => void
}) {
  const update = (part: Partial<FineTunerConfig>) => onChange({ ...value, ...part })
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <Cpu className="h-5 w-5 text-primary" />
          <div>
            <CardTitle>FineTuner Configuration</CardTitle>
            <CardDescription>
              Fine-tune with Unsloth — Gemma 4/3, Qwen 3/3.5, Llama 3 models supported.
            </CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <Label>Model</Label>
          <Select
            value={value.model_name}
            onValueChange={(v) => update({ model_name: v, model_type: inferModelType(v) })}
          >
            <SelectTrigger>
              <SelectValue placeholder="Select a model" />
            </SelectTrigger>
            <SelectContent>
              {FINETUNE_MODELS.map((family) => (
                <div key={family.family}>
                  <div className="px-2 py-1.5 text-xs font-semibold text-muted-foreground">{family.family}</div>
                  {family.models.map((m) => (
                    <SelectItem key={m.value} value={m.value}>
                      {m.label}
                    </SelectItem>
                  ))}
                </div>
              ))}
            </SelectContent>
          </Select>
          <p className="text-xs text-muted-foreground">
            Unsloth-optimized models for fine-tuning.
          </p>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <PathInputWithUpload
            label="Train Data Path"
            value={value.train_data_path ?? ''}
            onChange={(v) => update({ train_data_path: v })}
            placeholder="output/prepared.jsonl"
            help="JSONL with messages — type a path or upload"
          />
          <LabelInput
            label="Save As"
            value={value.save_as ?? ''}
            onChange={(v) => update({ save_as: v || undefined, output_dir: undefined })}
            placeholder="finetuned"
            help="Folder name for the saved adapter, e.g. 'biomedical_qwen3_v1'"
          />
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <LabelInput label="Max Seq Length" type="number" value={String(value.max_seq_length ?? 2048)} onChange={(v) => update({ max_seq_length: parseInt(v, 10) || 2048 })} />
          <LabelInput label="Epochs" type="number" value={String(value.num_train_epochs ?? 1)} onChange={(v) => update({ num_train_epochs: parseInt(v, 10) || 1 })} />
          <LabelInput label="Batch Size" type="number" value={String(value.per_device_train_batch_size ?? 2)} onChange={(v) => update({ per_device_train_batch_size: parseInt(v, 10) || 2 })} />
          <LabelInput label="Learning Rate" type="text" value={String(value.learning_rate ?? 2e-4)} onChange={(v) => update({ learning_rate: parseFloat(v) || 2e-4 })} />
        </div>
      </CardContent>
    </Card>
  )
}
