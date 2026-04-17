/**
 * ChatML Converter stage configuration form.
 *
 * Provides input/output path fields, a prepare-for-finetuning toggle,
 * and an optional chat template override.
 */
import { FileText } from 'lucide-react'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import type { ChatMLConverterConfig } from '@/types/config'
import { LabelInput, PathInputWithUpload } from './shared'

/** Configuration card for the ChatML Converter pipeline stage. */
export function ChatMLConverterForm({
  value,
  onChange,
}: {
  value: ChatMLConverterConfig
  onChange: (v: ChatMLConverterConfig) => void
}) {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <FileText className="h-5 w-5 text-primary" />
          <div>
            <CardTitle>ChatML Converter</CardTitle>
            <CardDescription>Normalize and prepare ChatML datasets for fine-tuning</CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <PathInputWithUpload label="Input Path" value={value.input_path ?? ''} onChange={(v) => onChange({ ...value, input_path: v })} placeholder="output/dataset.jsonl" help="Path to ChatML JSONL file — type a path or upload" />
          <LabelInput label="Output Path" value={value.output_path ?? ''} onChange={(v) => onChange({ ...value, output_path: v })} placeholder="output/prepared.jsonl" help="Path for prepared dataset (e.g. FineTuner input)" />
        </div>
        <div className="flex items-center gap-2">
          <Switch
            id="prepare_finetuning"
            checked={value.prepare_for_finetuning}
            onCheckedChange={(checked) => onChange({ ...value, prepare_for_finetuning: checked })}
          />
          <Label htmlFor="prepare_finetuning" className="font-normal cursor-pointer">
            Prepare for Fine-tuning
          </Label>
        </div>
        <LabelInput label="Chat Template (optional)" value={value.chat_template ?? ''} onChange={(v) => onChange({ ...value, chat_template: v })} placeholder="auto" help="Specific chat template format (leave empty for auto-detection)" />
      </CardContent>
    </Card>
  )
}
