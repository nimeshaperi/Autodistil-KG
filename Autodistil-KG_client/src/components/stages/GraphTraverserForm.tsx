/**
 * Graph Traverser stage configuration form.
 *
 * Covers Neo4j connection, Redis state storage, LLM provider,
 * traversal strategy and parameters, graph filters / seed nodes,
 * dataset generation settings, and alignment configuration.
 */
import { Database } from 'lucide-react'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Switch } from '@/components/ui/switch'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import type { GraphTraverserConfig, RedisConfig, LLMConfig, TraversalStrategy } from '@/types/config'
import {
  ConfigCollapsible,
  LabelInput,
  LLMProviderFields,
  Globe,
} from './shared'

/* ------------------------------------------------------------------ */
/*  Constants                                                           */
/* ------------------------------------------------------------------ */

const STRATEGY_OPTIONS = [
  { value: 'bfs', label: 'Breadth-First Search', description: 'Explore graph layer by layer from seed nodes.' },
  { value: 'dfs', label: 'Depth-First Search', description: 'Explore graph depth-first along each branch.' },
  { value: 'random', label: 'Random Walk', description: 'Randomly select neighbours at each step.' },
  { value: 'semantic', label: 'Semantic (LLM-guided)', description: 'LLM selects the most relevant neighbour based on context.' },
  { value: 'reasoning', label: 'Reasoning (multi-hop)', description: 'Deep multi-hop reasoning with subgraph exploration.' },
]

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

/* ------------------------------------------------------------------ */
/*  Component                                                           */
/* ------------------------------------------------------------------ */

/** Full configuration card for the Graph Traverser pipeline stage. */
export function GraphTraverserForm({
  value,
  onChange,
}: {
  value: GraphTraverserConfig
  onChange: (v: GraphTraverserConfig) => void
}) {
  const update = (part: Partial<GraphTraverserConfig>) => onChange({ ...value, ...part })
  const updateTraversal = (t: Partial<GraphTraverserConfig['traversal']>) =>
    update({ traversal: { ...value.traversal, ...t } })
  const updateDataset = (d: Partial<GraphTraverserConfig['dataset']>) =>
    update({ dataset: { ...value.dataset, ...d } })
  const updateAlignment = (a: Partial<NonNullable<GraphTraverserConfig['alignment']>>) =>
    update({ alignment: { quality_filter: false, quality_threshold: 0.7, ...value.alignment, ...a } })

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <Database className="h-5 w-5 text-muted-foreground" />
          <div>
            <CardTitle>Graph Traverser</CardTitle>
            <CardDescription>Configure Neo4j connection, traversal strategy, and LLM provider.</CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-2">
        <ConfigCollapsible title="Neo4j Database" icon={<Database className="h-5 w-5 text-muted-foreground" />} defaultOpen>
          <div className="grid grid-cols-2 gap-3">
            <LabelInput label="URI" value={value.neo4j?.uri ?? ''} onChange={(uri) => update({ neo4j: { ...value.neo4j!, uri } })} placeholder="neo4j://localhost:7687" />
            <LabelInput label="Database" value={value.neo4j?.database ?? ''} onChange={(d) => update({ neo4j: { ...value.neo4j!, database: d } })} placeholder="neo4j" />
            <LabelInput label="Username" value={value.neo4j?.username ?? ''} onChange={(u) => update({ neo4j: { ...value.neo4j!, username: u } })} placeholder="neo4j" />
            <LabelInput label="Password" type="password" value={value.neo4j?.password ?? ''} onChange={(p) => update({ neo4j: { ...value.neo4j!, password: p } })} placeholder="••••••••" />
          </div>
        </ConfigCollapsible>

        <ConfigCollapsible title="Redis State Storage" icon={<Database className="h-5 w-5 text-muted-foreground" />} defaultOpen={false}>
          <div className="grid grid-cols-2 gap-3">
            <LabelInput label="Host" value={value.redis?.host ?? ''} onChange={(v) => update({ redis: { ...defaultRedis, ...value.redis, host: v } })} placeholder="localhost" />
            <LabelInput label="Port" type="number" value={String(value.redis?.port ?? 6379)} onChange={(v) => update({ redis: { ...defaultRedis, ...value.redis, port: v === '' ? 6379 : parseInt(v, 10) || 6379 } })} placeholder="6379" />
            <LabelInput label="DB" type="number" value={String(value.redis?.db ?? 0)} onChange={(v) => update({ redis: { ...defaultRedis, ...value.redis, db: v === '' ? 0 : Math.max(0, parseInt(v, 10) || 0) } })} placeholder="0" />
            <LabelInput label="Password" type="password" value={value.redis?.password ?? ''} onChange={(p) => update({ redis: { ...defaultRedis, ...value.redis, password: p } })} placeholder="optional" />
            <LabelInput label="Key prefix" value={value.redis?.key_prefix ?? ''} onChange={(v) => update({ redis: { ...defaultRedis, ...value.redis, key_prefix: v } })} placeholder="graph_traverser:" />
          </div>
        </ConfigCollapsible>

        <ConfigCollapsible title="LLM Provider" icon={<Globe className="h-5 w-5 text-muted-foreground" />} defaultOpen={false}>
          <LLMProviderFields
            value={value.llm ?? defaultLLM}
            onChange={(llm) => update({ llm })}
          />
        </ConfigCollapsible>

        <ConfigCollapsible title="Traversal Settings" defaultOpen>
          <div className="space-y-4">
            <div className="grid grid-cols-4 gap-3">
              <div className="space-y-2">
                <Label>Strategy</Label>
                <Select value={value.traversal.strategy} onValueChange={(v) => updateTraversal({ strategy: v as TraversalStrategy })}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {STRATEGY_OPTIONS.map((o) => (
                      <SelectItem key={o.value} value={o.value}>{o.label}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <p className="text-xs text-muted-foreground">
                  {STRATEGY_OPTIONS.find((o) => o.value === value.traversal.strategy)?.description}
                </p>
              </div>
              <LabelInput label="Max Nodes" type="number" value={String(value.traversal.max_nodes)} onChange={(v) => updateTraversal({ max_nodes: parseInt(v, 10) || 0 })} />
              <LabelInput label="Max Depth" type="number" value={String(value.traversal.max_depth)} onChange={(v) => updateTraversal({ max_depth: parseInt(v, 10) || 0 })} />
              <LabelInput
                label="Workers"
                type="number"
                value={String(value.traversal.num_workers ?? 1)}
                onChange={(v) => updateTraversal({ num_workers: Math.max(1, parseInt(v, 10) || 1) })}
                help="Parallel traversal workers (1 = sequential)"
              />
            </div>
            {value.traversal.strategy === 'reasoning' && (
              <div className="grid grid-cols-2 gap-3">
                <LabelInput
                  label="Reasoning Depth"
                  type="number"
                  value={String(value.traversal.reasoning_depth ?? 2)}
                  onChange={(v) => updateTraversal({ reasoning_depth: parseInt(v, 10) || 2 })}
                  help="Subgraph depth to explore around each node"
                />
                <LabelInput
                  label="Max Paths per Node"
                  type="number"
                  value={String(value.traversal.max_paths_per_node ?? 15)}
                  onChange={(v) => updateTraversal({ max_paths_per_node: parseInt(v, 10) || 15 })}
                  help="Maximum paths to reason over per node"
                />
                <LabelInput
                  label="Path Batch Size"
                  type="number"
                  value={String(value.traversal.path_batch_size ?? 5)}
                  onChange={(v) => updateTraversal({ path_batch_size: parseInt(v, 10) || 5 })}
                  help="Paths per LLM call (higher = fewer calls, faster)"
                />
              </div>
            )}
            {(value.traversal.strategy === 'semantic' || value.traversal.strategy === 'reasoning') && (
              <div className="rounded-lg border bg-muted/50 p-3">
                <p className="text-xs text-muted-foreground">
                  {value.traversal.strategy === 'semantic'
                    ? 'Semantic traversal uses the configured LLM to decide which neighbouring node is most relevant at each step.'
                    : 'Reasoning traversal performs multi-hop subgraph exploration and path analysis using the LLM to generate rich, contextual conversations.'}
                </p>
              </div>
            )}
          </div>
        </ConfigCollapsible>

        <ConfigCollapsible title="Graph Filters (optional)" defaultOpen={false}>
          <div className="space-y-3">
            <LabelInput
              label="Relationship Types"
              value={value.traversal.relationship_types ?? ''}
              onChange={(v) => updateTraversal({ relationship_types: v || undefined })}
              placeholder="HAS_PART, RELATED_TO"
              help="Comma-separated list of relationship types to follow (empty = all)"
            />
            <LabelInput
              label="Node Labels"
              value={value.traversal.node_labels ?? ''}
              onChange={(v) => updateTraversal({ node_labels: v || undefined })}
              placeholder="Person, Organization"
              help="Comma-separated list of node labels to include (empty = all)"
            />
            <LabelInput
              label="Seed Node IDs"
              value={value.traversal.seed_node_ids ?? ''}
              onChange={(v) => updateTraversal({ seed_node_ids: v || undefined })}
              placeholder="node_123, node_456"
              help="Comma-separated starting node IDs (empty = auto-select)"
            />
          </div>
        </ConfigCollapsible>

        <ConfigCollapsible title="Dataset Generation" defaultOpen>
          <div className="space-y-3">
            <div className="space-y-2">
              <Label>Seed Prompts (one per line)</Label>
              <Textarea
                value={(value.dataset.seed_prompts || []).join('\n')}
                onChange={(e) => updateDataset({ seed_prompts: e.target.value.split('\n').filter(Boolean) })}
                rows={3}
                placeholder="What can you tell me about this node? Describe: {properties}"
              />
            </div>
            <LabelInput label="Output Path" value={value.output_path ?? ''} onChange={(v) => update({ output_path: v })} placeholder="output/dataset.jsonl" />
            <div className="flex items-center gap-2">
              <Switch
                id="include_metadata"
                checked={value.dataset.include_metadata}
                onCheckedChange={(checked) => updateDataset({ include_metadata: checked })}
              />
              <Label htmlFor="include_metadata" className="font-normal cursor-pointer">
                Include Metadata
              </Label>
            </div>
            {/* Eval dataset controls */}
            <div className="space-y-3 rounded-lg border p-3">
              <p className="text-sm font-medium">Evaluation Dataset</p>
              <p className="text-xs text-muted-foreground">
                A subset of traversed nodes is held out for evaluation. Questions are optionally rephrased so the eval tests comprehension, not memorisation.
              </p>
              <LabelInput
                label="Eval Sample Ratio (0–1)"
                value={String(value.dataset.eval_sample_ratio ?? 0.2)}
                onChange={(v) => updateDataset({ eval_sample_ratio: parseFloat(v) || 0.2 })}
                placeholder="0.2"
              />
              <div className="flex items-center gap-2">
                <Switch
                  id="eval_rephrase"
                  checked={value.dataset.eval_rephrase ?? true}
                  onCheckedChange={(checked) => updateDataset({ eval_rephrase: checked })}
                />
                <Label htmlFor="eval_rephrase" className="font-normal cursor-pointer">
                  Rephrase eval questions (LLM rewrites questions to test comprehension)
                </Label>
              </div>
            </div>
          </div>
        </ConfigCollapsible>
        <ConfigCollapsible title="Alignment (optional)" defaultOpen={false}>
          <div className="space-y-4">
            <div className="rounded-lg border bg-muted/50 p-3">
              <p className="text-xs text-muted-foreground">
                Steer generated training data toward a specific domain, style, or quality bar. All fields are optional.
              </p>
            </div>
            {/* Domain alignment */}
            <div className="space-y-3">
              <p className="text-sm font-medium">Domain Focus</p>
              <LabelInput
                label="Domain"
                value={value.alignment?.domain_focus ?? ''}
                onChange={(v) => updateAlignment({ domain_focus: v || undefined })}
                placeholder="e.g. clinical pharmacology and drug interactions"
                help="Free-text description of the target domain. Injected into all generation prompts."
              />
              <LabelInput
                label="Keywords"
                value={value.alignment?.domain_keywords ?? ''}
                onChange={(v) => updateAlignment({ domain_keywords: v || undefined })}
                placeholder="pharmacology, drug, enzyme, receptor"
                help="Comma-separated keywords that bias node selection and path priority."
              />
            </div>
            {/* Style alignment */}
            <div className="space-y-3">
              <p className="text-sm font-medium">Style &amp; Audience</p>
              <LabelInput
                label="Target Audience"
                value={value.alignment?.target_audience ?? ''}
                onChange={(v) => updateAlignment({ target_audience: v || undefined })}
                placeholder="e.g. medical students, senior engineers"
                help="Included in the system message for generated training data."
              />
              <div className="space-y-2">
                <Label>Style Guide</Label>
                <Textarea
                  value={value.alignment?.style_guide ?? ''}
                  onChange={(e) => updateAlignment({ style_guide: e.target.value || undefined })}
                  rows={2}
                  placeholder="e.g. Use concise bullet points suitable for a clinical reference card"
                />
                <p className="text-xs text-muted-foreground">Prose instructions for answer tone and format.</p>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <LabelInput
                  label="Min Answer Length (words)"
                  type="number"
                  value={value.alignment?.min_answer_length != null ? String(value.alignment.min_answer_length) : ''}
                  onChange={(v) => updateAlignment({ min_answer_length: v ? parseInt(v, 10) || undefined : undefined })}
                  placeholder="optional"
                />
                <LabelInput
                  label="Max Answer Length (words)"
                  type="number"
                  value={value.alignment?.max_answer_length != null ? String(value.alignment.max_answer_length) : ''}
                  onChange={(v) => updateAlignment({ max_answer_length: v ? parseInt(v, 10) || undefined : undefined })}
                  placeholder="optional"
                />
              </div>
            </div>
            {/* Quality alignment */}
            <div className="space-y-3">
              <p className="text-sm font-medium">Quality Filter</p>
              <div className="flex items-center gap-2">
                <Switch
                  id="quality_filter"
                  checked={value.alignment?.quality_filter ?? false}
                  onCheckedChange={(checked) => updateAlignment({ quality_filter: checked })}
                />
                <Label htmlFor="quality_filter" className="font-normal cursor-pointer">
                  Enable quality gate — LLM scores each Q&amp;A pair and discards low-quality ones
                </Label>
              </div>
              {value.alignment?.quality_filter && (
                <LabelInput
                  label="Quality Threshold (0–1)"
                  type="number"
                  value={String(value.alignment?.quality_threshold ?? 0.7)}
                  onChange={(v) => updateAlignment({ quality_threshold: Math.max(0, Math.min(1, parseFloat(v) || 0.7)) })}
                  help="Minimum average score (relevance, groundedness, completeness) to keep a pair."
                />
              )}
            </div>
            {/* Reference alignment */}
            <div className="space-y-3">
              <p className="text-sm font-medium">Reference Grounding</p>
              <LabelInput
                label="Reference Texts Path"
                value={value.alignment?.reference_texts_path ?? ''}
                onChange={(v) => updateAlignment({ reference_texts_path: v || undefined })}
                placeholder="path/to/reference.txt or reference.jsonl"
                help="Path to plain-text or JSONL reference material. Relevant excerpts are injected into prompts for grounding."
              />
            </div>
          </div>
        </ConfigCollapsible>
      </CardContent>
    </Card>
  )
}
