import { useState, useRef, useEffect } from 'react'
import { Send, Bot, User, Loader2, Settings2, Database, Sparkles, ChevronDown } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
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
import { ScrollArea } from '@/components/ui/scroll-area'
import { Badge } from '@/components/ui/badge'
import {
  inferenceLLM,
  inferenceGraphRAG,
  type InferenceLLMResponse,
  type InferenceGraphRAGResponse,
} from '@/api/client'

type InferenceMode = 'llm' | 'graphrag'

interface Message {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  metadata?: {
    provider?: string
    model?: string
    source_nodes?: { text: string; score?: number }[]
    mode?: InferenceMode
  }
}

interface LLMSettings {
  provider: import('@/types/config').LLMProviderType
  model: string
  api_key: string
  base_url: string
  project_id: string
  location: string
  temperature: number
  max_tokens: string
  system_message: string
}

interface GraphRAGSettings {
  neo4j_uri: string
  neo4j_user: string
  neo4j_password: string
  neo4j_database: string
  llm_api_key: string
  llm_model: string
  embedding_api_key: string
  embedding_model: string
  retrievers: import('@/types/config').RetrieverType[]
  num_agents: number
  similarity_top_k: number
}

const DEFAULT_LLM: LLMSettings = {
  provider: 'openai',
  model: 'gpt-4',
  api_key: '',
  base_url: '',
  project_id: '',
  location: '',
  temperature: 0.7,
  max_tokens: '',
  system_message: '',
}

const DEFAULT_GRAPHRAG: GraphRAGSettings = {
  neo4j_uri: 'bolt://localhost:7687',
  neo4j_user: 'neo4j',
  neo4j_password: '',
  neo4j_database: 'neo4j',
  llm_api_key: '',
  llm_model: 'gpt-4',
  embedding_api_key: '',
  embedding_model: 'text-embedding-3-small',
  retrievers: ['vector', 'cypher', 'synonym'],
  num_agents: 1,
  similarity_top_k: 5,
}

const PROVIDERS = [
  { value: 'openai', label: 'OpenAI' },
  { value: 'claude', label: 'Claude (Anthropic)' },
  { value: 'gemini', label: 'Gemini (Google)' },
  { value: 'ollama', label: 'Ollama (local)' },
  { value: 'vllm', label: 'vLLM (local)' },
]

export default function ModelPlayground() {
  const [mode, setMode] = useState<InferenceMode>('llm')
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [llmSettings, setLlmSettings] = useState<LLMSettings>(DEFAULT_LLM)
  const [graphragSettings, setGraphragSettings] = useState<GraphRAGSettings>(DEFAULT_GRAPHRAG)
  const [settingsOpen, setSettingsOpen] = useState(true)
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages])

  const handleSubmit = async () => {
    const trimmed = input.trim()
    if (!trimmed || loading) return

    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      content: trimmed,
    }
    setMessages((prev) => [...prev, userMsg])
    setInput('')
    setError(null)
    setLoading(true)

    try {
      if (mode === 'graphrag') {
        const result: InferenceGraphRAGResponse = await inferenceGraphRAG({
          question: trimmed,
          neo4j_uri: graphragSettings.neo4j_uri,
          neo4j_user: graphragSettings.neo4j_user,
          neo4j_password: graphragSettings.neo4j_password,
          neo4j_database: graphragSettings.neo4j_database,
          llm_api_key: graphragSettings.llm_api_key,
          llm_model: graphragSettings.llm_model,
          embedding_api_key: graphragSettings.embedding_api_key || graphragSettings.llm_api_key,
          embedding_model: graphragSettings.embedding_model,
          retrievers: graphragSettings.retrievers,
          num_agents: graphragSettings.num_agents,
          similarity_top_k: graphragSettings.similarity_top_k,
        })
        const assistantMsg: Message = {
          id: crypto.randomUUID(),
          role: 'assistant',
          content: result.answer,
          metadata: {
            mode: 'graphrag',
            source_nodes: (result.source_nodes as { text: string; score?: number }[])?.slice(0, 5),
            ...result.metadata,
          },
        }
        setMessages((prev) => [...prev, assistantMsg])
      } else {
        // Build message list for LLM
        const chatMessages: { role: string; content: string }[] = []
        if (llmSettings.system_message.trim()) {
          chatMessages.push({ role: 'system', content: llmSettings.system_message })
        }
        // Include conversation history
        for (const m of messages) {
          if (m.role === 'user' || m.role === 'assistant') {
            chatMessages.push({ role: m.role, content: m.content })
          }
        }
        chatMessages.push({ role: 'user', content: trimmed })

        const result: InferenceLLMResponse = await inferenceLLM({
          provider: llmSettings.provider,
          model: llmSettings.model || undefined,
          api_key: llmSettings.api_key || undefined,
          base_url: llmSettings.base_url || undefined,
          project_id: llmSettings.project_id || undefined,
          location: llmSettings.location || undefined,
          messages: chatMessages,
          temperature: llmSettings.temperature,
          max_tokens: llmSettings.max_tokens ? parseInt(llmSettings.max_tokens) : undefined,
        })
        const assistantMsg: Message = {
          id: crypto.randomUUID(),
          role: 'assistant',
          content: result.response,
          metadata: {
            provider: result.provider,
            model: result.model ?? llmSettings.model,
            mode: 'llm',
          },
        }
        setMessages((prev) => [...prev, assistantMsg])
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Inference failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-4">
      {/* Mode selector */}
      <div className="flex items-center gap-2">
        <Button
          variant={mode === 'llm' ? 'default' : 'outline'}
          size="sm"
          onClick={() => setMode('llm')}
          className="gap-1.5"
        >
          <Sparkles className="h-3.5 w-3.5" />
          Base LLM
        </Button>
        <Button
          variant={mode === 'graphrag' ? 'default' : 'outline'}
          size="sm"
          onClick={() => setMode('graphrag')}
          className="gap-1.5"
        >
          <Database className="h-3.5 w-3.5" />
          Graph RAG
        </Button>
      </div>

      {/* Settings */}
      <Collapsible open={settingsOpen} onOpenChange={setSettingsOpen}>
        <CollapsibleTrigger className="flex w-full items-center gap-2 py-1.5 text-left text-sm font-medium text-muted-foreground hover:text-foreground [&[data-state=open]>svg:last-child]:rotate-180">
          <Settings2 className="h-4 w-4" />
          {mode === 'llm' ? 'LLM Settings' : 'Graph RAG Settings'}
          <ChevronDown className="h-4 w-4 ml-auto transition-transform" />
        </CollapsibleTrigger>
        <CollapsibleContent>
          {mode === 'llm' ? (
            <LLMSettingsPanel settings={llmSettings} onChange={setLlmSettings} />
          ) : (
            <GraphRAGSettingsPanel settings={graphragSettings} onChange={setGraphragSettings} />
          )}
        </CollapsibleContent>
      </Collapsible>

      {/* Chat area */}
      <Card className="flex flex-col" style={{ height: 'calc(100vh - 420px)', minHeight: '300px' }}>
        <CardHeader className="py-3 px-4 border-b shrink-0">
          <CardTitle className="text-sm flex items-center gap-2">
            {mode === 'llm' ? (
              <>
                <Sparkles className="h-4 w-4" />
                {llmSettings.provider} — {llmSettings.model || 'default'}
              </>
            ) : (
              <>
                <Database className="h-4 w-4" />
                Graph RAG — {graphragSettings.llm_model}
              </>
            )}
            {messages.length > 0 && (
              <Button
                variant="ghost"
                size="sm"
                className="ml-auto text-xs h-6"
                onClick={() => setMessages([])}
              >
                Clear
              </Button>
            )}
          </CardTitle>
        </CardHeader>
        <CardContent className="flex-1 p-0 overflow-hidden flex flex-col">
          <ScrollArea className="flex-1 p-4" ref={scrollRef}>
            {messages.length === 0 && !loading && (
              <div className="flex flex-col items-center justify-center h-full text-muted-foreground py-12">
                <Bot className="h-10 w-10 mb-3 opacity-50" />
                <p className="text-sm">
                  {mode === 'llm'
                    ? 'Send a message to prompt the model'
                    : 'Ask a question to query the knowledge graph'}
                </p>
              </div>
            )}
            <div className="space-y-4">
              {messages.map((msg) => (
                <div key={msg.id} className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : ''}`}>
                  {msg.role === 'assistant' && (
                    <div className="shrink-0 w-7 h-7 rounded-full bg-primary/10 flex items-center justify-center">
                      <Bot className="h-4 w-4 text-primary" />
                    </div>
                  )}
                  <div
                    className={`max-w-[80%] rounded-lg px-3 py-2 text-sm ${
                      msg.role === 'user'
                        ? 'bg-primary text-primary-foreground'
                        : 'bg-muted'
                    }`}
                  >
                    <p className="whitespace-pre-wrap">{msg.content}</p>
                    {msg.metadata?.mode === 'graphrag' && msg.metadata.source_nodes && msg.metadata.source_nodes.length > 0 && (
                      <Collapsible>
                        <CollapsibleTrigger className="text-xs text-muted-foreground mt-2 hover:underline">
                          {msg.metadata.source_nodes.length} source node(s)
                        </CollapsibleTrigger>
                        <CollapsibleContent>
                          <div className="mt-2 space-y-1.5">
                            {msg.metadata.source_nodes.map((node, i) => (
                              <div key={i} className="text-xs p-2 rounded bg-background/50 border">
                                <p className="truncate">{node.text}</p>
                                {node.score != null && (
                                  <Badge variant="secondary" className="mt-1 text-[10px]">
                                    score: {node.score.toFixed(3)}
                                  </Badge>
                                )}
                              </div>
                            ))}
                          </div>
                        </CollapsibleContent>
                      </Collapsible>
                    )}
                    {msg.metadata?.mode === 'llm' && msg.metadata.provider && (
                      <p className="text-[10px] text-muted-foreground mt-1.5 opacity-60">
                        {msg.metadata.provider} / {msg.metadata.model}
                      </p>
                    )}
                  </div>
                  {msg.role === 'user' && (
                    <div className="shrink-0 w-7 h-7 rounded-full bg-primary flex items-center justify-center">
                      <User className="h-4 w-4 text-primary-foreground" />
                    </div>
                  )}
                </div>
              ))}
              {loading && (
                <div className="flex gap-3">
                  <div className="shrink-0 w-7 h-7 rounded-full bg-primary/10 flex items-center justify-center">
                    <Bot className="h-4 w-4 text-primary" />
                  </div>
                  <div className="bg-muted rounded-lg px-3 py-2">
                    <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
                  </div>
                </div>
              )}
            </div>
          </ScrollArea>
          {error && (
            <div className="px-4 py-2 text-sm text-destructive border-t bg-destructive/5">
              {error}
            </div>
          )}
          <div className="p-3 border-t flex gap-2">
            <Input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  handleSubmit()
                }
              }}
              placeholder={mode === 'llm' ? 'Type a message...' : 'Ask a question about the knowledge graph...'}
              disabled={loading}
              className="flex-1"
            />
            <Button onClick={handleSubmit} disabled={loading || !input.trim()} size="icon">
              <Send className="h-4 w-4" />
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

function LLMSettingsPanel({
  settings,
  onChange,
}: {
  settings: LLMSettings
  onChange: (s: LLMSettings) => void
}) {
  const update = (patch: Partial<LLMSettings>) => onChange({ ...settings, ...patch })

  return (
    <Card className="mt-2">
      <CardContent className="p-4 grid grid-cols-2 gap-3">
        <div className="col-span-2 sm:col-span-1">
          <Label className="text-xs">Provider</Label>
          <Select value={settings.provider} onValueChange={(v) => update({ provider: v as import('@/types/config').LLMProviderType })}>
            <SelectTrigger className="h-8 text-xs mt-1">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {PROVIDERS.map((p) => (
                <SelectItem key={p.value} value={p.value} className="text-xs">
                  {p.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="col-span-2 sm:col-span-1">
          <Label className="text-xs">Model</Label>
          <Input
            className="h-8 text-xs mt-1"
            value={settings.model}
            onChange={(e) => update({ model: e.target.value })}
            placeholder="e.g. gpt-4, claude-3-opus"
          />
        </div>
        <div className="col-span-2">
          <Label className="text-xs">API Key</Label>
          <Input
            className="h-8 text-xs mt-1"
            type="password"
            value={settings.api_key}
            onChange={(e) => update({ api_key: e.target.value })}
            placeholder="sk-..."
          />
        </div>
        {(settings.provider === 'openai' || settings.provider === 'ollama' || settings.provider === 'vllm') && (
          <div className="col-span-2">
            <Label className="text-xs">Base URL</Label>
            <Input
              className="h-8 text-xs mt-1"
              value={settings.base_url}
              onChange={(e) => update({ base_url: e.target.value })}
              placeholder={settings.provider === 'ollama' ? 'http://localhost:11434' : settings.provider === 'vllm' ? 'http://localhost:8000' : ''}
            />
          </div>
        )}
        {settings.provider === 'gemini' && (
          <>
            <div>
              <Label className="text-xs">Project ID</Label>
              <Input className="h-8 text-xs mt-1" value={settings.project_id} onChange={(e) => update({ project_id: e.target.value })} />
            </div>
            <div>
              <Label className="text-xs">Location</Label>
              <Input className="h-8 text-xs mt-1" value={settings.location} onChange={(e) => update({ location: e.target.value })} placeholder="us-central1" />
            </div>
          </>
        )}
        <div>
          <Label className="text-xs">Temperature</Label>
          <Input
            className="h-8 text-xs mt-1"
            type="number"
            min={0}
            max={2}
            step={0.1}
            value={settings.temperature}
            onChange={(e) => update({ temperature: parseFloat(e.target.value) || 0.7 })}
          />
        </div>
        <div>
          <Label className="text-xs">Max Tokens</Label>
          <Input
            className="h-8 text-xs mt-1"
            type="number"
            value={settings.max_tokens}
            onChange={(e) => update({ max_tokens: e.target.value })}
            placeholder="auto"
          />
        </div>
        <div className="col-span-2">
          <Label className="text-xs">System Message</Label>
          <Textarea
            className="text-xs mt-1 min-h-[60px]"
            value={settings.system_message}
            onChange={(e) => update({ system_message: e.target.value })}
            placeholder="Optional system instructions..."
          />
        </div>
      </CardContent>
    </Card>
  )
}

function GraphRAGSettingsPanel({
  settings,
  onChange,
}: {
  settings: GraphRAGSettings
  onChange: (s: GraphRAGSettings) => void
}) {
  const update = (patch: Partial<GraphRAGSettings>) => onChange({ ...settings, ...patch })

  return (
    <Card className="mt-2">
      <CardContent className="p-4 space-y-3">
        <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Neo4j Connection</p>
        <div className="grid grid-cols-2 gap-3">
          <div className="col-span-2">
            <Label className="text-xs">URI</Label>
            <Input className="h-8 text-xs mt-1" value={settings.neo4j_uri} onChange={(e) => update({ neo4j_uri: e.target.value })} />
          </div>
          <div>
            <Label className="text-xs">Username</Label>
            <Input className="h-8 text-xs mt-1" value={settings.neo4j_user} onChange={(e) => update({ neo4j_user: e.target.value })} />
          </div>
          <div>
            <Label className="text-xs">Password</Label>
            <Input className="h-8 text-xs mt-1" type="password" value={settings.neo4j_password} onChange={(e) => update({ neo4j_password: e.target.value })} />
          </div>
          <div className="col-span-2">
            <Label className="text-xs">Database</Label>
            <Input className="h-8 text-xs mt-1" value={settings.neo4j_database} onChange={(e) => update({ neo4j_database: e.target.value })} />
          </div>
        </div>

        <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider pt-2">LLM & Embeddings</p>
        <div className="grid grid-cols-2 gap-3">
          <div className="col-span-2">
            <Label className="text-xs">LLM API Key</Label>
            <Input className="h-8 text-xs mt-1" type="password" value={settings.llm_api_key} onChange={(e) => update({ llm_api_key: e.target.value })} placeholder="OpenAI API key" />
          </div>
          <div>
            <Label className="text-xs">LLM Model</Label>
            <Input className="h-8 text-xs mt-1" value={settings.llm_model} onChange={(e) => update({ llm_model: e.target.value })} />
          </div>
          <div>
            <Label className="text-xs">Embedding Model</Label>
            <Input className="h-8 text-xs mt-1" value={settings.embedding_model} onChange={(e) => update({ embedding_model: e.target.value })} />
          </div>
        </div>

        <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider pt-2">Retrieval</p>
        <div className="grid grid-cols-3 gap-3">
          <div>
            <Label className="text-xs">Agents</Label>
            <Input className="h-8 text-xs mt-1" type="number" min={1} max={5} value={settings.num_agents} onChange={(e) => update({ num_agents: parseInt(e.target.value) || 1 })} />
          </div>
          <div>
            <Label className="text-xs">Top-K</Label>
            <Input className="h-8 text-xs mt-1" type="number" min={1} max={20} value={settings.similarity_top_k} onChange={(e) => update({ similarity_top_k: parseInt(e.target.value) || 5 })} />
          </div>
          <div>
            <Label className="text-xs">Retrievers</Label>
            <p className="text-[10px] text-muted-foreground mt-1">
              {settings.retrievers.join(', ')}
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
