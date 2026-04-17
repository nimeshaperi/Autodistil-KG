import { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Textarea } from '@/components/ui/textarea'
import {
  inferenceLLM,
  inferenceGraphRAG,
  inferenceFinetuned,
  listRegisteredModels,
  listModels,
  registerModel,
} from '../api/client'
import type { LLMProviderType, RegisteredModel, AvailableModel } from '../types/config'

type Mode = 'llm' | 'finetuned' | 'graphrag'

export default function InferencePlayground() {
  const [mode, setMode] = useState<Mode>('llm')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  // LLM state
  const [provider, setProvider] = useState<LLMProviderType>('ollama')
  const [model, setModel] = useState('')
  const [apiKey, setApiKey] = useState('')
  const [baseUrl, setBaseUrl] = useState('')
  const [prompt, setPrompt] = useState('')

  // Finetuned state
  const [registeredModels, setRegisteredModels] = useState<RegisteredModel[]>([])
  const [availableModels, setAvailableModels] = useState<AvailableModel[]>([])
  const [selectedModelId, setSelectedModelId] = useState('')
  const [ftPrompt, setFtPrompt] = useState('')
  const [maxNewTokens, setMaxNewTokens] = useState('512')
  // Register form
  const [showRegister, setShowRegister] = useState(false)
  const [regId, setRegId] = useState('')
  const [regPath, setRegPath] = useState('')
  const [regBase, setRegBase] = useState('')
  const [regDesc, setRegDesc] = useState('')

  // GraphRAG state
  const [question, setQuestion] = useState('')
  const [neo4jUri, setNeo4jUri] = useState('bolt://localhost:7687')
  const [neo4jUser, setNeo4jUser] = useState('neo4j')
  const [neo4jPassword, setNeo4jPassword] = useState('')
  const [llmApiKey, setLlmApiKey] = useState('')
  const [llmModel, setLlmModel] = useState('gpt-4')
  const [llmBaseUrl, setLlmBaseUrl] = useState('')

  // Load models when switching to finetuned tab
  useEffect(() => {
    if (mode === 'finetuned') {
      loadModels()
    }
  }, [mode])

  const loadModels = async () => {
    try {
      const [reg, avail] = await Promise.all([listRegisteredModels(), listModels()])
      setRegisteredModels(reg)
      setAvailableModels(avail)
    } catch {
      // silently fail — models may not be available yet
    }
  }

  const handleRegister = async () => {
    if (!regId || !regPath) return
    try {
      await registerModel({
        model_id: regId,
        model_path: regPath,
        base_model: regBase || undefined,
        description: regDesc || undefined,
      })
      setShowRegister(false)
      setRegId('')
      setRegPath('')
      setRegBase('')
      setRegDesc('')
      await loadModels()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Registration failed')
    }
  }

  const handleQuickRegister = async (m: AvailableModel) => {
    const modelId = `run-${m.run_id.slice(0, 8)}`
    try {
      await registerModel({ model_id: modelId, model_path: m.model_path })
      await loadModels()
      setSelectedModelId(modelId)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Quick registration failed')
    }
  }

  const handleSubmit = async () => {
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      if (mode === 'llm') {
        const res = await inferenceLLM({
          provider,
          model: model || undefined,
          api_key: apiKey || undefined,
          base_url: baseUrl || undefined,
          messages: [{ role: 'user', content: prompt }],
        })
        setResult(res.response)
      } else if (mode === 'finetuned') {
        if (!selectedModelId) {
          setError('Select a model first')
          return
        }
        const res = await inferenceFinetuned({
          model_id: selectedModelId,
          messages: [{ role: 'user', content: ftPrompt }],
          max_new_tokens: parseInt(maxNewTokens) || 512,
        })
        setResult(res.response)
      } else {
        const res = await inferenceGraphRAG({
          question,
          neo4j_uri: neo4jUri,
          neo4j_user: neo4jUser,
          neo4j_password: neo4jPassword,
          llm_api_key: llmApiKey || undefined,
          llm_model: llmModel || undefined,
          llm_base_url: llmBaseUrl || undefined,
        })
        setResult(res.answer)
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }

  // Unregistered models from run discovery that aren't in the registry yet
  const unregisteredModels = availableModels.filter(
    (m) => !registeredModels.some((r) => r.model_path === m.model_path)
  )

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Inference Playground</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex gap-2">
            <Button variant={mode === 'llm' ? 'default' : 'outline'} size="sm" onClick={() => setMode('llm')}>
              LLM Prompt
            </Button>
            <Button variant={mode === 'finetuned' ? 'default' : 'outline'} size="sm" onClick={() => setMode('finetuned')}>
              Finetuned Model
            </Button>
            <Button variant={mode === 'graphrag' ? 'default' : 'outline'} size="sm" onClick={() => setMode('graphrag')}>
              GraphRAG Query
            </Button>
          </div>

          {mode === 'llm' ? (
            <div className="space-y-3">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <Label>Provider</Label>
                  <Select value={provider} onValueChange={(v) => setProvider(v as LLMProviderType)}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="openai">OpenAI</SelectItem>
                      <SelectItem value="claude">Claude</SelectItem>
                      <SelectItem value="gemini">Gemini</SelectItem>
                      <SelectItem value="ollama">Ollama</SelectItem>
                      <SelectItem value="vllm">vLLM</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <Label>Model</Label>
                  <Input value={model} onChange={(e) => setModel(e.target.value)} placeholder="e.g. gpt-4, gemma3:4b" />
                </div>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <Label>API Key</Label>
                  <Input type="password" value={apiKey} onChange={(e) => setApiKey(e.target.value)} placeholder="Optional" />
                </div>
                <div>
                  <Label>Base URL</Label>
                  <Input value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)} placeholder="Optional" />
                </div>
              </div>
              <div>
                <Label>Prompt</Label>
                <Textarea rows={4} value={prompt} onChange={(e) => setPrompt(e.target.value)} placeholder="Enter your prompt..." />
              </div>
            </div>
          ) : mode === 'finetuned' ? (
            <div className="space-y-3">
              {/* Model selector */}
              <div>
                <Label>Model</Label>
                {registeredModels.length > 0 ? (
                  <Select value={selectedModelId} onValueChange={setSelectedModelId}>
                    <SelectTrigger><SelectValue placeholder="Select a finetuned model" /></SelectTrigger>
                    <SelectContent>
                      {registeredModels.map((m) => (
                        <SelectItem key={m.model_id} value={m.model_id}>
                          {m.model_id}{m.description ? ` — ${m.description}` : ''}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                ) : (
                  <p className="text-sm text-muted-foreground py-2">No registered models. Register one below.</p>
                )}
              </div>

              {/* Unregistered models from pipeline runs */}
              {unregisteredModels.length > 0 && (
                <div className="border rounded-md p-3 space-y-2">
                  <p className="text-xs font-medium text-muted-foreground">Discovered from pipeline runs (click to register):</p>
                  {unregisteredModels.map((m) => (
                    <Button
                      key={m.run_id}
                      variant="outline"
                      size="sm"
                      className="mr-2 mb-1"
                      onClick={() => handleQuickRegister(m)}
                    >
                      {m.label}
                    </Button>
                  ))}
                </div>
              )}

              {/* Register new model */}
              <div>
                <Button variant="outline" size="sm" onClick={() => setShowRegister(!showRegister)}>
                  {showRegister ? 'Cancel' : 'Register New Model'}
                </Button>
              </div>
              {showRegister && (
                <div className="border rounded-md p-3 space-y-2">
                  <div className="grid grid-cols-2 gap-2">
                    <div>
                      <Label className="text-xs">Model ID</Label>
                      <Input value={regId} onChange={(e) => setRegId(e.target.value)} placeholder="e.g. adhd-kg-v1" />
                    </div>
                    <div>
                      <Label className="text-xs">Model Path</Label>
                      <Input value={regPath} onChange={(e) => setRegPath(e.target.value)} placeholder="/path/to/finetuned/adapter" />
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    <div>
                      <Label className="text-xs">Base Model (optional)</Label>
                      <Input value={regBase} onChange={(e) => setRegBase(e.target.value)} placeholder="e.g. unsloth/gemma-2-2b-it" />
                    </div>
                    <div>
                      <Label className="text-xs">Description (optional)</Label>
                      <Input value={regDesc} onChange={(e) => setRegDesc(e.target.value)} placeholder="Brief description" />
                    </div>
                  </div>
                  <Button size="sm" onClick={handleRegister}>Register</Button>
                </div>
              )}

              {/* Inference form */}
              <div>
                <Label>Max New Tokens</Label>
                <Input type="number" value={maxNewTokens} onChange={(e) => setMaxNewTokens(e.target.value)} className="w-32" />
              </div>
              <div>
                <Label>Prompt</Label>
                <Textarea rows={4} value={ftPrompt} onChange={(e) => setFtPrompt(e.target.value)} placeholder="Enter your prompt for the finetuned model..." />
              </div>
            </div>
          ) : (
            <div className="space-y-3">
              <div>
                <Label>Question</Label>
                <Textarea rows={3} value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="Ask a question about the knowledge graph..." />
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <Label>Neo4j URI</Label>
                  <Input value={neo4jUri} onChange={(e) => setNeo4jUri(e.target.value)} />
                </div>
                <div>
                  <Label>Neo4j User</Label>
                  <Input value={neo4jUser} onChange={(e) => setNeo4jUser(e.target.value)} />
                </div>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <Label>Neo4j Password</Label>
                  <Input type="password" value={neo4jPassword} onChange={(e) => setNeo4jPassword(e.target.value)} />
                </div>
                <div>
                  <Label>LLM API Key</Label>
                  <Input type="password" value={llmApiKey} onChange={(e) => setLlmApiKey(e.target.value)} placeholder="API key" />
                </div>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <Label>LLM Model</Label>
                  <Input value={llmModel} onChange={(e) => setLlmModel(e.target.value)} placeholder="e.g. gpt-4, qwen/qwen3.5-27b" />
                </div>
                <div>
                  <Label>LLM Base URL</Label>
                  <Input value={llmBaseUrl} onChange={(e) => setLlmBaseUrl(e.target.value)} placeholder="e.g. https://openrouter.ai/api/v1" />
                </div>
              </div>
            </div>
          )}

          <Button onClick={handleSubmit} disabled={loading}>
            {loading ? 'Running...' : mode === 'llm' ? 'Send Prompt' : mode === 'finetuned' ? 'Run Inference' : 'Query GraphRAG'}
          </Button>
        </CardContent>
      </Card>

      {(result || error) && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">{error ? 'Error' : 'Response'}</CardTitle>
          </CardHeader>
          <CardContent>
            {error ? (
              <p className="text-sm text-destructive">{error}</p>
            ) : (
              <pre className="text-sm whitespace-pre-wrap bg-muted p-3 rounded-md max-h-96 overflow-y-auto">{result}</pre>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  )
}
