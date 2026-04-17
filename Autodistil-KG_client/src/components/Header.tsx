import { TabsList, TabsTrigger } from '@/components/ui/tabs'

const TABS = [
  { id: 'configure', label: 'Configure Pipeline' },
  { id: 'monitor', label: 'Monitor Progress' },
  { id: 'results', label: 'Results & Output' },
] as const

export default function Header() {
  return (
    <header className="bg-card border-b border-border px-4 py-4">
      <h1 className="text-xl font-bold text-foreground">Autodistil-KG Pipeline</h1>
      <p className="text-sm text-muted-foreground mt-0.5">
        Build instruction-tuning datasets from knowledge graphs and fine-tune language models.
      </p>
      <TabsList className="mt-4 w-fit h-auto p-0 bg-transparent border-b border-border rounded-none gap-0">
        {TABS.map(({ id, label }) => (
          <TabsTrigger
            key={id}
            value={id}
            className="rounded-t-md border-b-2 border-transparent data-[state=active]:border-primary data-[state=active]:bg-primary/10 data-[state=active]:shadow-none"
          >
            {label}
          </TabsTrigger>
        ))}
      </TabsList>
    </header>
  )
}
