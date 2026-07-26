import type { ModelAdapter } from '../adapters/types'

interface ModelSelectorProps {
  adapters: ModelAdapter[]
  onSelect: (adapter: ModelAdapter) => void
  disabled?: boolean
}

export default function ModelSelector({ adapters, onSelect, disabled }: ModelSelectorProps) {
  return (
    <div>
      <h2 className="mb-2 text-sm font-semibold text-on-surface-variant uppercase tracking-wide">Model</h2>
      <select
        onChange={e => {
          const a = adapters.find(a => a.modelId === e.target.value)
          if (a) onSelect(a)
        }}
        disabled={disabled}
        className="w-full rounded-lg border border-outline bg-surface-container px-4 py-3 text-on-surface transition-colors focus:border-primary focus:ring-2 focus:ring-primary/30 focus:outline-none disabled:opacity-50"
      >
        <option value="" className="bg-surface-dim">-- Select a model --</option>
        {adapters.map(a => (
          <option key={a.modelId} value={a.modelId} className="bg-surface-dim">
            {a.metadata.name}
          </option>
        ))}
      </select>
      {adapters.length === 0 && (
        <p className="mt-1 text-sm text-on-surface-variant">No models registered yet</p>
      )}
    </div>
  )
}
