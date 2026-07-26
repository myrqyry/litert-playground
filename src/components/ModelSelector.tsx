import type { ModelAdapter } from '../adapters/types'

interface ModelSelectorProps {
  adapters: ModelAdapter[]
  onSelect: (adapter: ModelAdapter) => void
  disabled?: boolean
}

export default function ModelSelector({ adapters, onSelect, disabled }: ModelSelectorProps) {
  return (
    <div>
      <h2>Model</h2>
      <select
        onChange={e => {
          const a = adapters.find(a => a.modelId === e.target.value)
          if (a) onSelect(a)
        }}
        disabled={disabled}
        style={{ width: '100%', padding: 8 }}
      >
        <option value="">-- Select a model --</option>
        {adapters.map(a => (
          <option key={a.modelId} value={a.modelId}>
            {a.metadata.name}
          </option>
        ))}
      </select>
      {adapters.length === 0 && (
        <p style={{ color: '#999', fontSize: '0.85em' }}>No models registered yet</p>
      )}
    </div>
  )
}
