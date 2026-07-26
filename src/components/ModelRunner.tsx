import { useState } from 'react'
import { useModelRunner } from '../hooks/useModelRunner'
import type { ModelAdapter } from '../adapters/types'
import ModelSelector from './ModelSelector'
import InputEditor from './InputEditor'
import OutputViewer from './OutputViewer'

interface ModelRunnerProps {
  adapters: ModelAdapter[]
  selectedId?: string | null
  onSelect?: (id: string | null) => void
}

export default function ModelRunner({ adapters, onSelect }: ModelRunnerProps) {
  const { loadModel, runInference, outputs, error, loading, loaded } = useModelRunner()
  const [selectedAdapter, setSelectedAdapter] = useState<ModelAdapter | null>(null)
  const [inputValues, setInputValues] = useState<Record<string, any>>({})

  const handleSelect = (adapter: ModelAdapter) => {
    if (onSelect && (adapter as any).isPipeline) {
      onSelect(adapter.modelId)
      return
    }
    setSelectedAdapter(adapter)
    setInputValues({})
    loadModel(adapter)
  }

  const handleRun = () => runInference(inputValues)

  return (
    <div className="min-h-screen bg-surface-dim">
      <div className="mx-auto p-6" style={{ maxWidth: 800 }}>
        <h1 className="mb-6 text-3xl font-bold text-on-surface">LiteRT Playground</h1>

        <ModelSelector adapters={adapters} onSelect={handleSelect} disabled={loading} />

        {error && (
          <div className="mt-3 rounded-lg bg-error-container p-3 text-sm text-on-error-container">
            {error}
          </div>
        )}

        {selectedAdapter && loaded && (
          <div className="mt-6 space-y-6">
            <InputEditor specs={selectedAdapter.inputSpecs} onChange={setInputValues} />

            <button
              onClick={handleRun}
              disabled={loading}
              className="inline-flex items-center justify-center rounded-full bg-primary px-8 py-3 text-sm font-medium text-on-primary shadow-md transition-all duration-300 hover:scale-[1.02] hover:shadow-lg active:scale-[0.97] disabled:opacity-50 disabled:shadow-none"
              style={{ transitionTimingFunction: 'var(--ease-spring)' }}
            >
              {loading ? 'Running...' : 'Run Inference'}
            </button>

            <OutputViewer outputs={outputs} />
          </div>
        )}

        {selectedAdapter && !loaded && loading && (
          <p className="mt-3 text-on-surface-variant">Loading model...</p>
        )}
      </div>
    </div>
  )
}
