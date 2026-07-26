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

export default function ModelRunner({ adapters, selectedId: externalId, onSelect }: ModelRunnerProps) {
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

  const handleRun = () => {
    runInference(inputValues)
  }

  return (
    <div style={{ maxWidth: 800, margin: 'auto', padding: 20 }}>
      <h1>LiteRT Playground</h1>

      <ModelSelector adapters={adapters} onSelect={handleSelect} disabled={loading} />

      {error && (
        <div style={{ color: '#d32f2f', background: '#ffebee', padding: 8, borderRadius: 4, marginTop: 12 }}>
          {error}
        </div>
      )}

      {selectedAdapter && loaded && (
        <>
          <InputEditor specs={selectedAdapter.inputSpecs} onChange={setInputValues} />

          <button
            onClick={handleRun}
            disabled={loading}
            style={{ marginTop: 16, padding: '10px 24px', fontSize: 16, cursor: loading ? 'wait' : 'pointer' }}
          >
            {loading ? 'Running...' : 'Run Inference'}
          </button>

          <OutputViewer outputs={outputs} />
        </>
      )}

      {selectedAdapter && !loaded && loading && (
        <p style={{ color: '#666', marginTop: 12 }}>Loading model...</p>
      )}
    </div>
  )
}
