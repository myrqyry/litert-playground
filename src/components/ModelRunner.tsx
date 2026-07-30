import { useState } from 'react'
import { useModelRunner } from '../hooks/useModelRunner'
import type { Accelerator } from '../hooks/useModelRunner'
import type { ModelAdapter, TensorSpec } from '../adapters/types'
import ModelSelector from './ModelSelector'
import InputEditor from './InputEditor'
import ImageInput from './ImageInput'
import OutputViewer from './OutputViewer'

function isVisionSpec(spec: TensorSpec): boolean {
  const s = spec.shape
  if (s.length !== 4) return false
  // NCHW: [B, C, H, W] with both spatial dims > 4
  if (s[2] > 4 && s[3] > 4) return true
  // NHWC: [B, H, W, C] with both spatial dims > 4 and channels in [1..4]
  if (s[1] > 4 && s[2] > 4 && s[3] >= 1 && s[3] <= 4) return true
  return false
}

interface ModelRunnerProps {
  adapters: ModelAdapter[]
  selectedId?: string | null
  onSelect?: (id: string | null) => void
}

const ACCEL_OPTIONS: { value: Accelerator; label: string }[] = [
  { value: 'webgpu', label: 'WebGPU' },
  { value: 'wasm', label: 'WASM (CPU)' },
  { value: 'webnn', label: 'WebNN (NPU)' },
]

export default function ModelRunner({ adapters, onSelect }: ModelRunnerProps) {
  const { loadModel, runInference, outputs, outputTensors, outputSpecs, accelerator, setAccelerator, error, loading, loaded } = useModelRunner()
  const [selectedAdapter, setSelectedAdapter] = useState<ModelAdapter | null>(null)
  const [inputValues, setInputValues] = useState<Record<string, any>>({})
  const [search, setSearch] = useState('')

  const handleSelect = (adapter: ModelAdapter) => {
    if (onSelect && adapter.isPipeline) {
      onSelect(adapter.modelId)
      return
    }
    setSelectedAdapter(adapter)
    setInputValues({})
    loadModel(adapter)
  }

  const handleRun = () => runInference(inputValues)

  const filtered = search
    ? adapters.filter(a =>
        a.metadata.name.toLowerCase().includes(search.toLowerCase()) ||
        a.metadata.tags.some(t => t.toLowerCase().includes(search.toLowerCase())) ||
        a.modelId.toLowerCase().includes(search.toLowerCase())
      )
    : adapters

  return (
    <div className="min-h-screen bg-surface-dim">
      <div className="mx-auto p-6" style={{ maxWidth: 800 }}>
        <div className="mb-4 flex items-center justify-between">
          <h1 className="text-3xl font-bold text-on-surface">LiteRT Playground</h1>
          <div className="flex items-center gap-2">
            <label className="text-xs text-on-surface-variant">Accelerator:</label>
            <select
              value={accelerator}
              onChange={e => setAccelerator(e.target.value as Accelerator)}
              className="rounded-lg border border-outline bg-surface-container px-2 py-1 text-xs text-on-surface"
            >
              {ACCEL_OPTIONS.map(o => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </select>
          </div>
        </div>

        <input
          type="text"
          placeholder="Search models..."
          value={search}
          onChange={e => setSearch(e.target.value)}
          className="mb-3 w-full rounded-lg border border-outline bg-surface-container px-4 py-2 text-sm text-on-surface transition-colors focus:border-primary focus:ring-2 focus:ring-primary/30 focus:outline-none"
        />

        <ModelSelector adapters={filtered} onSelect={handleSelect} disabled={loading} />

        {error && (
          <div className="mt-3 rounded-lg bg-error-container p-3 text-sm text-on-error-container">
            {error}
          </div>
        )}

        {selectedAdapter && loaded && (
          <div className="mt-6 space-y-6">
            {selectedAdapter.inputSpecs.some(isVisionSpec) ? (
              <ImageInput specs={selectedAdapter.inputSpecs} onChange={setInputValues} />
            ) : (
              <InputEditor specs={selectedAdapter.inputSpecs} onChange={setInputValues} />
            )}

            <button
              onClick={handleRun}
              disabled={loading}
              className="inline-flex items-center justify-center rounded-full bg-primary px-8 py-3 text-sm font-medium text-on-primary shadow-md transition-all duration-300 hover:scale-[1.02] hover:shadow-lg active:scale-[0.97] disabled:opacity-50 disabled:shadow-none"
              style={{ transitionTimingFunction: 'var(--ease-spring)' }}
            >
              {loading ? 'Running...' : `Run Inference (${accelerator.toUpperCase()})`}
            </button>

            <OutputViewer outputs={outputs} outputTensors={outputTensors} outputSpecs={outputSpecs} />
          </div>
        )}

        {selectedAdapter && !loaded && loading && (
          <p className="mt-3 text-on-surface-variant">Loading model...</p>
        )}
      </div>
    </div>
  )
}
