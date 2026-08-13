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
  if (s[2] > 4 && s[3] > 4) return true
  if (s[1] > 4 && s[2] > 4 && s[3] >= 1 && s[3] <= 4) return true
  return false
}

interface ModelRunnerProps {
  adapters: ModelAdapter[]
  selectedId?: string | null
  onSelect?: (id: string | null) => void
}

const ACCEL_OPTIONS: { value: Accelerator; label: string }[] = [
  { value: 'auto', label: 'Auto (GPU → NPU → CPU)' },
  { value: 'webgpu', label: 'WebGPU' },
  { value: 'webnn', label: 'WebNN' },
  { value: 'wasm', label: 'WASM (CPU)' },
]

function metric(value: number | undefined): string {
  return value === undefined ? '—' : `${Math.round(value)} ms`
}

export default function ModelRunner({ adapters, onSelect }: ModelRunnerProps) {
  const {
    loadModel,
    runInference,
    preflightModel,
    outputs,
    outputTensors,
    outputSpecs,
    accelerator,
    setAccelerator,
    resolvedAccelerator,
    modelInfo,
    preflight,
    telemetry,
    error,
    loading,
    loaded,
  } = useModelRunner()
  const [selectedAdapter, setSelectedAdapter] = useState<ModelAdapter | null>(null)
  const [inputValues, setInputValues] = useState<Record<string, unknown>>({})
  const [search, setSearch] = useState('')

  const handleSelect = (adapter: ModelAdapter) => {
    if (onSelect && adapter.isPipeline) {
      onSelect(adapter.modelId)
      return
    }
    setSelectedAdapter(adapter)
    setInputValues({})
    void loadModel(adapter)
  }

  const handleAcceleratorChange = (next: Accelerator) => {
    setAccelerator(next)
    if (selectedAdapter) void loadModel(selectedAdapter, next)
  }

  const filtered = search
    ? adapters.filter(a =>
        a.metadata.name.toLowerCase().includes(search.toLowerCase()) ||
        a.metadata.tags.some(t => t.toLowerCase().includes(search.toLowerCase())) ||
        a.modelId.toLowerCase().includes(search.toLowerCase())
      )
    : adapters

  const recentTelemetry = telemetry.slice(-4).reverse()

  return (
    <div className="min-h-screen bg-surface-dim">
      <div className="mx-auto p-6" style={{ maxWidth: 900 }}>
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <div>
            <h1 className="text-3xl font-bold text-on-surface">LiteRT Playground</h1>
            <p className="mt-1 text-sm text-on-surface-variant">
              Shared runtime qualification lab for browser and local inference packages.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <label className="text-xs text-on-surface-variant">Accelerator:</label>
            <select
              value={accelerator}
              onChange={e => handleAcceleratorChange(e.target.value as Accelerator)}
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
            <section className="rounded-2xl border border-outline/40 bg-surface-container p-4">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <p className="text-xs font-medium uppercase tracking-wide text-on-surface-variant">Runtime receipt</p>
                  <p className="mt-1 text-sm font-semibold text-on-surface">
                    requested {accelerator.toUpperCase()} → resolved {(resolvedAccelerator ?? 'unknown').toUpperCase()}
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => void preflightModel()}
                  disabled={loading}
                  className="rounded-full border border-outline px-4 py-2 text-xs font-medium text-on-surface transition-colors hover:bg-surface-container-high disabled:opacity-50"
                >
                  {loading ? 'Working…' : 'Run preflight'}
                </button>
              </div>

              <div className="mt-3 grid grid-cols-2 gap-3 text-xs sm:grid-cols-4">
                <div>
                  <p className="text-on-surface-variant">Compile</p>
                  <p className="font-medium text-on-surface">{metric(modelInfo?.compileDurationMs)}</p>
                </div>
                <div>
                  <p className="text-on-surface-variant">Fallbacks</p>
                  <p className="font-medium text-on-surface">{modelInfo?.fallbackCount ?? 0}</p>
                </div>
                <div>
                  <p className="text-on-surface-variant">Preflight inference</p>
                  <p className="font-medium text-on-surface">{metric(preflight?.inferenceDurationMs)}</p>
                </div>
                <div>
                  <p className="text-on-surface-variant">Preflight outputs</p>
                  <p className="font-medium text-on-surface">{preflight?.outputCount ?? '—'}</p>
                </div>
              </div>

              {recentTelemetry.length > 0 && (
                <div className="mt-4 border-t border-outline/30 pt-3">
                  <p className="mb-2 text-xs font-medium text-on-surface-variant">Recent runtime events</p>
                  <div className="space-y-1 font-mono text-[11px] text-on-surface-variant">
                    {recentTelemetry.map((entry, index) => (
                      <div key={`${entry.timestamp}-${entry.event}-${index}`} className="flex flex-wrap justify-between gap-2">
                        <span>{entry.event} · {entry.resolvedBackend}</span>
                        <span>
                          {entry.inferenceDurationMs !== undefined
                            ? `${Math.round(entry.inferenceDurationMs)} ms`
                            : `${Math.round(entry.compileDurationMs)} ms`}
                          {entry.fallbackCount > 0 ? ` · ${entry.fallbackCount} fallback` : ''}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </section>

            {selectedAdapter.inputSpecs.some(isVisionSpec) ? (
              <ImageInput specs={selectedAdapter.inputSpecs} onChange={setInputValues} />
            ) : (
              <InputEditor specs={selectedAdapter.inputSpecs} onChange={setInputValues} />
            )}

            <button
              onClick={() => void runInference(inputValues)}
              disabled={loading}
              className="inline-flex items-center justify-center rounded-full bg-primary px-8 py-3 text-sm font-medium text-on-primary shadow-md transition-all duration-300 hover:scale-[1.02] hover:shadow-lg active:scale-[0.97] disabled:opacity-50 disabled:shadow-none"
              style={{ transitionTimingFunction: 'var(--ease-spring)' }}
            >
              {loading ? 'Running...' : `Run Inference (${(resolvedAccelerator ?? accelerator).toUpperCase()})`}
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
