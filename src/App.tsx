import { useState } from 'react'
import ModelRunner from './components/ModelRunner'
import { Qwen3TtsPanel } from './components/Qwen3TtsPanel'
import { registeredAdapters } from './adapters/registry'
import type { ModelAdapter } from './adapters/types'

const ttsEntry: ModelAdapter = {
  modelId: 'qwen3-tts',
  metadata: { name: 'Qwen3-TTS (Speech Synthesis)', description: 'Multilingual TTS via 3-graph pipeline', modelPath: '', tags: ['tts', 'pipeline'] },
  inputSpecs: [],
  outputSpecs: [],
  isPipeline: true,
  prepareInputs: () => ({}),
  parseOutputs: async () => ({}),
}

export default function App() {
  const [selectedId, setSelectedId] = useState<string | null>(null)

  if (selectedId === 'qwen3-tts') {
    return (
      <div className="min-h-screen bg-surface-dim">
        <div className="mx-auto max-w-2xl p-4">
          <button
            className="mb-4 text-sm text-secondary transition-colors hover:text-on-secondary-container"
            onClick={() => setSelectedId(null)}
          >
            ← Back to model list
          </button>
          <h1 className="mb-4 text-2xl font-bold text-on-surface">Qwen3-TTS</h1>
          <Qwen3TtsPanel />
        </div>
      </div>
    )
  }

  return (
    <ModelRunner
      adapters={[...registeredAdapters, ttsEntry]}
      selectedId={selectedId}
      onSelect={setSelectedId}
    />
  )
}
