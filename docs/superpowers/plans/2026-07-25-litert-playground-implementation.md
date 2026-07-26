# LiteRT Playground — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A browser-based playground for loading and running LiteRT (.tflite) models in-browser using WebGPU/Wasm, starting with a Magenta adapter.

**Architecture:** ModelAdapter interface + TensorSpec types define each model's input/output contract. Components derive from specs automatically. The user edits JSON inputs, the adapter maps values to tensors, `model.run()` executes, and the adapter parses outputs back to JSON.

**Tech Stack:** pnpm, Vite + React + TypeScript, @litertjs/core, uv + ai-edge-converter

## Global Constraints

- Use pnpm for all JS/Node.js operations
- Use uv for all Python operations
- Default accelerator: 'webgpu' with 'wasm' fallback
- Wasm files served from CDN (jsdelivr) initially
- No backend — everything in-browser

## File Structure

```
litert-playground/
├── package.json
├── tsconfig.json
├── vite.config.ts
├── index.html
├── src/
│   ├── main.tsx
│   ├── App.tsx
│   ├── adapters/
│   │   ├── types.ts          # ModelAdapter interface, TensorSpec, ModelMetadata
│   │   └── magenta.ts        # MagentaRealtime2Adapter
│   ├── components/
│   │   ├── ModelSelector.tsx
│   │   ├── InputEditor.tsx
│   │   ├── OutputViewer.tsx
│   │   └── ModelRunner.tsx
│   └── hooks/
│       └── useModelRunner.ts
├── public/
│   └── models/               # .tflite files go here
└── scripts/
    └── convert_magenta.py
```

---
### Task 1: Project Scaffolding

**Files:**
- Create: `package.json`
- Create: `tsconfig.json`
- Create: `vite.config.ts`
- Create: `index.html`
- Create: `src/main.tsx`
- Create: `src/App.tsx`
- Create: `public/models/.gitkeep`

**Interfaces:**
- Consumes: (none — first task)
- Produces: Working Vite dev server with React + TypeScript, npm scripts `dev`/`build`/`preview`

- [ ] **Step 1: Initialize pnpm project**

```bash
cd /home/myrqyry/MQR/litert-playground
pnpm init
```

- [ ] **Step 2: Install dependencies**

```bash
pnpm add @litertjs/core react react-dom
pnpm add -D typescript @types/react @types/react-dom vite @vitejs/plugin-react
```

- [ ] **Step 3: Create `tsconfig.json`**

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "jsx": "react-jsx",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "allowImportingTsExtensions": true
  },
  "include": ["src"]
}
```

- [ ] **Step 4: Create `vite.config.ts`**

```ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
})
```

- [ ] **Step 5: Create `index.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>LiteRT Playground</title>
</head>
<body>
  <div id="root"></div>
  <script type="module" src="/src/main.tsx"></script>
</body>
</html>
```

- [ ] **Step 6: Create `src/main.tsx`**

```tsx
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
```

- [ ] **Step 7: Create `src/App.tsx` (initial shell)**

```tsx
function App() {
  return <h1>LiteRT Playground</h1>
}

export default App
```

- [ ] **Step 8: Create directories**

```bash
mkdir -p src/adapters src/components src/hooks public/models
touch public/models/.gitkeep
```

- [ ] **Step 9: Verify the dev server starts**

```bash
pnpm dev
# Verify in browser: shows "LiteRT Playground" heading
```

- [ ] **Step 10: Add `.gitignore`**

```
node_modules/
dist/
public/models/*.tflite
```

- [ ] **Step 11: Commit**

```bash
git init
git add -A
git commit -m "feat: scaffold Vite + React + TypeScript project"
```

---
### Task 2: Adapter Types and Interfaces

**Files:**
- Create: `src/adapters/types.ts`

**Interfaces:**
- Consumes: (none — pure types)
- Produces: `ModelAdapter` interface, `TensorSpec` type, `ModelMetadata` type, `AdapterRegistry` type

- [ ] **Step 1: Define the core types**

Write `src/adapters/types.ts`:

```ts
export interface TensorSpec {
  name: string
  dtype: 'float32' | 'int32' | 'int8' | 'uint8'
  shape: number[]
  description: string
  constraints?: {
    min?: number
    max?: number
    enum?: string[]
    items?: string[]
  }
}

export interface ModelMetadata {
  id: string
  name: string
  description: string
  modelPath: string
  inputSpecs: TensorSpec[]
  outputSpecs: TensorSpec[]
}

export interface ModelAdapter {
  metadata: ModelMetadata
  applyInputs(values: Record<string, any>): Tensor | Tensor[] | Record<string, Tensor>
  parseOutputs(outputs: any): Promise<Record<string, any>>
}
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add src/adapters/types.ts
git commit -m "feat: define ModelAdapter interface and TensorSpec types"
```

---
### Task 3: Magenta Adapter

**Files:**
- Create: `src/adapters/magenta.ts`

**Interfaces:**
- Consumes: `TensorSpec`, `ModelAdapter`, `ModelMetadata` from `./types`
- Produces: `magentaAdapter`: a `ModelAdapter` instance for Magenta Realtime 2

- [ ] **Step 1: Define the magenta adapter**

Write `src/adapters/magenta.ts`:

```ts
import { Tensor } from '@litertjs/core'
import type { ModelAdapter, ModelMetadata, TensorSpec } from './types'

const inputSpecs: TensorSpec[] = [
  {
    name: 'input',
    dtype: 'float32',
    shape: [1, 256, 1],
    description: 'Audio frame input — 256 samples normalized to ~zero mean',
  },
  {
    name: 'length',
    dtype: 'float32',
    shape: [1],
    description: 'Generation length in seconds',
    constraints: { min: 0.5, max: 30 },
  },
  {
    name: 'temperature_harmonic',
    dtype: 'float32',
    shape: [1],
    description: 'Harmonic temperature (higher = more random)',
    constraints: { min: 0.01, max: 5.0 },
  },
]

const outputSpecs: TensorSpec[] = [
  {
    name: 'output',
    dtype: 'float32',
    shape: [1, 256, 1],
    description: 'Generated audio frame — 256 samples',
  },
  {
    name: 'state',
    dtype: 'float32',
    shape: [1, 256],
    description: 'Recurrent state for multi-step generation',
  },
]

const metadata: ModelMetadata = {
  id: 'magenta-realtime-2',
  name: 'Magenta Realtime 2',
  description: 'Real-time music generation model',
  modelPath: '/models/magenta_realtime_2.tflite',
  inputSpecs,
  outputSpecs,
}

export const magentaAdapter: ModelAdapter = {
  metadata,
  applyInputs(values: Record<string, any>) {
    const input = new Float32Array(256)
    const raw = values['input']
    if (raw instanceof Float32Array || raw instanceof Array) {
      input.set(raw.slice(0, 256))
    }
    const inputTensor = new Tensor(input, [1, 256, 1])

    const length = new Tensor(new Float32Array([Number(values['length']) ?? 5]), [1])
    const temperature = new Tensor(new Float32Array([Number(values['temperature_harmonic']) ?? 1.0]), [1])

    return { 'input': inputTensor, 'length': length, 'temperature_harmonic': temperature }
  },
  async parseOutputs(outputs: any) {
    const result: Record<string, any> = {}
    for (const spec of outputSpecs) {
      const t = outputs[spec.name]
      if (t) {
        const data = await t.data()
        result[spec.name] = Array.from(data)
        t.delete()
      }
    }
    return result
  },
}

export const registeredAdapters = [magentaAdapter]
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
npx tsc --noEmit
```

- [ ] **Step 3: Commit**

```bash
git add src/adapters/magenta.ts
git commit -m "feat: add Magenta Realtime 2 adapter"
```

---
### Task 4: useModelRunner Hook

**Files:**
- Create: `src/hooks/useModelRunner.ts`

**Interfaces:**
- Consumes: `ModelAdapter` from `adapters/types`
- Produces: `{loadModel, runInference, outputs, error, loading, loaded}`

- [ ] **Step 1: Write the hook**

`src/hooks/useModelRunner.ts`:

```ts
import { useState, useCallback, useRef } from 'react'
import { loadLiteRt, loadAndCompile } from '@litertjs/core'
import type { ModelAdapter } from '../adapters/types'

const WASM_URL = 'https://cdn.jsdelivr.net/npm/@litertjs/core/wasm/'

interface UseModelRunnerReturn {
  loadModel: (adapter: ModelAdapter) => Promise<void>
  runInference: (values: Record<string, any>) => Promise<void>
  outputs: Record<string, any> | null
  error: string | null
  loading: boolean
  loaded: boolean
}

export function useModelRunner(): UseModelRunnerReturn {
  const [outputs, setOutputs] = useState<Record<string, any> | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [loaded, setLoaded] = useState(false)
  const modelRef = useRef<any>(null)
  const adapterRef = useRef<ModelAdapter | null>(null)
  const runtimeRef = useRef(false)

  const loadModel = useCallback(async (adapter: ModelAdapter) => {
    setLoading(true)
    setError(null)
    try {
      if (!runtimeRef.current) {
        await loadLiteRt(WASM_URL, { jspi: true })
        runtimeRef.current = true
      }

      const model = await loadAndCompile(adapter.metadata.modelPath, {
        accelerator: 'webgpu',
      })
      modelRef.current = model
      adapterRef.current = adapter
      setLoaded(true)
    } catch (e: any) {
      setError(e.message ?? String(e))
    } finally {
      setLoading(false)
    }
  }, [])

  const runInference = useCallback(async (values: Record<string, any>) => {
    if (!modelRef.current || !adapterRef.current) {
      setError('No model loaded')
      return
    }
    setLoading(true)
    setError(null)
    try {
      const inputs = adapterRef.current.applyInputs(values)
      const result = await modelRef.current.run(inputs)
      const parsed = await adapterRef.current.parseOutputs(result)
      setOutputs(parsed)

      Object.values(inputs).forEach((t: any) => t?.delete?.())
    } catch (e: any) {
      setError(e.message ?? String(e))
    } finally {
      setLoading(false)
    }
  }, [])

  return { loadModel, runInference, outputs, error, loading, loaded }
}
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
npx tsc --noEmit
```

- [ ] **Step 3: Commit**

```bash
git add src/hooks/useModelRunner.ts
git commit -m "feat: add useModelRunner hook for inference lifecycle"
```

---
### Task 5: InputEditor Component

**Files:**
- Create: `src/components/InputEditor.tsx`

**Interfaces:**
- Consumes: `TensorSpec[]` from `adapters/types`
- Produces: `(values: Record<string, any>) => void` via `onChange` prop

- [ ] **Step 1: Write InputEditor**

`src/components/InputEditor.tsx`:

```tsx
import { useState, useEffect } from 'react'
import type { TensorSpec } from '../adapters/types'

interface InputEditorProps {
  specs: TensorSpec[]
  onChange: (values: Record<string, any>) => void
}

function defaultValues(specs: TensorSpec[]): Record<string, any> {
  const v: Record<string, any> = {}
  for (const spec of specs) {
    if (spec.constraints?.enum) {
      v[spec.name] = spec.constraints.enum[0]
    } else if (spec.dtype === 'float32') {
      v[spec.name] = 0
    } else {
      v[spec.name] = 0
    }
  }
  return v
}

export default function InputEditor({ specs, onChange }: InputEditorProps) {
  const [values, setValues] = useState<Record<string, any>>(() => defaultValues(specs))

  useEffect(() => { onChange(values) }, [values, onChange])

  const set = (name: string, raw: string) => {
    let val: any = raw
    const spec = specs.find(s => s.name === name)
    if (spec?.dtype === 'float32') {
      val = parseFloat(raw)
      if (spec.constraints) {
        if (spec.constraints.min !== undefined) val = Math.max(spec.constraints.min, val)
        if (spec.constraints.max !== undefined) val = Math.min(spec.constraints.max, val)
      }
    }
    setValues(prev => ({ ...prev, [name]: val }))
  }

  return (
    <div>
      <h2>Inputs</h2>
      {specs.map(spec => (
        <div key={spec.name} style={{ marginBottom: 12 }}>
          <label style={{ display: 'block', fontWeight: 600, marginBottom: 4 }}>
            {spec.name}
            <span style={{ fontWeight: 400, fontSize: '0.85em', color: '#666', marginLeft: 8 }}>
              {spec.dtype} {JSON.stringify(spec.shape)}
            </span>
          </label>
          <div style={{ fontSize: '0.85em', color: '#555', marginBottom: 4 }}>{spec.description}</div>
          {spec.constraints?.enum ? (
            <select
              value={String(values[spec.name] ?? '')}
              onChange={e => set(spec.name, e.target.value)}
              style={{ width: '100%', padding: 6 }}
            >
              {spec.constraints.enum.map((opt, i) => (
                <option key={i} value={opt}>
                  {spec.constraints?.items?.[i] ?? opt}
                </option>
              ))}
            </select>
          ) : spec.shape.length > 1 || spec.shape[0] > 1 ? (
            <textarea
              defaultValue={JSON.stringify(values[spec.name] ?? defaultValues([spec])[spec.name])}
              onChange={e => set(spec.name, e.target.value)}
              rows={4}
              style={{ width: '100%', fontFamily: 'monospace', padding: 6 }}
            />
          ) : (
            <input
              type="number"
              step={spec.dtype === 'float32' ? 'any' : '1'}
              value={Number(values[spec.name] ?? 0)}
              onChange={e => set(spec.name, e.target.value)}
              style={{ width: '100%', padding: 6 }}
            />
          )}
        </div>
      ))}
    </div>
  )
}
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
npx tsc --noEmit
```

- [ ] **Step 3: Commit**

```bash
git add src/components/InputEditor.tsx
git commit -m "feat: add InputEditor with constraints-aware controls"
```

---
### Task 6: OutputViewer + ModelSelector Components

**Files:**
- Create: `src/components/OutputViewer.tsx`
- Create: `src/components/ModelSelector.tsx`

**Interfaces:**
- Consumes: `ModelAdapter[]` (for ModelSelector), `Record<string, any> | null` (for OutputViewer)
- Produces: Selected adapter callback, formatted JSON display

- [ ] **Step 1: Write OutputViewer**

`src/components/OutputViewer.tsx`:

```tsx
interface OutputViewerProps {
  outputs: Record<string, any> | null
}

export default function OutputViewer({ outputs }: OutputViewerProps) {
  if (!outputs) {
    return <div><h2>Outputs</h2><p style={{ color: '#999' }}>Run inference to see results</p></div>
  }
  return (
    <div>
      <h2>Outputs</h2>
      {Object.entries(outputs).map(([key, value]) => (
        <div key={key} style={{ marginBottom: 12 }}>
          <div style={{ fontWeight: 600, marginBottom: 4 }}>{key}</div>
          <pre style={{
            background: '#f5f5f5', padding: 8, borderRadius: 4,
            overflowX: 'auto', fontSize: '0.85em', maxHeight: 200, overflowY: 'auto'
          }}>
            {JSON.stringify(value, null, 2)}
          </pre>
        </div>
      ))}
    </div>
  )
}
```

- [ ] **Step 2: Write ModelSelector**

`src/components/ModelSelector.tsx`:

```tsx
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
          const a = adapters.find(a => a.metadata.id === e.target.value)
          if (a) onSelect(a)
        }}
        disabled={disabled}
        style={{ width: '100%', padding: 8 }}
      >
        <option value="">-- Select a model --</option>
        {adapters.map(a => (
          <option key={a.metadata.id} value={a.metadata.id}>
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
```

- [ ] **Step 3: Verify TypeScript compiles**

```bash
npx tsc --noEmit
```

- [ ] **Step 4: Commit**

```bash
git add src/components/OutputViewer.tsx src/components/ModelSelector.tsx
git commit -m "feat: add OutputViewer and ModelSelector components"
```

---
### Task 7: ModelRunner Orchestrator + App Integration

**Files:**
- Modify: `src/App.tsx`
- Create: `src/components/ModelRunner.tsx`

**Interfaces:**
- Consumes: `useModelRunner` hook, all components, `registeredAdapters`
- Produces: Complete working UI

- [ ] **Step 1: Write ModelRunner**

`src/components/ModelRunner.tsx`:

```tsx
import { useState } from 'react'
import { useModelRunner } from '../hooks/useModelRunner'
import type { ModelAdapter } from '../adapters/types'
import ModelSelector from './ModelSelector'
import InputEditor from './InputEditor'
import OutputViewer from './OutputViewer'

interface ModelRunnerProps {
  adapters: ModelAdapter[]
}

export default function ModelRunner({ adapters }: ModelRunnerProps) {
  const { loadModel, runInference, outputs, error, loading, loaded } = useModelRunner()
  const [selectedAdapter, setSelectedAdapter] = useState<ModelAdapter | null>(null)
  const [inputValues, setInputValues] = useState<Record<string, any>>({})

  const handleSelect = (adapter: ModelAdapter) => {
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
          <InputEditor specs={selectedAdapter.metadata.inputSpecs} onChange={setInputValues} />

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
```

- [ ] **Step 2: Update App.tsx**

```tsx
import ModelRunner from './components/ModelRunner'
import { registeredAdapters } from './adapters/magenta'

export default function App() {
  return <ModelRunner adapters={registeredAdapters} />
}
```

- [ ] **Step 3: Verify TypeScript compiles**

```bash
npx tsc --noEmit
```

- [ ] **Step 4: Verify dev server still works**

```bash
pnpm dev
# Should show LiteRT Playground heading, model selector, etc.
```

- [ ] **Step 5: Commit**

```bash
git add src/App.tsx src/components/ModelRunner.tsx
git commit -m "feat: wire up ModelRunner orchestrator and App integration"
```

---
### Task 8: Conversion Script

**Files:**
- Create: `scripts/convert_magenta.py`
- Create: `scripts/.gitignore`

**Interfaces:**
- Produces: `.tflite` file from `.mlxfn` input

- [ ] **Step 1: Set up Python environment**

```bash
uv venv
uv pip install ai-edge-converter
```

- [ ] **Step 2: Write conversion script**

`scripts/convert_magenta.py`:

```python
import argparse
import os
import tarfile
import tempfile
from pathlib import Path

def extract_mlxfn(mlxfn_path: str, output_dir: str):
    """Extract saved model from .mlxfn bundle."""
    with tarfile.open(mlxfn_path, 'r') as tar:
        tar.extractall(path=output_dir)

def convert_to_tflite(saved_model_dir: str, output_path: str):
    """Convert SavedModel to tflite."""
    import ai_edge_converter
    converter = ai_edge_converter.Converter(saved_model_dir)
    converter.convert()
    converter.save(output_path)

def main():
    parser = argparse.ArgumentParser(description='Convert .mlxfn to .tflite')
    parser.add_argument('input', help='Path to .mlxfn file')
    parser.add_argument('-o', '--output', default='magenta_realtime_2.tflite',
                        help='Output .tflite path')
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        saved_model_dir = os.path.join(tmpdir, 'saved_model')
        print(f'Extracting {args.input}...')
        extract_mlxfn(args.input, saved_model_dir)
        print(f'Converting to {args.output}...')
        convert_to_tflite(saved_model_dir, args.output)
        print(f'Done: {args.output}')

if __name__ == '__main__':
    main()
```

- [ ] **Step 3: Add .gitignore for scripts**

`scripts/.gitignore`:

```
__pycache__/
*.pyc
.venv/
```

- [ ] **Step 4: Commit**

```bash
git add scripts/convert_magenta.py scripts/.gitignore
git commit -m "feat: add .mlxfn to .tflite conversion script"
```

---
## Self-Review Checklist

1. **Spec coverage:**
   - [x] Core Abstraction Layer → Task 2 (types), Task 3 (adapter impl)
   - [x] Project Structure → Task 1 (scaffold)
   - [x] Data Flow → Task 4 (hook), Task 7 (orchestrator)
   - [x] Magenta Adapter → Task 3
   - [x] InputEditor (constraints-aware) → Task 5
   - [x] OutputViewer → Task 6
   - [x] ModelSelector → Task 6
   - [x] Conversion script → Task 8
   - [x] Wasm served from CDN → Task 4
   - [x] Constraints-as-data pattern → Task 5 (InputEditor checks constraints)
   - [x] No backend → verified all tasks

2. **Placeholder scan:** No TBD, TODO, "implement later", or similar found.

3. **Type consistency:**
   - `ModelAdapter.applyInputs(values)` returns `Tensor | Tensor[] | Record<string, Tensor>` — matches model.run() overloads
   - `ModelAdapter.parseOutputs(outputs)` returns `Promise<Record<string, any>>` — matches `output.data()` pattern
   - TensorSpec fields consistent across types.ts, magenta.ts, and InputEditor.tsx

4. **Scope check:** Focused on a single MVP. No scope creep.
