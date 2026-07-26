# LiteRT Playground — Design Doc

## 1. Purpose

A browser-based playground for running LiteRT (.tflite) models in-browser using WebGPU/Wasm. Users load models, configure inputs, run inference, and inspect outputs — all client-side, no backend.

## 2. Core Abstraction Layer

Every model is wrapped by a `ModelAdapter` interface:

```typescript
interface ModelAdapter {
  modelId: string
  modelPath: string
  metadata: ModelMetadata
  inputSpecs: TensorSpec[]
  outputSpecs: TensorSpec[]
  load(model: tflite.Model): Promise<void>
  applyInputs(values: Record<string, any>, session: tflite.Session): Promise<void>
  parseOutputs(session: tflite.Session): Promise<Record<string, any>>
}

interface TensorSpec {
  name: string
  dtype: 'float32' | 'int32' | 'int8' | 'uint8'
  shape: number[]
  description: string
  constraints?: {
    min?: number
    max?: number
    enum?: string[]  // e.g. instrument names for index values
    items?: string[] // human-readable labels for enum indices
  }
}
```

The adapter knows what tensors its model expects and how to translate raw tensor data to/from meaningful JSON. The runner just calls `adapter.createInputs(session)` and `adapter.parseOutputs(session)`.

## 3. Project Structure

```
litert-playground/
├── package.json
├── tsconfig.json
├── vite.config.ts
├── index.html
├── src/
│   ├── main.tsx              # App entry
│   ├── App.tsx               # Router/layout
│   ├── adapters/
│   │   ├── types.ts          # ModelAdapter, TensorSpec interfaces
│   │   └── magenta.ts        # MagentaRealtime2 adapter
│   ├── components/
│   │   ├── ModelSelector.tsx # Dropdown of available models
│   │   ├── InputEditor.tsx   # JSON editors for model inputs
│   │   ├── OutputViewer.tsx  # Formatted JSON output display
│   │   └── ModelRunner.tsx   # Load -> configure -> run -> view
│   └── hooks/
│       └── useModelRunner.ts # Core inference lifecycle
├── public/
│   └── models/               # .tflite files go here (gitignored)
└── scripts/
    └── convert_magenta.py    # .mlxfn -> .tflite conversion
```

## 4. Data Flow

```
User selects model
  → Load .tflite from public/models/
  → Adapter exposes inputSpecs/outputSpecs
  → InputEditor renders JSON fields from inputSpecs
  → User edits input JSON
  → ModelRunner creates session, maps JSON -> tensors via adapter.createInputs()
  → Run inference
  → adapter.parseOutputs(session) returns JSON
  → OutputViewer renders formatted JSON
```

## 5. MVP: Magenta Realtime 2 Adapter

The Magenta adapter maps these tensors:

**Inputs:**
- `input` (float32, [1, 256, 1]) — audio frame, mean-centered normalized
- `length` (float32, [1]) — generation length in seconds
- `temperature_harmonic` (float32, [1]) — harmonic temperature (tied to `nsynth_temperature` from original)

**Outputs:**
- `output` (float32, [1, 256, 1]) — generated audio frame
- `state` (float32, [1, 256]) — recurrent state

Input is one 256-sample frame at a time. For real use, the user feeds previous output as next input. The MVP shows one step — the user provides `input`, the model returns `output`. The edit boxes are plain JSON editors with clear labels/descriptions from the TensorSpec.

**Constraints system** — the TensorSpec has a `constraints` field. The UI checks it before inference:
- `min`/`max` → clamp the value
- `enum` → show a dropdown instead of free-text
- `items` → label the enum indices
This is the "constraints are easy to add when discovered" mechanism — the adapter author just fills in the constraint fields and the UI automatically adapts.

### Conversion

The `.mlxfn` format is Magenta's bundled model format (protobuf + weights). Conversion needs:
- Extract the SavedModel from `.mlxfn` (it's a tar-like bundle)
- `tflite_convert` from TF or `ai-edge-converter` to produce `.tflite`
- The conversion script (`scripts/convert_magenta.py`) will handle the full pipeline

## 6. Constraints for MVP (What We're Not Building)

- **Smart controls** (sliders, audio scrubbers, waveform viz) — add when we know the real range of each tensor
- **Model upload UI** — users manually place .tflite in `public/models/`
- **Multi-model** — model selector exists but only shows Magenta at launch
- **Audio playback** — no WebAudio integration yet
- **Session management** — one model at a time
- **Backend** — everything in-browser

## 7. Tech Stack

| Tool | Use |
|------|-----|
| pnpm | Package manager |
| Vite + React + TypeScript | Frontend |
| @litertjs/core | TFLite inference (WebGPU/Wasm) |
| uv | Python venv for conversion script |
| ai-edge-converter | .mlxfn -> .tflite conversion |

## 8. Constraints-as-Data Pattern

The key architectural insight: controls derive from adapter data, not hardcoded UIs. When an adapter's TensorSpec has `constraints.enum`, the InputEditor automatically shows a `<select>`. When constraints has `min`/`max`, it clamps. When absent, it shows a raw JSON editor.

This means adding a new model type = write one adapter file with filled-in TensorSpecs. No UI changes needed.
