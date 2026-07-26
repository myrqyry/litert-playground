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
