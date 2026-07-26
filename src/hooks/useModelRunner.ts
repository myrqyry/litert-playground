import { useState, useCallback, useRef } from 'react'
import { loadLiteRt, loadAndCompile, Tensor } from '@litertjs/core'
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
      let inputs = adapterRef.current.prepareInputs(values)
      if (!Object.keys(inputs).length && Object.keys(values).length) {
        inputs = {}
        for (const spec of adapterRef.current.inputSpecs) {
          const v = values[spec.name]
          if (v instanceof Float32Array) inputs[spec.name] = new Tensor(v, spec.shape)
        }
      }
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
