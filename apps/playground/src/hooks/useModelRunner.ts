import { useState, useCallback, useRef } from 'react'
import { loadLiteRt, loadAndCompile, Tensor } from '@litertjs/core'
import type { ModelAdapter, TensorSpec } from '../adapters/types'

const WASM_URL = 'https://cdn.jsdelivr.net/npm/@litertjs/core/wasm/'

export type Accelerator = 'webgpu' | 'wasm' | 'webnn'

export interface RawTensor {
  data: Float32Array
  shape: number[]
}

interface UseModelRunnerReturn {
  loadModel: (adapter: ModelAdapter, accelerator?: Accelerator) => Promise<void>
  runInference: (values: Record<string, any>) => Promise<void>
  outputs: Record<string, any> | null
  outputTensors: Record<string, RawTensor> | null
  outputSpecs: TensorSpec[]
  accelerator: Accelerator
  setAccelerator: (a: Accelerator) => void
  error: string | null
  loading: boolean
  loaded: boolean
}

export function useModelRunner(): UseModelRunnerReturn {
  const [outputs, setOutputs] = useState<Record<string, any> | null>(null)
  const [outputTensors, setOutputTensors] = useState<Record<string, RawTensor> | null>(null)
  const [outputSpecs, setOutputSpecs] = useState<TensorSpec[]>([])
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [loaded, setLoaded] = useState(false)
  const [accelerator, setAccelerator] = useState<Accelerator>('webgpu')
  const modelRef = useRef<any>(null)
  const adapterRef = useRef<ModelAdapter | null>(null)
  const runtimeRef = useRef(false)

  const loadModel = useCallback(async (adapter: ModelAdapter, acc?: Accelerator) => {
    const targetAcc = acc ?? accelerator
    setLoading(true)
    setError(null)
    setLoaded(false)
    try {
      if (!runtimeRef.current) {
        await loadLiteRt(WASM_URL, { jspi: targetAcc === 'webnn' })
        runtimeRef.current = true
      }

      const model = await loadAndCompile(adapter.metadata.modelPath, {
        accelerator: targetAcc,
        webNNOptions: targetAcc === 'webnn' ? { devicePreference: 'npu' } : undefined,
      })
      modelRef.current = model
      adapterRef.current = adapter
      setLoaded(true)
    } catch (e: any) {
      if (targetAcc !== 'wasm') {
        try {
          const model = await loadAndCompile(adapter.metadata.modelPath, { accelerator: 'wasm' })
          modelRef.current = model
          adapterRef.current = adapter
          setAccelerator('wasm')
          setLoaded(true)
          return
        } catch {}
      }
      setError(e.message ?? String(e))
    } finally {
      setLoading(false)
    }
  }, [accelerator])

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

      const raw: Record<string, RawTensor> = {}
      for (const spec of adapterRef.current.outputSpecs) {
        const tensor = result[spec.name]
        if (tensor) {
          const arr = await (tensor as Tensor).data()
          raw[spec.name] = { data: new Float32Array(arr), shape: spec.shape }
        }
      }
      setOutputTensors(raw)
      setOutputSpecs(adapterRef.current.outputSpecs)

      Object.values(inputs).forEach((t: any) => t?.delete?.())
    } catch (e: any) {
      setError(e.message ?? String(e))
    } finally {
      setLoading(false)
    }
  }, [])

  return { loadModel, runInference, outputs, outputTensors, outputSpecs, accelerator, setAccelerator, error, loading, loaded }
}
