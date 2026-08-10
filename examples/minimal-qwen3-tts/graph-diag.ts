import { loadAndCompile, loadLiteRt } from '@litertjs/core'

export interface GraphDiagStep {
  name: string
  path: string
  status: 'pending' | 'fetch' | 'compiling' | 'ok' | 'fail'
  fetchMs?: number
  compileMs?: number
  fetchedBytes?: number
  totalBytes?: number
  jsHeapUsedMB?: number
  jsHeapTotalMB?: number
  error?: string
}

export interface GraphDiagResult {
  ok: boolean
  runtimeMs: number
  backend: string
  steps: GraphDiagStep[]
  finishedAt: string
}

const WASM_BASE = 'https://cdn.jsdelivr.net/npm/@litertjs/core@2.5.3/wasm/'
const MODEL_PREFIX = '/models/qwen3-tts/'

async function fetchBytes(url: string): Promise<{ bytes: Uint8Array; fetchMs: number; totalBytes?: number }> {
  const started = performance.now()
  const response = await fetch(url)
  if (!response.ok || !response.body) throw new Error(`HTTP ${response.status}`)
  const totalBytes = Number(response.headers.get('content-length')) || undefined
  const reader = response.body.getReader()
  const chunks: Uint8Array[] = []
  let loaded = 0
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    chunks.push(value)
    loaded += value.byteLength
    if (loaded % 100_000_000 < 4_000_000 || loaded === totalBytes) {
      console.log(`diag: ${url.split('/').pop()} fetch ${loaded}/${totalBytes ?? '?'}`)
    }
  }
  const bytes = new Uint8Array(loaded)
  let offset = 0
  for (const chunk of chunks) {
    bytes.set(chunk, offset)
    offset += chunk.byteLength
  }
  return { bytes, fetchMs: Math.round(performance.now() - started), totalBytes }
}

function jsHeap(): { jsHeapUsedMB: number; jsHeapTotalMB: number } {
  const m = (performance as unknown as { memory?: { usedJSHeapSize: number; totalJSHeapSize: number } }).memory
  if (!m) return { jsHeapUsedMB: 0, jsHeapTotalMB: 0 }
  return { jsHeapUsedMB: Math.round(m.usedJSHeapSize / 1048576), jsHeapTotalMB: Math.round(m.totalJSHeapSize / 1048576) }
}

export async function runGraphDiag(only?: string): Promise<GraphDiagResult> {
  const steps: GraphDiagStep[] = []
  const started = performance.now()
  try {
    await loadLiteRt(WASM_BASE, { jspi: true })
  } catch (cause) {
    return {
      ok: false,
      runtimeMs: Math.round(performance.now() - started),
      backend: 'unknown',
      steps: [{ name: 'runtime', path: WASM_BASE, status: 'fail', error: String(cause) }],
      finishedAt: new Date().toISOString(),
    }
  }
  const runtimeMs = Math.round(performance.now() - started)

  const graphs: Array<{ name: string; path: string }> = [
    { name: 'talker', path: 'talker_int4.tflite' },
    { name: 'mtp', path: 'mtp_fp32.tflite' },
    { name: 'codec', path: 'codec_decoder_fp32.tflite' },
  ].filter((graph) => only === undefined || graph.name === only)

  for (const graph of graphs) {
    const step: GraphDiagStep = { name: graph.name, path: graph.path, status: 'fetch' }
    steps.push(step)
    console.log(`diag: ${graph.name} fetch start (${graph.path})`)
    try {
      const { bytes, fetchMs, totalBytes } = await fetchBytes(MODEL_PREFIX + graph.path)
      step.fetchMs = fetchMs
      step.fetchedBytes = bytes.byteLength
      step.totalBytes = totalBytes
      step.status = 'compiling'
      console.log(`diag: ${graph.name} fetch complete ${bytes.byteLength} bytes in ${fetchMs}ms`)
      const compileStart = performance.now()
      console.log(`diag: ${graph.name} compile start heap=${JSON.stringify(jsHeap())}`)
      await loadAndCompile(bytes, { accelerator: 'wasm' })
      step.compileMs = Math.round(performance.now() - compileStart)
      const heap = jsHeap()
      step.jsHeapUsedMB = heap.jsHeapUsedMB
      step.jsHeapTotalMB = heap.jsHeapTotalMB
      step.status = 'ok'
      console.log(`diag: ${graph.name} compile complete in ${step.compileMs}ms heap=${JSON.stringify(heap)}`)
    } catch (cause) {
      step.status = 'fail'
      step.error = cause instanceof Error ? cause.message : String(cause)
      console.log(`diag: ${graph.name} FAILED: ${step.error}`)
      return {
        ok: false,
        runtimeMs,
        backend: 'wasm',
        steps,
        finishedAt: new Date().toISOString(),
      }
    }
  }

  return {
    ok: true,
    runtimeMs,
    backend: 'wasm',
    steps,
    finishedAt: new Date().toISOString(),
  }
}
