self.onmessage = async (event) => {
  const { graphs } = event.data
  try {
    const { compileGraphs } = await import('http://localhost:5176/examples/minimal-qwen3-tts/graph-diag.ts')
    const result = await compileGraphs(graphs, '/litert-wasm/')
    self.postMessage({ type: 'result', result })
  } catch (cause) {
    self.postMessage({
      type: 'result',
      result: {
        ok: false,
        runtimeMs: 0,
        backend: 'unknown',
        steps: [{ name: 'worker', path: 'worker', status: 'fail', error: String(cause) }],
        finishedAt: new Date().toISOString(),
      },
    })
  }
}
