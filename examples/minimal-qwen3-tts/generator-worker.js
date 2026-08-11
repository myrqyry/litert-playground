// Classic worker shell for the generator phase. Served verbatim by litertWasmProxy
// so self.location is a real http URL (Emscripten resolves the wasm relative to it).
// The module bridge is loaded via dynamic import() (legal in classic workers).
(async () => {
  const origin = self.location.origin;
  const moduleUrl = `${origin}/packages/qwen3-tts/src/workers/generator.worker.ts`;
  try {
    await import(moduleUrl);
  } catch (cause) {
    self.postMessage({ type: 'error', error: { code: 'WORKER_BOOTSTRAP_FAILED', message: String(cause), stage: 'worker' } });
  }
})();
