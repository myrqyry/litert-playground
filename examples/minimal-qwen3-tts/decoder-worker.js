// Classic worker shell for the decoder phase. See generator-worker.js.
(async () => {
  const origin = self.location.origin;
  const moduleUrl = `${origin}/packages/qwen3-tts/src/workers/decoder.worker.ts`;
  try {
    await import(moduleUrl);
  } catch (cause) {
    self.postMessage({ type: 'error', error: { code: 'WORKER_BOOTSTRAP_FAILED', message: String(cause), stage: 'worker' } });
  }
})();
