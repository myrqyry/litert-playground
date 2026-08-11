/// <reference lib="webworker" />
import { createLiteRtRuntime } from '@litert-playground/runtime-litert';
import { createCachingAssetResolver, createHttpAssetResolver } from '@litert-playground/inference-core';
import { GeneratorPhase } from '../phases/generator';
import type { GeneratorWorkerRequest, GeneratorWorkerResponse } from './protocol';
import { serializeError } from './protocol';

let phase: GeneratorPhase | undefined;

async function buildContext(modelBase: string) {
  const assets = createCachingAssetResolver(createHttpAssetResolver(modelBase));
  return createLiteRtRuntime({ assets });
}

self.onmessage = async (event: MessageEvent<GeneratorWorkerRequest>) => {
  const req = event.data;
  try {
    if (req.type === 'initialize') {
      const context = await buildContext(req.modelBase);
      phase = new GeneratorPhase(req.variant, {
        onProgress: (progress) => {
          self.postMessage({ type: 'progress', progress } satisfies GeneratorWorkerResponse);
        },
      });
      await phase.load(context);
      self.postMessage({ type: 'ready' } satisfies GeneratorWorkerResponse);
      return;
    }
    if (req.type === 'generate') {
      if (!phase) throw new Error('generator not initialized');
      const frames = await phase.generate(req.input, req.config);
      const phaseReceipt = {
        name: phase.name,
        backend: 'wasm' as const,
        loadMs: phase.loadMs,
        compileMs: phase.compileMs,
        inferenceMs: phase.inferenceMs,
      };
      self.postMessage(
        { type: 'frames', requestId: req.requestId, frames, phaseReceipt } satisfies GeneratorWorkerResponse,
        [frames.frames.buffer],
      );
      return;
    }
    if (req.type === 'cancel') {
      // Host terminates the worker for cancellation; nothing to do here.
      return;
    }
  } catch (cause) {
    const requestId = 'requestId' in req ? req.requestId : undefined;
    self.postMessage({ type: 'error', requestId, error: serializeError(cause) } satisfies GeneratorWorkerResponse);
  }
};

export {};
