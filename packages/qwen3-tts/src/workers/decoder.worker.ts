/// <reference lib="webworker" />
import { createLiteRtRuntime } from '@litert-playground/runtime-litert';
import { createCachingAssetResolver, createHttpAssetResolver } from '@litert-playground/inference-core';
import { DecoderPhase } from '../phases/decoder';
import type { DecoderWorkerRequest, DecoderWorkerResponse } from './protocol';
import { serializeError } from './protocol';

let phase: DecoderPhase | undefined;

async function buildContext(modelBase: string) {
  const assets = createCachingAssetResolver(createHttpAssetResolver(modelBase));
  return createLiteRtRuntime({ assets });
}

self.onmessage = async (event: MessageEvent<DecoderWorkerRequest>) => {
  const req = event.data;
  try {
    if (req.type === 'initialize') {
      const context = await buildContext(req.modelBase);
      phase = new DecoderPhase(req.variant);
      await phase.load(context);
      self.postMessage({ type: 'ready' } satisfies DecoderWorkerResponse);
      return;
    }
    if (req.type === 'decode') {
      if (!phase) throw new Error('decoder not initialized');
      const audio = await phase.decode(req.frames);
      const phaseReceipt = {
        name: phase.name,
        backend: 'wasm' as const,
        loadMs: phase.loadMs,
        compileMs: phase.compileMs,
        inferenceMs: phase.inferenceMs,
      };
      self.postMessage(
        { type: 'audio', requestId: req.requestId, audio, phaseReceipt } satisfies DecoderWorkerResponse,
        [audio.buffer],
      );
      return;
    }
  } catch (cause) {
    const requestId = 'requestId' in req ? req.requestId : undefined;
    self.postMessage({ type: 'error', requestId, error: serializeError(cause) } satisfies DecoderWorkerResponse);
  }
};

export {};
