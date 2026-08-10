import type { AudioInferenceResult } from '@litert-playground/inference-core';
import type { KokoroPipeline } from '@litert-playground/kokoro';
import type { Qwen3TtsPipeline } from '@litert-playground/qwen3-tts';

export interface PodcastTts {
  synthesize(text: string, voice: string, signal?: AbortSignal): Promise<AudioInferenceResult>;
}

export class KokoroPodcastTts implements PodcastTts {
  constructor(private readonly pipeline: KokoroPipeline) {}

  synthesize(text: string, voice: string, signal?: AbortSignal): Promise<AudioInferenceResult> {
    return this.pipeline.run({ text }, { voice }, signal);
  }
}

export class Qwen3PodcastTts implements PodcastTts {
  constructor(private readonly pipeline: Qwen3TtsPipeline) {}

  synthesize(text: string, voice: string, signal?: AbortSignal): Promise<AudioInferenceResult> {
    return this.pipeline.run({ text }, { voice }, signal);
  }
}

export class CloudPodcastTts implements PodcastTts {
  constructor(
    private readonly delegate: (
      text: string,
      voice: string,
      signal?: AbortSignal,
    ) => Promise<AudioInferenceResult>,
  ) {}

  synthesize(text: string, voice: string, signal?: AbortSignal): Promise<AudioInferenceResult> {
    return this.delegate(text, voice, signal);
  }
}
