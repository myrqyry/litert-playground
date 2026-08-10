import type { AudioInferenceResult } from '@litert-playground/inference-core';
import type { KokoroPipeline } from "./kokoro/pipeline";
import type { Qwen3TtsPipeline } from "./qwen3-tts/pipeline";

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
