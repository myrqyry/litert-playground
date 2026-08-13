import {
  type ModelManifest,
  type Pipeline,
  type PipelineProgress,
  type PipelineStatus,
  type RuntimeContext,
  type TextInferenceResult,
} from '@litert-playground/inference-core';
import {
  type TextGenerationConfig,
  type TextGenerationInput,
  type TextMessage,
} from './types';
import { litertLmManifest } from './manifest';

interface LiteRtLmModule {
  Engine: {
    create: (settings: {
      model: string | Blob | ReadableStream<Uint8Array>;
      backend?: 'webgpu' | 'wasm' | 'cpu';
      mainExecutorSettings?: { maxNumTokens?: number };
    }) => Promise<LiteRtLmEngine>;
  };
}

interface LiteRtLmEngine {
  createConversation(config?: { maxContextTokens?: number }): Promise<LiteRtLmConversation>;
  delete(): Promise<void>;
}

interface LiteRtLmConversation {
  sendMessage(
    message: { role: string; content: string } | Array<{ role: string; content: string }>,
  ): Promise<{ text?: string }>;
  sendMessageStreaming(
    message: { role: string; content: string } | Array<{ role: string; content: string }>,
  ): ReadableStream<{ text?: string }>;
  cancel(): void;
  delete(): Promise<void>;
}

export interface LiteRtLmTextConfig extends TextGenerationConfig {
  model: string | Blob | ReadableStream<Uint8Array>;
  maxContextTokens?: number;
}

const DEFAULTS: Pick<LiteRtLmTextConfig, 'model' | 'maxContextTokens'> = {
  model: 'litert-community/Qwen3-0.6B/resolve/main/Qwen3-0.6B.litertlm',
  maxContextTokens: 4096,
};function toLiteRtMessages(input: TextGenerationInput): Array<{ role: string; content: string }> {
  const messages: Array<{ role: string; content: string }> = [];
  if (input.systemPrompt) {
    messages.push({ role: 'system', content: input.systemPrompt });
  }
  for (const msg of input.messages) {
    const role = msg.role === 'model' ? 'assistant' : msg.role;
    messages.push({ role, content: msg.content });
  }
  return messages;
}

async function readStream(
  stream: ReadableStream<{ text?: string }>,
  signal?: AbortSignal,
): Promise<string> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let full = '';
  try {
    while (true) {
      if (signal?.aborted) throw new Error('CANCELLED');
      const { done, value } = await reader.read();
      if (done) break;
      if (value?.text) full += value.text;
    }
  } finally {
    reader.releaseLock();
  }
  void decoder;
  return full;
}

export class LiteRtLmTextPipeline
  implements Pipeline<TextGenerationInput, TextInferenceResult, LiteRtLmTextConfig>
{
  readonly manifest: ModelManifest;
  status: PipelineStatus = 'idle';
  onProgress?: (progress: PipelineProgress) => void;

  private context: RuntimeContext | null = null;
  private engine: LiteRtLmEngine | null = null;
  private conversation: LiteRtLmConversation | null = null;
  private loadMs = 0;

  constructor(manifest: ModelManifest = litertLmManifest) {
    this.manifest = manifest;
  }

  async load(context: RuntimeContext): Promise<void> {
    if (this.status === 'ready') return;
    this.status = 'loading';
    this.context = context;
    const loadStart = performance.now();
    try {
      this.report({ phase: 'loading', step: 1, total: 2 });
      const module = (await import('@litert-lm/core')) as unknown as LiteRtLmModule;
      this.report({ phase: 'loading', step: 2, total: 2 });
      const model =
        this.manifest.assets.find((a) => a.id === 'model')?.path ??
        this.manifest.assets[0]?.path ??
        DEFAULTS.model;
      const backend = context.backend === 'webnn' ? undefined : context.backend;
      this.engine = await module.Engine.create({
        model,
        backend,
        mainExecutorSettings: { maxNumTokens: DEFAULTS.maxContextTokens },
      });
      this.loadMs = performance.now() - loadStart;
      this.status = 'ready';
    } catch (e) {
      this.status = 'error';
      throw e instanceof Error ? e : new Error(String(e));
    }
  }

  async run(
    input: TextGenerationInput,
    config?: LiteRtLmTextConfig,
    signal?: AbortSignal,
  ): Promise<TextInferenceResult> {
    if (this.status !== 'ready') throw new Error('Pipeline not ready');
    if (!this.engine) throw new Error('LiteRT-LM pipeline not loaded');
    this.status = 'running';
    const cfg = { ...DEFAULTS, ...config };
    try {
      if (signal?.aborted) throw new Error('CANCELLED');
      const messages = toLiteRtMessages(input);
      this.conversation = await this.engine.createConversation({
        maxContextTokens: cfg.maxContextTokens,
      });
      const prompt = messages.pop()!;
      for (const msg of messages) {
        if (signal?.aborted) throw new Error('CANCELLED');
        await this.conversation.sendMessage(msg);
      }
      const stream = this.conversation.sendMessageStreaming(prompt);
      const text = await readStream(stream, signal);
      this.status = 'ready';
      return {
        kind: 'text',
        text,
      } satisfies TextInferenceResult;
    } catch (e) {
      this.status = 'ready';
      throw e instanceof Error ? e : new Error(String(e));
    }
  }

  async dispose(): Promise<void> {
    await this.conversation?.delete();
    this.conversation = null;
    await this.engine?.delete();
    this.engine = null;
    this.context = null;
    this.status = 'disposed';
  }

  private report(progress: PipelineProgress): void {
    this.onProgress?.(progress);
  }
}

export type { TextGenerationConfig, TextGenerationInput, TextMessage };
