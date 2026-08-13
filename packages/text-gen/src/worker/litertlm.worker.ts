import type { Message, SamplerParameters, SessionConfig } from '@litert-lm/core';
import type { ConversationConfig } from '@litert-lm/core';
import type {
  LiteRtLmWorkerGenerationConfig,
  LiteRtLmWorkerMessage,
  LiteRtLmWorkerRequest,
  LiteRtLmWorkerResponse,
} from './protocol';

interface WorkerScope {
  onmessage: ((event: MessageEvent) => void) | null;
  postMessage(message: unknown): void;
}

interface LiteRtLmModule {
  Engine: {
    create: (settings: {
      model: string | Blob | ReadableStream<Uint8Array>;
      mainExecutorSettings?: { maxNumTokens?: number };
    }) => Promise<LiteRtLmEngine>;
  };
}

interface LiteRtLmEngine {
  createConversation(config?: ConversationConfig): Promise<LiteRtLmConversation>;
  delete(): Promise<void>;
}

interface LiteRtLmConversation {
  sendMessage(message: Message): Promise<{ text?: string }>;
  sendMessageStreaming(message: Message): ReadableStream<Message>;
  cancel(): void;
  delete(): Promise<void>;
}

type Conversation = Awaited<ReturnType<LiteRtLmEngine['createConversation']>>;

const worker = self as unknown as WorkerScope;

let engine: LiteRtLmEngine | undefined;
const activeConversations = new Map<string, Conversation>();

function extractText(message: Message): string {
  if (typeof message.content === 'string') return message.content;
  if (Array.isArray(message.content)) {
    return message.content
      .filter((part) => part.type === 'text' && typeof part.text === 'string')
      .map((part) => (part as { text: string }).text)
      .join('');
  }
  return '';
}

function extractReasoning(message: Message): string {
  return message.channels?.reasoning ?? message.channels?.thought ?? message.channels?.think ?? '';
}

function emit(response: LiteRtLmWorkerResponse): void {
  worker.postMessage(response);
}

function buildConversationConfig(config?: LiteRtLmWorkerGenerationConfig): ConversationConfig | undefined {
  const conversationConfig: ConversationConfig = {};
  const hasSampler =
    config?.temperature !== undefined ||
    config?.topK !== undefined ||
    config?.topP !== undefined ||
    config?.seed !== undefined;
  if (hasSampler || config?.maxOutputTokens !== undefined) {
    const sessionConfig: SessionConfig = {};
    if (hasSampler) {
      const samplerParams: SamplerParameters = {};
      if (config?.temperature !== undefined) samplerParams.temperature = config.temperature;
      if (config?.topK !== undefined) samplerParams.k = config.topK;
      if (config?.topP !== undefined) samplerParams.p = config.topP;
      if (config?.seed !== undefined) samplerParams.seed = config.seed;
      sessionConfig.samplerParams = samplerParams;
    }
    if (config?.maxOutputTokens !== undefined) sessionConfig.maxOutputTokens = config.maxOutputTokens;
    conversationConfig.sessionConfig = sessionConfig;
  }
  if (config?.systemPrompt?.trim()) {
    conversationConfig.preface = { messages: [{ role: 'system', content: config.systemPrompt }] };
  }
  return Object.keys(conversationConfig).length > 0 ? conversationConfig : undefined;
}

async function replayHistory(conversation: Conversation, history?: LiteRtLmWorkerMessage[]): Promise<void> {
  if (!history?.length) return;
  // LiteRT-LM populates the conversation KV cache on each send, so history
  // must be replayed in order rather than in parallel.
  for (const message of history) {
    await conversation.sendMessage({
      role: message.role === 'assistant' ? 'model' : message.role,
      content: message.content,
    });
  }
}

async function streamResponse(stream: ReadableStream<Message>, onMessage: (message: Message) => void): Promise<void> {
  if (Symbol.asyncIterator in stream) {
    for await (const chunk of stream) onMessage(chunk);
  } else {
    const reader = (stream as ReadableStream<Message>).getReader();
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        onMessage(value);
      }
    } finally {
      reader.releaseLock();
    }
  }
}

async function disposeConversation(id: string, conversation: Conversation): Promise<void> {
  if (activeConversations.get(id) !== conversation) return;
  activeConversations.delete(id);
  await conversation.delete();
}

async function generate(id: string, prompt: string, config?: LiteRtLmWorkerGenerationConfig): Promise<void> {
  if (!engine) throw new Error('LiteRT-LM is not loaded');
  const conversation = await engine.createConversation(buildConversationConfig(config));
  activeConversations.set(id, conversation);
  try {
    await replayHistory(conversation, config?.history);
    await streamResponse(conversation.sendMessageStreaming({ role: 'user', content: prompt }), (message) => {
      const reasoning = extractReasoning(message);
      if (reasoning) emit({ type: 'reasoning', id, text: reasoning });
      const text = extractText(message);
      if (text) emit({ type: 'token', id, text });
    });
    emit({ type: 'complete', id });
  } finally {
    await disposeConversation(id, conversation);
  }
}

async function disposeAllConversations(): Promise<void> {
  for (const [id, conversation] of activeConversations) {
    conversation.cancel();
    await conversation.delete();
  }
  activeConversations.clear();
}

worker.onmessage = async (event: MessageEvent<LiteRtLmWorkerRequest>) => {
  const data = event.data;
  try {
    switch (data.type) {
      case 'load':
        await disposeAllConversations();
        await engine?.delete();
        const module = (await import('@litert-lm/core')) as unknown as LiteRtLmModule;
        engine = await module.Engine.create({
          model: data.model,
          mainExecutorSettings: { maxNumTokens: 8192 },
        });
        emit({ type: 'ready' });
        break;
      case 'generate':
        await generate(data.id, data.prompt, data.config);
        break;
      case 'cancel':
        activeConversations.get(data.id)?.cancel();
        break;
      case 'dispose':
        await disposeAllConversations();
        await engine?.delete();
        engine = undefined;
        emit({ type: 'disposed' });
        break;
    }
  } catch (error) {
    emit({
      type: 'error',
      id: data.type === 'generate' ? data.id : undefined,
      message: error instanceof Error ? error.message : String(error),
    });
  }
};
