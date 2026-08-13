export interface LiteRtLmWorkerMessage {
  role: 'user' | 'assistant' | 'model';
  content: string;
}

export interface LiteRtLmWorkerGenerationConfig {
  temperature?: number;
  maxOutputTokens?: number;
  topK?: number;
  topP?: number;
  seed?: number;
  maxContextTokens?: number;
  history?: LiteRtLmWorkerMessage[];
  systemPrompt?: string;
}

export type LiteRtLmWorkerRequest =
  | { type: 'load'; model: string | Blob }
  | { type: 'generate'; id: string; prompt: string; config?: LiteRtLmWorkerGenerationConfig }
  | { type: 'cancel'; id: string }
  | { type: 'dispose' };

export type LiteRtLmWorkerResponse =
  | { type: 'ready' }
  | { type: 'token'; id: string; text: string }
  | { type: 'reasoning'; id: string; text: string }
  | { type: 'complete'; id: string }
  | { type: 'error'; id?: string; message: string }
  | { type: 'disposed' };
