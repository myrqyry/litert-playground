export interface QwenTtsInput {
  text: string;
}

export interface QwenTtsConfig {
  temperature?: number;
  topK?: number;
  repetitionPenalty?: number;
  voice?: string;
  maxFrames?: number;
  language?: string;
}

export const DEFAULTS: Required<QwenTtsConfig> = {
  temperature: 0.85,
  topK: 25,
  repetitionPenalty: 1.05,
  voice: 'demo_speaker',
  maxFrames: 512,
  language: 'english',
};

export const HIDDEN = 1024;
export const CODEC_VOCAB = 3072;
export const CODEC_EOS = 2150;
export const NEG_INF = -1e9;

export const LANGUAGE_IDS: Record<string, number> = {
  english: 2050,
  chinese: 2055,
  japanese: 2058,
  korean: 2064,
  german: 2053,
  french: 2061,
  spanish: 2054,
  italian: 2070,
  portuguese: 2071,
  russian: 2069,
};
