import { beforeEach, describe, expect, it, vi } from "vitest";
import type { RuntimeContext } from '@litert-playground/inference-core';
import { TransformersTextPipeline } from "./transformers-pipeline";
import { LiteRtLmTextPipeline } from "./litertlm-pipeline";

const { mockPipeline, mockEngineCreate, mockSendMessageStreaming } = vi.hoisted(() => ({
  mockPipeline: vi.fn(),
  mockEngineCreate: vi.fn(),
  mockSendMessageStreaming: vi.fn(),
}));

vi.mock("@huggingface/transformers", () => ({
  pipeline: mockPipeline,
}));

vi.mock("@litert-lm/core", () => ({
  Engine: {
    create: mockEngineCreate,
  },
}));

function fakeContext(): RuntimeContext {
  return {
    backend: "wasm",
    assets: {} as RuntimeContext["assets"],
    liteRt: {} as RuntimeContext["liteRt"],
  };
}

function streamOf(chunks: string[]): ReadableStream<{ text: string }> {
  return new ReadableStream({
    start(controller) {
      for (const c of chunks) controller.enqueue({ text: c });
      controller.close();
    },
  });
}

beforeEach(() => {
  mockPipeline.mockReset();
  mockEngineCreate.mockReset();
  mockSendMessageStreaming.mockReset();

  mockPipeline.mockResolvedValue(async () => ({
    output: [{ generated_text: "Hello from transformers" }],
  }));

  mockEngineCreate.mockResolvedValue({
    createConversation: async () => ({
      sendMessage: async () => ({ text: "" }),
      sendMessageStreaming: mockSendMessageStreaming,
      cancel: () => {},
      delete: async () => {},
    }),
    delete: async () => {},
  });

  mockSendMessageStreaming.mockReturnValue(
    streamOf(["Hel", "lo", " from ", "litert-lm"])
  );
});

describe("TransformersTextPipeline", () => {
  it("throws on run before load", async () => {
    const p = new TransformersTextPipeline();
    await expect(
      p.run({ messages: [{ role: "user", content: "hi" }] })
    ).rejects.toThrow("Pipeline not ready");
  });

  it("has correct manifest", () => {
    const p = new TransformersTextPipeline();
    expect(p.manifest.capabilities).toContain("text-generation");
  });

  it("generates text via pipeline", async () => {
    const p = new TransformersTextPipeline();
    await p.load(fakeContext());

    const result = await p.run(
      { systemPrompt: "Be brief.", messages: [{ role: "user", content: "hi" }] },
      { modelId: "onnx-community/Qwen3-0.6B-ONNX", device: "wasm" }
    );

    expect(result.kind).toBe("text");
    expect(result.text).toBe("Hello from transformers");
    expect(mockPipeline).toHaveBeenCalledWith(
      "text-generation",
      "onnx-community/Qwen3-0.6B-ONNX",
      expect.objectContaining({ device: "wasm" })
    );
  });

  it("honors an aborted signal", async () => {
    const p = new TransformersTextPipeline();
    await p.load(fakeContext());

    const controller = new AbortController();
    controller.abort();

    await expect(
      p.run(
        { messages: [{ role: "user", content: "x" }] },
        { modelId: "onnx-community/Qwen3-0.6B-ONNX", device: "wasm" },
        controller.signal
      )
    ).rejects.toThrow("CANCELLED");
  });
});

describe("LiteRtLmTextPipeline", () => {
  it("throws on run before load", async () => {
    const p = new LiteRtLmTextPipeline();
    await expect(
      p.run({ messages: [{ role: "user", content: "hi" }] })
    ).rejects.toThrow("Pipeline not ready");
  });

  it("has correct manifest", () => {
    const p = new LiteRtLmTextPipeline();
    expect(p.manifest.capabilities).toContain("text-generation");
  });

  it("replays history then streams the prompt", async () => {
    const p = new LiteRtLmTextPipeline();
    await p.load(fakeContext());

    const result = await p.run(
      {
        systemPrompt: "Be brief.",
        messages: [
          { role: "user", content: "hi" },
          { role: "assistant", content: "yo" },
        ],
      },
      { model: "litert-community/Qwen3-0.6B/resolve/main/Qwen3-0.6B.litertlm" }
    );

    expect(result.kind).toBe("text");
    expect(result.text).toBe("Hello from litert-lm");
    expect(mockEngineCreate).toHaveBeenCalledWith(
      expect.objectContaining({
        mainExecutorSettings: { maxNumTokens: 4096 },
      })
    );
    expect(mockSendMessageStreaming).toHaveBeenCalled();
  });
});
