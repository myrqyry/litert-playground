import { describe, expect, it } from 'vitest';
import { packCodecFrames, unpackCodecFrames } from './codec-frames';

describe('codec-frames', () => {
  it('round-trips number[][] through the flat Uint16Array layout', () => {
    const frames = [
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ];
    const packed = packCodecFrames(frames);
    expect(packed.frameCount).toBe(3);
    expect(packed.codebooks).toBe(3);
    expect(Array.from(packed.frames)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9]);
    expect(unpackCodecFrames(packed)).toEqual(frames);
  });

  it('handles an empty frame list', () => {
    const packed = packCodecFrames([], 16);
    expect(packed.frameCount).toBe(0);
    expect(packed.frames.length).toBe(0);
    expect(unpackCodecFrames(packed)).toEqual([]);
  });
});
