export interface CodecFrames {
  frames: Uint16Array;
  frameCount: number;
  codebooks: number;
}

export function packCodecFrames(allFrames: number[][], codebooks?: number): CodecFrames {
  const cb = codebooks ?? (allFrames.length > 0 ? allFrames[0].length : 16);
  const frames = new Uint16Array(allFrames.length * cb);
  for (let f = 0; f < allFrames.length; f++) {
    for (let c = 0; c < cb; c++) {
      frames[f * cb + c] = allFrames[f][c] ?? 0;
    }
  }
  return { frames, frameCount: allFrames.length, codebooks: cb };
}

export function unpackCodecFrames(frames: CodecFrames): number[][] {
  const out: number[][] = [];
  for (let f = 0; f < frames.frameCount; f++) {
    const row: number[] = [];
    for (let c = 0; c < frames.codebooks; c++) {
      row.push(frames.frames[f * frames.codebooks + c]);
    }
    out.push(row);
  }
  return out;
}
