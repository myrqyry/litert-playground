const HIDDEN = 1024
const CODEC_VOCAB = 3072
const CODEC_PAD = 2148
const CODEC_BOS = 2149
const CODEC_THINK = 2154
const CODEC_THINK_BOS = 2156
const CODEC_THINK_EOS = 2157
const TTS_PAD = 151671
const TTS_BOS = 151672
const TTS_EOS = 151673

export interface PromptResult {
  prefill: Float32Array
  trailing: Float32Array[]
  ttsPad: Float32Array
}

export function buildPrompt(
  text: string,
  speakerEmb: Float32Array,
  langId: number,
  tokenizer: { encode(text: string): number[] },
  codecEmb: Float32Array,
  textEmbData: Float32Array,
  project: (row: Float32Array) => Float32Array,
): PromptResult {
  const ids = tokenizer.encode(`<|im_start|>assistant\n${text}<|im_end|>\n<|im_start|>assistant\n`)

  const embed = (id: number) => {
    const base = id * HIDDEN
    if (base + HIDDEN > textEmbData.length) return new Float32Array(HIDDEN)
    return project(textEmbData.slice(base, base + HIDDEN))
  }

  const ttsBos = embed(TTS_BOS)
  const ttsEos = embed(TTS_EOS)
  const ttsPad = embed(TTS_PAD)

  const control: number[] = [CODEC_THINK, CODEC_THINK_BOS, langId, CODEC_THINK_EOS]
  const codecPreEmb = control.map(c => codecEmb.slice(c * HIDDEN, (c + 1) * HIDDEN))

  const roleIds = ids.slice(0, 3)
  const roleEmb = roleIds.map(embed)

  const bodyEmb: Float32Array[] = []
  for (const ce of codecPreEmb) {
    const p = new Float32Array(HIDDEN)
    for (let i = 0; i < HIDDEN; i++) p[i] = ttsPad[i] + ce[i]
    bodyEmb.push(p)
  }
  const speakerRow = new Float32Array(HIDDEN)
  for (let i = 0; i < HIDDEN; i++) speakerRow[i] = ttsPad[i] + speakerEmb[i]
  bodyEmb.push(speakerRow)
  const bosRow = new Float32Array(HIDDEN)
  for (let i = 0; i < HIDDEN; i++) bosRow[i] = ttsPad[i] + codecEmb[CODEC_PAD * HIDDEN + i]
  bodyEmb.push(bosRow)

  const firstTextId = ids[3]
  const firstTextEmb = embed(firstTextId)
  const codecEosEmb = codecEmb.slice(CODEC_BOS * HIDDEN, (CODEC_BOS + 1) * HIDDEN)
  const firstTextRow = new Float32Array(HIDDEN)
  for (let i = 0; i < HIDDEN; i++) firstTextRow[i] = firstTextEmb[i] + codecEosEmb[i]

  const prefill = new Float32Array((roleEmb.length + bodyEmb.length + 1) * HIDDEN)
  let off = 0
  for (const e of roleEmb) { prefill.set(e, off); off += HIDDEN }
  for (const e of bodyEmb) { prefill.set(e, off); off += HIDDEN }
  prefill.set(firstTextRow, off)

  const trailingIds = ids.slice(4)
  const trailingEmb = trailingIds.map(id => {
    const e = embed(id)
    const r = new Float32Array(HIDDEN)
    for (let i = 0; i < HIDDEN; i++) r[i] = e[i] + ttsPad[i]
    return r
  })

  return { prefill, trailing: [...trailingEmb, ttsEos], ttsPad }
}