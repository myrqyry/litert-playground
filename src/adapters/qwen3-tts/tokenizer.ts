export class BPETokenizer {
  private vocab: Map<string, number>
  private reverseVocab: Map<number, string>
  private addedTokens: Map<string, number>
  private pat: RegExp

  constructor(json: any) {
    this.vocab = new Map(Object.entries(json.model.vocab || {}))
    this.reverseVocab = new Map([...this.vocab].map(([k, v]) => [v, k]))
    this.addedTokens = new Map()
    for (const t of json.added_tokens || []) {
      this.addedTokens.set(t.content, t.id)
    }
    this.pat = /(?:[sdmt]|ll|ve|re)| ?\w+| ?[^\w\s]+|\s+(?!\S)|\s+/g
  }

  encode(text: string): number[] {
    const ids: number[] = []
    let remaining = text
    while (remaining.length > 0) {
      let matched = false
      for (const [token, id] of this.addedTokens) {
        if (remaining.startsWith(token)) {
          ids.push(id)
          remaining = remaining.slice(token.length)
          matched = true
          break
        }
      }
      if (matched) continue

      this.pat.lastIndex = 0
      const m = this.pat.exec(remaining)
      if (!m) break
      const word = m[0]
      const id = this.vocab.get(word)
      if (id !== undefined) {
        ids.push(id)
      } else {
        for (let i = 0; i < word.length; i++) {
          ids.push(word.charCodeAt(i) + 3)
        }
      }
    }
    return ids
  }

  decode(ids: number[]): string {
    const bytes: number[] = []
    for (const id of ids) {
      const s = this.reverseVocab.get(id)
      if (s) {
        for (let i = 0; i < s.length; i++) bytes.push(s.charCodeAt(i))
      } else if (id >= 3 && id < 259) {
        bytes.push(id - 3)
      }
    }
    const decoder = new TextDecoder('utf-8', { fatal: false })
    return decoder.decode(new Uint8Array(bytes))
  }
}
