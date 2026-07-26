import { describe, it, expect } from 'vitest'
import { BPETokenizer } from './tokenizer'

const fixture = {
  version: '1.0',
  model: {
    type: 'BPE',
    vocab: {
      'Ġhello': 1, 'Ġworld': 2, 'Ġ': 3, '!': 4, 'hello': 5, 'world': 6,
      '.': 7, ',': 8, '?': 9, 'Ġhi': 10, 'Ġthere': 11,
    },
    merges: [],
  },
  added_tokens: [
    { id: 0, content: '<|endoftext|>', single_word: false, special: true },
    { id: 151935, content: '<|im_start|>', single_word: false, special: true },
    { id: 151936, content: '<|im_end|>', single_word: false, special: true },
  ],
}

describe('BPETokenizer', () => {
  it('splits text into BPE tokens and decodes back', () => {
    const tok = new BPETokenizer(fixture)
    const ids = tok.encode('hello world')
    expect(ids.length).toBeGreaterThan(0)
    expect(ids.every(id => typeof id === 'number')).toBe(true)
  })

  it('handles special tokens', () => {
    const tok = new BPETokenizer(fixture)
    const ids = tok.encode('<|im_start|>')
    expect(ids).toContain(151935)
  })

  it('decode round-trips', () => {
    const tok = new BPETokenizer(fixture)
    const ids = tok.encode('hello world')
    const decoded = tok.decode(ids)
    expect(decoded).toBe('hello world')
  })
})
