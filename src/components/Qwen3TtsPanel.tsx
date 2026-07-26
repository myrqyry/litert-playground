import { useState, useEffect } from 'react'
import { Qwen3TtsPipeline, TTSProgress } from '../adapters/qwen3-tts/pipeline'

let pipeline: Qwen3TtsPipeline | null = null

export function Qwen3TtsPanel() {
  const [text, setText] = useState('Hello, welcome to my world.')
  const [generating, setGenerating] = useState(false)
  const [progress, setProgress] = useState('')
  const [audioUrl, setAudioUrl] = useState<string | null>(null)
  const [status, setStatus] = useState('Not loaded')

  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (pipeline?.ready) { setStatus('Ready'); return }
    if (pipeline && !pipeline.ready) { pipeline = null }
    setStatus('Loading...')
    setError(null)
    const p = new Qwen3TtsPipeline('/models/qwen3-tts')
    p.onProgress = (pr: TTSProgress) => {
      setProgress(`${pr.phase} ${pr.step}/${pr.total}`)
    }
    p.load()
      .then(() => { pipeline = p; setStatus('Ready'); setError(null) })
      .catch((e: unknown) => { setStatus('Load failed'); setError(String(e)) })
  }, [])

  const handleGenerate = async () => {
    if (!pipeline || generating) return
    setGenerating(true)
    setAudioUrl(null)
    try {
      setProgress('Generating...')
      const audio = await pipeline.synthesize(text)
      const wav = encodeWav(audio, 24000)
      const url = URL.createObjectURL(new Blob([wav], { type: 'audio/wav' }))
      setAudioUrl(url)
      setProgress('Done')
    } catch (e: unknown) {
      setProgress(`Error: ${e}`)
    }
    setGenerating(false)
  }

  const handleRetry = () => {
    pipeline = null
    setStatus('Not loaded')
    setError(null)
    setProgress('')
    window.location.reload()
  }

  return (
    <div className="space-y-4">
      <div className="text-sm text-on-surface-variant">Status: {status}</div>
      {error && (
        <div className="rounded-lg bg-error-container p-3 text-sm text-on-error-container">
          {error}
          <button onClick={handleRetry} className="ml-3 underline">Retry</button>
        </div>
      )}
      <textarea
        className="h-24 w-full rounded-lg border border-outline bg-surface-container p-3 font-mono text-sm text-on-surface transition-colors focus:border-primary focus:ring-2 focus:ring-primary/30 focus:outline-none"
        value={text}
        onChange={e => setText(e.target.value)}
        placeholder="Enter text to synthesize..."
      />
      <button
        className="inline-flex items-center justify-center rounded-full bg-primary px-8 py-3 text-sm font-medium text-on-primary shadow-md transition-all duration-300 hover:scale-[1.02] hover:shadow-lg active:scale-[0.97] disabled:opacity-50 disabled:shadow-none"
        style={{ transitionTimingFunction: 'var(--ease-spring)' }}
        onClick={handleGenerate}
        disabled={generating || status !== 'Ready'}
      >
        {generating ? 'Generating...' : 'Generate Speech'}
      </button>
      {progress && <div className="text-xs text-on-surface-variant">{progress}</div>}
      {audioUrl && (
        <audio controls src={audioUrl} className="w-full" />
      )}
    </div>
  )
}

function encodeWav(samples: Float32Array, sampleRate: number): ArrayBuffer {
  const numChannels = 1
  const bitsPerSample = 16
  const byteRate = sampleRate * numChannels * bitsPerSample / 8
  const blockAlign = numChannels * bitsPerSample / 8
  const dataSize = samples.length * blockAlign
  const buf = new ArrayBuffer(44 + dataSize)
  const v = new DataView(buf)

  const w = (off: number, str: string) => {
    for (let i = 0; i < str.length; i++) v.setUint8(off + i, str.charCodeAt(i))
  }
  w(0, 'RIFF'); v.setUint32(4, 36 + dataSize, true); w(8, 'WAVE')
  w(12, 'fmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true)
  v.setUint16(22, numChannels, true); v.setUint32(24, sampleRate, true)
  v.setUint32(28, byteRate, true); v.setUint16(32, blockAlign, true)
  v.setUint16(34, bitsPerSample, true)
  w(36, 'data'); v.setUint32(40, dataSize, true)

  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]))
    v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true)
  }
  return buf
}
