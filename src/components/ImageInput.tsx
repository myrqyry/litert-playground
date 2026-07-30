import { useRef, useState, useCallback } from 'react'
import type { TensorSpec } from '../adapters/types'

function isNchw(spec: TensorSpec): boolean {
  return spec.shape.length === 4 && spec.shape[1] <= 4 && spec.shape[1] >= 1
}

function getImageDims(spec: TensorSpec): { h: number; w: number; c: number } {
  if (isNchw(spec)) return { h: spec.shape[2], w: spec.shape[3], c: spec.shape[1] }
  return { h: spec.shape[1], w: spec.shape[2], c: spec.shape[3] }
}

function isBgr(spec: TensorSpec): boolean {
  return spec.description.toLowerCase().includes('bgr')
}

function isZeroTo255(spec: TensorSpec): boolean {
  const d = spec.description.toLowerCase()
  return d.includes('0-255') || d.includes('0–255') || d.includes('no normalization')
}

interface ImageInputProps {
  specs: TensorSpec[]
  onChange: (values: Record<string, any>) => void
}

export default function ImageInput({ specs, onChange }: ImageInputProps) {
  const spec = specs.find(s => s.shape.length === 4)
  const [preview, setPreview] = useState<string | null>(null)
  const [imgDims, setImgDims] = useState<string>('')
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const processImage = useCallback((file: File) => {
    if (!spec) return
    const url = URL.createObjectURL(file)
    const img = new Image()
    img.onload = () => {
      processImageData(img, spec, setPreview, setImgDims, onChange, canvasRef)
      URL.revokeObjectURL(url)
    }
    img.src = url
  }, [spec, onChange])

  const handleFile = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]; if (f) processImage(f)
  }, [processImage])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    const f = e.dataTransfer.files[0]; if (f?.type.startsWith('image/')) processImage(f)
  }, [processImage])

  const handlePaste = useCallback((e: React.ClipboardEvent) => {
    const f = e.clipboardData.files[0]; if (f?.type.startsWith('image/')) processImage(f)
  }, [processImage])

  return (
    <div className="space-y-3">
      <h2 className="text-sm font-semibold text-on-surface">Image Input</h2>
      {spec && <p className="text-xs text-on-surface-variant">
        Expected: {spec.shape.join('×')} {spec.dtype} — {spec.description}
      </p>}

      <div
        onDrop={handleDrop}
        onPaste={handlePaste}
        onDragOver={e => e.preventDefault()}
        onClick={() => inputRef.current?.click()}
        className="border-2 border-dashed border-outline-variant rounded-xl p-8 text-center cursor-pointer hover:border-primary transition-colors"
      >
        {preview ? (
          <div className="space-y-2">
            <img src={preview} className="max-h-56 mx-auto rounded-lg shadow" alt="Preview" />
            <p className="text-xs text-m3-onSurfaceVariant">{imgDims}</p>
              <button
                onClick={e => { e.stopPropagation(); setPreview(null); onChange({}) }}
                className="text-xs text-error hover:underline"
            >
              Remove
            </button>
          </div>
        ) : (
          <div className="text-on-surface-variant space-y-1">
            <svg className="w-10 h-10 mx-auto opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
            <p className="text-sm font-medium">Click to upload / Drop image / Paste</p>
            <p className="text-xs">PNG, JPG, WebP</p>
          </div>
        )}
        <input ref={inputRef} type="file" accept="image/*" onChange={handleFile} className="hidden" />
      </div>
      <canvas ref={canvasRef} className="hidden" />
    </div>
  )
}

function processImageData(
  img: HTMLImageElement,
  spec: TensorSpec,
  setPreview: (url: string) => void,
  setImgDims: (d: string) => void,
  onChange: (v: Record<string, any>) => void,
  canvasRef: React.RefObject<HTMLCanvasElement | null>,
) {
  const { h: targetH, w: targetW, c: channels } = getImageDims(spec)
  const nchw = isNchw(spec)
  const bgr = isBgr(spec)
  const raw = isZeroTo255(spec)

  const r = Math.min(targetW / img.width, targetH / img.height)
  const cropW = Math.round(img.width * r)
  const cropH = Math.round(img.height * r)

  const canvas = canvasRef.current!
  canvas.width = targetW
  canvas.height = targetH
  const ctx = canvas.getContext('2d')!
  ctx.fillStyle = bgr ? '#727272' : '#000'
  ctx.fillRect(0, 0, targetW, targetH)
  const dx = (targetW - cropW) / 2
  const dy = (targetH - cropH) / 2
  ctx.drawImage(img, dx, dy, cropW, cropH)

  // Generate preview first
  setPreview(canvas.toDataURL('image/webp', 0.7))
  setImgDims(`${cropW}×${cropH} → ${targetW}×${targetH}`)

  const imageData = ctx.getImageData(0, 0, targetW, targetH)
  const pixels = imageData.data

  const tensorSize = targetH * targetW * channels
  const tensor = new Float32Array(tensorSize)

  for (let y = 0; y < targetH; y++) {
    for (let x = 0; x < targetW; x++) {
      const srcIdx = (y * targetW + x) * 4
      let r = pixels[srcIdx], g = pixels[srcIdx + 1], b = pixels[srcIdx + 2]

      if (bgr) [r, b] = [b, r]

      const vals = [r, g, b]

      if (nchw) {
        for (let c = 0; c < channels; c++) {
          const idx = c * targetH * targetW + y * targetW + x
          tensor[idx] = raw ? vals[c] : vals[c] / 255
        }
      } else {
        for (let c = 0; c < channels; c++) {
          tensor[y * targetW * channels + x * channels + c] = raw ? vals[c] : vals[c] / 255
        }
      }
    }
  }

  onChange({ [spec.name]: tensor })
}
