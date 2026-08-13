export type InferenceEvent = 'queued' | 'started' | 'finished'

export interface InferenceEventDetails {
  label: string
  queueDepth: number
  queuedMs?: number
  durationMs?: number
}

export interface InferenceCoordinatorSnapshot {
  active: boolean
  activeLabel?: string
  queued: number
}

type InferenceListener = (details: InferenceEventDetails) => void

function abortError(): Error {
  if (typeof DOMException !== 'undefined') return new DOMException('Inference was aborted', 'AbortError')
  const error = new Error('Inference was aborted')
  error.name = 'AbortError'
  return error
}

export class InferenceCoordinator {
  private activeLabel: string | undefined
  private queued = 0
  private tail: Promise<void> = Promise.resolve()
  private listeners = new Map<InferenceEvent, Set<InferenceListener>>()

  isBusy(): boolean {
    return this.activeLabel !== undefined
  }

  snapshot(): InferenceCoordinatorSnapshot {
    return { active: this.isBusy(), activeLabel: this.activeLabel, queued: this.queued }
  }

  on(event: InferenceEvent, listener: InferenceListener): () => void {
    const listeners = this.listeners.get(event) ?? new Set<InferenceListener>()
    listeners.add(listener)
    this.listeners.set(event, listeners)
    return () => listeners.delete(listener)
  }

  run<T>(task: () => Promise<T>, signal?: AbortSignal, label = 'inference'): Promise<T> {
    if (signal?.aborted) return Promise.reject(abortError())

    const queuedAt = performance.now()
    this.queued += 1
    this.emit('queued', { label, queueDepth: this.queued })

    const run = this.tail.then(async () => {
      this.queued -= 1
      if (signal?.aborted) throw abortError()

      this.activeLabel = label
      const startedAt = performance.now()
      this.emit('started', {
        label,
        queueDepth: this.queued,
        queuedMs: startedAt - queuedAt,
      })

      try {
        return await task()
      } finally {
        const durationMs = performance.now() - startedAt
        this.activeLabel = undefined
        this.emit('finished', { label, queueDepth: this.queued, durationMs })
      }
    })

    this.tail = run.then(() => undefined, () => undefined)
    return run
  }

  private emit(event: InferenceEvent, details: InferenceEventDetails): void {
    this.listeners.get(event)?.forEach((listener) => listener(details))
  }
}

export const inferenceCoordinator = new InferenceCoordinator()
