import { describe, expect, it, vi } from 'vitest'
import { InferenceCoordinator } from './coordinator'

describe('InferenceCoordinator', () => {
  it('serializes inference tasks and reports queue state', async () => {
    const coordinator = new InferenceCoordinator()
    const order: string[] = []
    let releaseFirst!: () => void
    const firstGate = new Promise<void>((resolve) => { releaseFirst = resolve })

    const first = coordinator.run(async () => {
      order.push('first:start')
      await firstGate
      order.push('first:end')
      return 1
    }, undefined, 'first')

    const second = coordinator.run(async () => {
      order.push('second:start')
      return 2
    }, undefined, 'second')

    await vi.waitFor(() => expect(coordinator.snapshot().activeLabel).toBe('first'))
    expect(coordinator.snapshot().queued).toBe(1)
    expect(order).toEqual(['first:start'])

    releaseFirst()
    await expect(Promise.all([first, second])).resolves.toEqual([1, 2])
    expect(order).toEqual(['first:start', 'first:end', 'second:start'])
    expect(coordinator.snapshot()).toEqual({ active: false, activeLabel: undefined, queued: 0 })
  })

  it('rejects an aborted queued task without running it', async () => {
    const coordinator = new InferenceCoordinator()
    let releaseFirst!: () => void
    const firstGate = new Promise<void>((resolve) => { releaseFirst = resolve })
    const first = coordinator.run(() => firstGate, undefined, 'first')

    const controller = new AbortController()
    const task = vi.fn().mockResolvedValue('should-not-run')
    const second = coordinator.run(task, controller.signal, 'second')
    controller.abort()
    releaseFirst()

    await first
    await expect(second).rejects.toMatchObject({ name: 'AbortError' })
    expect(task).not.toHaveBeenCalled()
    expect(coordinator.snapshot().queued).toBe(0)
  })
})
