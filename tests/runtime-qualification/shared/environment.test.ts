import { describe, expect, it } from 'vitest'
import {
  captureBrowserEnvironment,
  captureNodeEnvironment,
} from './environment'

describe('qualification environments', () => {
  it('captures production node runtime defaults', () => {
    expect(captureNodeEnvironment('wasm')).toMatchObject({
      runtimePackage: '@litertjs/core',
      runtimeVersion: '2.5.3',
      requestedBackend: 'wasm',
    })
  })

  it('preserves browser capabilities and requested backend', () => {
    expect(captureBrowserEnvironment('webgpu', {
      name: 'Chromium',
      version: '140',
      operatingSystem: 'Linux',
      device: 'test-device',
      gpu: 'test-gpu',
      webgpuAvailable: true,
    })).toEqual({
      browser: 'Chromium',
      browserVersion: '140',
      operatingSystem: 'Linux',
      device: 'test-device',
      gpu: 'test-gpu',
      runtimePackage: '@litertjs/core',
      runtimeVersion: '2.5.3',
      requestedBackend: 'webgpu',
      webgpuAvailable: true,
    })
  })
})
