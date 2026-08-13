import type {
  QualificationBackend,
  QualificationEnvironment,
} from '../schema/types'

const runtimePackage = '@litertjs/core'
const runtimeVersion = '2.5.3'

export function captureNodeEnvironment(
  requestedBackend: QualificationBackend,
): QualificationEnvironment {
  return {
    runtimePackage,
    runtimeVersion,
    requestedBackend,
  }
}

export function captureBrowserEnvironment(
  requestedBackend: QualificationBackend,
  browser: {
    browser?: string
    browserVersion?: string
    name?: string
    version?: string
    operatingSystem?: string
    device?: string
    gpu?: string
    webgpuAvailable: boolean
  },
): QualificationEnvironment {
  return {
    browser: browser.browser ?? browser.name,
    browserVersion: browser.browserVersion ?? browser.version,
    operatingSystem: browser.operatingSystem,
    device: browser.device,
    gpu: browser.gpu,
    runtimePackage,
    runtimeVersion,
    requestedBackend,
    webgpuAvailable: browser.webgpuAvailable,
  }
}
