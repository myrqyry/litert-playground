import type {
  QualificationBackend,
  QualificationCase,
  QualificationContext,
  QualificationEnvironment,
  QualificationResult,
  QualificationRunOptions,
  QualificationSelection,
} from '../schema/types'
import { captureBrowserEnvironment } from './environment'
import { runQualificationMatrix } from './matrix'

export interface BrowserLaunchOptions {
  browserName: 'chromium' | 'firefox' | 'webkit'
  headless: boolean
}

export interface BrowserQualificationOptions extends QualificationRunOptions {
  launch: BrowserLaunchOptions
  selection: QualificationSelection
  launcher?: BrowserLauncher
}

export interface BrowserCapabilities {
  browser: string
  browserVersion: string
  operatingSystem?: string
  device?: string
  gpu?: string
  webgpuAvailable: boolean
}

export type BrowserLauncher = <T>(
  options: BrowserLaunchOptions,
  run: (capabilities: BrowserCapabilities) => Promise<T>,
) => Promise<T>

export async function runBrowserQualification(
  cases: QualificationCase[],
  options: BrowserQualificationOptions,
): Promise<QualificationResult[]> {
  const launcher = options.launcher ?? createPlaywrightLauncher()
  return launcher(options.launch, async (capabilities) => {
    const contexts: QualificationContext[] = cases
      .flatMap((qualificationCase) => qualificationCase.environments)
      .filter((environment, index, environments) =>
        environments.findIndex((candidate) =>
          candidate.requestedBackend === environment.requestedBackend,
        ) === index,
      )
      .map((environment) => createContext(environment.requestedBackend, capabilities))

    return runQualificationMatrix(cases, options.selection, contexts, options)
  })
}

function createContext(
  requestedBackend: QualificationBackend,
  capabilities: BrowserCapabilities,
): QualificationContext {
  const environment: QualificationEnvironment = captureBrowserEnvironment(
    requestedBackend,
    capabilities,
  )
  return {
    requestedBackend,
    environment,
    fetchAsset: async (asset) => {
      const response = await fetch(asset.url)
      if (!response.ok) throw new Error(`Asset request failed: ${response.status}`)
      const buffer = await response.arrayBuffer()
      if (buffer.byteLength !== asset.bytes) {
        throw new Error(`Asset ${asset.id} expected ${asset.bytes} bytes, got ${buffer.byteLength}`)
      }
      return buffer
    },
    runtime: { requestedBackend },
  }
}

function createPlaywrightLauncher(): BrowserLauncher {
  return async (options, run) => {
    const playwright = await import('playwright')
    const browserType = playwright[options.browserName]
    const browser = await browserType.launch({ headless: options.headless })
    const context = await browser.newContext()
    try {
      const capabilities: BrowserCapabilities = await context.newPage().then(async (page) => ({
        browser: options.browserName,
        browserVersion: browser.version(),
        webgpuAvailable: await page.evaluate(() => 'gpu' in navigator),
      }))
      return await run(capabilities)
    } finally {
      await context.close()
      await browser.close()
    }
  }
}
