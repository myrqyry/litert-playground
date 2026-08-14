import type {
  QualificationBackend,
  QualificationCase,
  QualificationContext,
  QualificationEnvironment,
  QualificationResult,
  QualificationRunOptions,
  QualificationSelection,
} from '../schema/types'
import type { ModelAssetDescriptor } from '../schema/types'
import { captureBrowserEnvironment } from './environment'
import { runQualificationMatrix } from './matrix'
import { createServer, type ViteDevServer } from 'vite'

interface BrowserRuntimeApi {
  initialize(path: string): Promise<void>
  loadAndCompile(model: number[], accelerator: QualificationBackend): Promise<{
    id: number
    inputs: Array<{ shape: number[]; dtype: string }>
    outputs: Array<{ shape: number[]; dtype: string }>
  }>
  run(id: number, values: Float32Array[], shapes: number[][]): Promise<number[][]>
  delete(id: number): void
}

declare global {
  interface Window {
    litertQualification: BrowserRuntimeApi
  }
}

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
  run: (
    capabilities: BrowserCapabilities,
    runtime: QualificationContext['runtime'],
    fetchAsset: QualificationContext['fetchAsset'],
  ) => Promise<T>,
) => Promise<T>

export async function runBrowserQualification(
  cases: QualificationCase[],
  options: BrowserQualificationOptions,
): Promise<QualificationResult[]> {
  const launcher = options.launcher ?? createPlaywrightLauncher()
  return launcher(options.launch, async (capabilities, runtime, fetchAsset) => {
    const contexts: QualificationContext[] = cases
      .flatMap((qualificationCase) => qualificationCase.environments)
      .filter((environment, index, environments) =>
        environments.findIndex((candidate) =>
          candidate.requestedBackend === environment.requestedBackend,
        ) === index,
      )
      .map((environment) => createContext(
        environment.requestedBackend,
        capabilities,
        runtime,
        fetchAsset,
      ))

    return runQualificationMatrix(cases, options.selection, contexts, options)
  })
}

function createContext(
  requestedBackend: QualificationBackend,
  capabilities: BrowserCapabilities,
  browserRuntime: QualificationContext['runtime'],
  browserFetch: QualificationContext['fetchAsset'],
): QualificationContext {
  const environment: QualificationEnvironment = captureBrowserEnvironment(
    requestedBackend,
    capabilities,
  )
  return {
    requestedBackend,
    environment,
    fetchAsset: browserFetch,
    runtime: browserRuntime,
  }
}

function createPlaywrightLauncher(): BrowserLauncher {
  return async (options, run) => {
    const playwright = await import('playwright')
    const server = await createQualificationServer()
    const browserType = playwright[options.browserName]
    const browser = await browserType.launch({ headless: options.headless })
    const context = await browser.newContext()
    try {
      const page = await context.newPage()
      const baseUrl = server.resolvedUrls?.local[0]
      if (!baseUrl) throw new Error('Qualification server did not expose a local URL')
      await page.goto(`${baseUrl}tests/runtime-qualification/browser-entry.html`)
      const capabilities: BrowserCapabilities = {
        browser: options.browserName,
        browserVersion: browser.version(),
        webgpuAvailable: await page.evaluate(() => 'gpu' in navigator),
      }
      const runtime = {
        initialize: () => page.evaluate((path) => window.litertQualification.initialize(path), '/node_modules/@litertjs/core/wasm'),
        loadAndCompile: async (model: Uint8Array, compileOptions: { accelerator: QualificationBackend }) => {
          const result = await page.evaluate(async ({ model, accelerator }) => window.litertQualification.loadAndCompile(model, accelerator), {
              model: Array.from(model),
            accelerator: compileOptions.accelerator,
          })
          return {
            getInputDetails: () => result.inputs,
            getOutputDetails: () => result.outputs,
            run: async (inputs: Array<{ data: unknown; shape?: readonly number[] }>) => {
              const values = inputs.map((input) => Array.from(input.data as Float32Array))
              const shapes = inputs.map((input) => [...(input.shape ?? [])])
              const outputs = await page.evaluate(({ id, values, shapes }) => window.litertQualification.run(id, values.map((value) => new Float32Array(value)), shapes), { id: result.id, values, shapes })
              return outputs.map((data: number[]) => ({ data: new Float32Array(data) }))
            },
            delete: () => page.evaluate((id) => window.litertQualification.delete(id), result.id),
          }
        },
      }
      const fetchAsset = async (asset: ModelAssetDescriptor): Promise<ArrayBuffer> => {
        const bytes = await page.evaluate(async (descriptor) => {
        const response = await fetch(descriptor.url)
        if (!response.ok) throw new Error(`Asset request failed: ${response.status}`)
        const buffer = await response.arrayBuffer()
        if (buffer.byteLength !== descriptor.bytes) throw new Error(`Asset ${descriptor.id} expected ${descriptor.bytes} bytes, got ${buffer.byteLength}`)
          return Array.from(new Uint8Array(buffer))
        }, asset)
        return new Uint8Array(bytes).buffer
      }
      return await run(capabilities, runtime, fetchAsset)
    } finally {
      await context.close()
      await browser.close()
      await server.close()
    }
  }
}

async function createQualificationServer(): Promise<ViteDevServer> {
  const server = await createServer({ root: process.cwd(), server: { port: 0 } })
  await server.listen()
  return server
}
