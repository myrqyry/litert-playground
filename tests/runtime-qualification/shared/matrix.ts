import type {
  QualificationCase,
  QualificationContext,
  QualificationResult,
  QualificationRunOptions,
  QualificationSelection,
} from '../schema/types'
import {
  createQualificationResult,
  normalizeQualificationError,
  writeQualificationResult,
} from './evidenceWriter'

export function selectQualificationCases(
  cases: QualificationCase[],
  selection: QualificationSelection,
): QualificationCase[] {
  const selected = selection.caseIds
    ? cases.filter((item) => selection.caseIds?.includes(item.id))
    : cases

  const missing = selection.caseIds?.find(
    (id) => !cases.some((item) => item.id === id),
  )
  if (missing) throw new Error(`Unknown qualification case: ${missing}`)
  return selected
}

export async function runQualificationCase(
  qualificationCase: QualificationCase,
  context: QualificationContext,
  options: QualificationRunOptions,
): Promise<QualificationResult> {
  let observed
  try {
    observed = await qualificationCase.run(context)
  } catch (error) {
    observed = {
      status: 'fail' as const,
      stage: 'run',
      error: normalizeQualificationError(error, 'run'),
    }
  }

  const result = createQualificationResult({
    caseId: qualificationCase.id,
    timestamp: (options.now ?? (() => new Date()))().toISOString(),
    playgroundRevision: options.playgroundRevision,
    runtimePackage: options.runtimePackage,
    runtimeVersion: options.runtimeVersion,
    environment: context.environment,
    model: qualificationCase.model,
    expected: qualificationCase.expected,
    observed,
  })

  if (options.resultsDirectory) {
    await writeQualificationResult(options.resultsDirectory, result)
  }
  return result
}

export async function runQualificationMatrix(
  cases: QualificationCase[],
  selection: QualificationSelection,
  contexts: QualificationContext[],
  options: QualificationRunOptions,
): Promise<QualificationResult[]> {
  const selected = selectQualificationCases(cases, selection)
  const selectedContexts = selection.backends
    ? contexts.filter((context) => selection.backends?.includes(context.requestedBackend))
    : contexts
  return Promise.all(selected.flatMap((item) => selectedContexts
    .filter((context) => item.environments.some(
      (environment) => environment.requestedBackend === context.requestedBackend,
    ))
    .map((context) => runQualificationCase(item, context, options))))
}
