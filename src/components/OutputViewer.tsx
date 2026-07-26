interface OutputViewerProps {
  outputs: Record<string, any> | null
}

export default function OutputViewer({ outputs }: OutputViewerProps) {
  if (!outputs) {
    return (
      <div>
        <h2 className="mb-2 text-sm font-semibold text-on-surface-variant uppercase tracking-wide">Outputs</h2>
        <p className="text-sm text-on-surface-variant">Run inference to see results</p>
      </div>
    )
  }
  return (
    <div>
      <h2 className="mb-3 text-sm font-semibold text-on-surface-variant uppercase tracking-wide">Outputs</h2>
      {Object.entries(outputs).map(([key, value]) => (
        <div key={key} className="mb-4">
          <div className="mb-1 text-sm font-semibold text-on-surface">{key}</div>
          <pre className="max-h-48 overflow-auto rounded-lg bg-surface-container px-4 py-3 font-mono text-xs text-on-surface [scrollbar-width:thin]">
            {JSON.stringify(value, null, 2)}
          </pre>
        </div>
      ))}
    </div>
  )
}
