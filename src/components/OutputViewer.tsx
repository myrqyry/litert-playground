interface OutputViewerProps {
  outputs: Record<string, any> | null
}

export default function OutputViewer({ outputs }: OutputViewerProps) {
  if (!outputs) {
    return <div><h2>Outputs</h2><p style={{ color: '#999' }}>Run inference to see results</p></div>
  }
  return (
    <div>
      <h2>Outputs</h2>
      {Object.entries(outputs).map(([key, value]) => (
        <div key={key} style={{ marginBottom: 12 }}>
          <div style={{ fontWeight: 600, marginBottom: 4 }}>{key}</div>
          <pre style={{
            background: '#f5f5f5', padding: 8, borderRadius: 4,
            overflowX: 'auto', fontSize: '0.85em', maxHeight: 200, overflowY: 'auto'
          }}>
            {JSON.stringify(value, null, 2)}
          </pre>
        </div>
      ))}
    </div>
  )
}
