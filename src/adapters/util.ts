export function flatten(input: number | number[] | number[][] | number[]): Float32Array {
  if (input instanceof Float32Array) return input
  if (typeof input === 'number') return new Float32Array([input])
  if (Array.isArray(input[0])) {
    const flat: number[] = []
    for (const sub of input as number[][]) for (const v of sub) flat.push(v)
    return new Float32Array(flat)
  }
  return new Float32Array(input as number[])
}

export function toNestedArray(flat: Float32Array, dims: number[]): any {
  if (dims.length === 0) return flat[0]
  const size = dims[0]
  const rest = dims.slice(1)
  const result: any[] = []
  let offset = 0
  for (let i = 0; i < size; i++) {
    const subLen = rest.reduce((a, b) => a * b, 1)
    if (rest.length === 1) {
      result.push(Array.from(flat.slice(offset, offset + subLen)))
    } else {
      result.push(toNestedArray(flat.slice(offset, offset + subLen), rest))
    }
    offset += subLen
  }
  return result
}
