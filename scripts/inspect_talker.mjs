import { loadLiteRt } from '@litertjs/core'

async function main() {
  const model = await loadLiteRt('public/models/qwen3-tts/talker_fp32.tflite')
  
  console.log('Signatures:')
  for (const [name, sig] of Object.entries(model.signatures)) {
    console.log(`\n  ${name}:`)
    console.log('    Inputs:')
    for (const [key, info] of Object.entries(sig.inputs)) {
      console.log(`      ${key}: shape=${JSON.stringify(info.shape)} dtype=${info.dtype}`)
    }
    console.log('    Outputs:')
    for (const [key, info] of Object.entries(sig.outputs)) {
      console.log(`      ${key}: shape=${JSON.stringify(info.shape)} dtype=${info.dtype}`)
    }
  }
}

main().catch(console.error)
