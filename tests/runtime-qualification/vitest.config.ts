import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    include: ['tests/runtime-qualification/**/*.test.ts'],
    environment: 'node',
  },
})
