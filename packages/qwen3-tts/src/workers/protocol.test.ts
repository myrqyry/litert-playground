import { describe, expect, it } from 'vitest';
import { serializeError } from './protocol';

describe('serializeError', () => {
  it('serializes an Error', () => {
    const out = serializeError(new Error('boom'));
    expect(out.message).toBe('boom');
    expect(out.code).toBe('UNKNOWN');
  });

  it('serializes a string', () => {
    expect(serializeError('kaboom').message).toBe('kaboom');
  });

  it('serializes an InferenceError-like object with code and stage', () => {
    const out = serializeError({ code: 'CANCELLED', message: 'stopped', stage: 'decode' });
    expect(out).toEqual({ code: 'CANCELLED', message: 'stopped', stage: 'decode' });
  });
});
