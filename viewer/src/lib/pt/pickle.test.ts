import { describe, expect, it } from 'vitest'
import { parseTorchPickle } from './pickle'
import { PtParseError, type PtParseErrorCode } from './types'

const DEER_PICKLE_BASE64 =
  'gAJ9cQAoWAoAAAB0cmFqZWN0b3J5cQFjdG9yY2guX3V0aWxzCl9yZWJ1aWxkX3RlbnNvcl92MgpxAigoWAcAAABzdG9yYWdlcQNjdG9yY2gKRmxvYXRTdG9yYWdlCnEEWAEAAAAwcQVYAwAAAGNwdXEGSgDAdQZ0cQdRSwAoSxBLEE0YAU34AUsDdHEIKEoAXGcASsB1BgBN6AVLA0sBdHEJiWNjb2xsZWN0aW9ucwpPcmRlcmVkRGljdApxCilScQt0cQxScQ1YCwAAAGNhbWVyYV9wb3NlcQ5oAigoaANoBFgBAAAAMXEPaAZLwHRxEFFLAEsQSwNLBIdxEUsMSwRLAYdxEoloCilScRN0cRRScRVYCgAAAGludHJpbnNpY3NxFmgCKChoA2gEWAEAAAAycRdoBkuQdHEYUUsASxBLA0sDh3EZSwlLA0sBh3EaiWgKKVJxG3RxHFJxHVgTAAAAcHRzM2RfZHluYW1pY19zY29yZXEeaAIoKGgDaARYAQAAADNxH2gGSgB0IgB0cSBRSwBLEE0YAU34AYdxIUpAJwIATfgBSwGHcSKJaAopUnEjdHEkUnEldS4='

function goldenPickle(): Uint8Array {
  return Uint8Array.from(atob(DEER_PICKLE_BASE64), (character) => character.charCodeAt(0))
}

function errorCode(bytes: Uint8Array): PtParseErrorCode {
  try {
    parseTorchPickle(bytes)
  } catch (error) {
    expect(error).toBeInstanceOf(PtParseError)
    return (error as PtParseError).code
  }
  throw new Error('Expected parseTorchPickle to reject the fixture.')
}

function findBytes(haystack: Uint8Array, text: string): number {
  const needle = new TextEncoder().encode(text)
  outer: for (let start = 0; start <= haystack.byteLength - needle.byteLength; start += 1) {
    for (let index = 0; index < needle.byteLength; index += 1) {
      if (haystack[start + index] !== needle[index]) {
        continue outer
      }
    }
    return start
  }
  throw new Error('Expected test bytes were not found.')
}

describe('parseTorchPickle', () => {
  it('parses the real inference pickle into inert, contiguous tensor descriptors', () => {
    const tensors = parseTorchPickle(goldenPickle())

    expect([...tensors.keys()]).toEqual([
      'trajectory',
      'camera_pose',
      'intrinsics',
      'pts3d_dynamic_score',
    ])
    expect(tensors.get('trajectory')).toEqual({
      storage: {
        key: '0',
        dtype: 'float32',
        location: 'cpu',
        elementCount: 108_380_160,
        byteLength: 433_520_640,
      },
      storageOffset: 0,
      shape: [16, 16, 280, 504, 3],
      stride: [6_773_760, 423_360, 1_512, 3, 1],
      requiresGrad: false,
    })
    expect(tensors.get('camera_pose')?.shape).toEqual([16, 3, 4])
    expect(tensors.get('camera_pose')?.storage.byteLength).toBe(768)
    expect(tensors.get('intrinsics')?.shape).toEqual([16, 3, 3])
    expect(tensors.get('pts3d_dynamic_score')).toMatchObject({
      storage: { key: '3', elementCount: 2_257_920, byteLength: 9_031_680 },
      shape: [16, 280, 504],
      stride: [141_120, 504, 1],
    })
  })

  it('rejects every global outside the three-name allowlist without echoing it', () => {
    const bytes = goldenPickle()
    const globalOffset = findBytes(bytes, 'collections\n')
    bytes[globalOffset] = 'x'.charCodeAt(0)

    try {
      parseTorchPickle(bytes)
      throw new Error('Expected parser rejection.')
    } catch (error) {
      expect(error).toBeInstanceOf(PtParseError)
      expect((error as PtParseError).code).toBe('UNSAFE_PICKLE')
      expect((error as Error).message).not.toContain('xollections')
    }
  })

  it('rejects unsupported protocols, opcodes, trailing bytes, and truncation', () => {
    const protocol = goldenPickle()
    protocol[1] = 4
    expect(errorCode(protocol)).toBe('UNSUPPORTED_PICKLE')

    const opcode = goldenPickle()
    opcode[2] = 0x4e // NONE is intentionally outside the allowlist.
    expect(errorCode(opcode)).toBe('UNSUPPORTED_PICKLE')

    const golden = goldenPickle()
    const trailing = new Uint8Array(golden.byteLength + 1)
    trailing.set(golden)
    expect(errorCode(trailing)).toBe('INVALID_PICKLE')
    expect(errorCode(golden.subarray(0, golden.byteLength - 1))).toBe('INVALID_PICKLE')
  })

  it('rejects non-contiguous descriptors and altered schema keys', () => {
    const nonContiguous = goldenPickle()
    // The BININT1 argument at 158 is the trajectory's final stride.
    expect(nonContiguous[157]).toBe(0x4b)
    nonContiguous[158] = 2
    expect(errorCode(nonContiguous)).toBe('INVALID_TENSOR_SCHEMA')

    const wrongKey = goldenPickle()
    wrongKey[findBytes(wrongKey, 'trajectory')] = 'x'.charCodeAt(0)
    expect(errorCode(wrongKey)).toBe('INVALID_TENSOR_SCHEMA')
  })
})
