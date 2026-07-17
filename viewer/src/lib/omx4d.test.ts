import { describe, expect, it } from 'vitest'
import {
  Omx4dParseError,
  getOmx4dFramePositions,
  getOmx4dTransferables,
  parseOmx4d,
  type Omx4dParseErrorCode,
} from './omx4d'

const MAGIC = new Uint8Array([0x4f, 0x4d, 0x58, 0x34, 0x44, 0x0d, 0x0a, 0x1a])

const OFFSETS = {
  positions: 2_048,
  colors: 2_096,
  dynamicScore: 2_104,
  sourceView: 2_112,
  cameraPose: 2_120,
  intrinsics: 2_248,
} as const

const PAYLOAD_BYTES = 2_320
const POSITIONS = [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15]

interface FixtureDescriptor {
  offset: number
  byteLength: number
  dtype: string
  shape: number[]
}

interface FixtureManifest {
  schemaVersion: number
  name: string
  fps: number
  frameCount: number
  durationSeconds: number
  sourceViewCount: number
  pointCount: number
  coordinateSystem: string
  units: string
  primitive: string
  bounds: { min: number[]; max: number[] }
  sampling: Record<string, unknown>
  warnings: string[]
  attributes: Record<string, FixtureDescriptor>
}

function fixtureManifest(): FixtureManifest {
  return {
    schemaVersion: 1,
    name: 'two-frame fixture',
    fps: 15,
    frameCount: 2,
    durationSeconds: 2 / 15,
    sourceViewCount: 2,
    pointCount: 2,
    coordinateSystem: 'right-handed-y-up',
    units: 'meters',
    primitive: 'points',
    bounds: { min: [0, 1, 2], max: [13, 14, 15] },
    sampling: {
      method: 'dynamic-reserved-voxel-v1',
      requestedPointCount: 2,
      selectedPointCount: 2,
      validCandidateCount: 2,
      dynamicThreshold: 0.5,
      dynamicReservedFraction: 0.25,
      dynamicSelectedPointCount: 1,
      identityHash: 'fixture',
    },
    warnings: ['fixture warning'],
    attributes: {
      positions: {
        offset: OFFSETS.positions,
        byteLength: 48,
        dtype: 'float32',
        shape: [2, 2, 3],
      },
      colors: {
        offset: OFFSETS.colors,
        byteLength: 6,
        dtype: 'uint8',
        shape: [2, 3],
      },
      dynamicScore: {
        offset: OFFSETS.dynamicScore,
        byteLength: 8,
        dtype: 'float32',
        shape: [2],
      },
      sourceView: {
        offset: OFFSETS.sourceView,
        byteLength: 4,
        dtype: 'uint16',
        shape: [2],
      },
      cameraPose: {
        offset: OFFSETS.cameraPose,
        byteLength: 128,
        dtype: 'float32',
        shape: [2, 4, 4],
      },
      intrinsics: {
        offset: OFFSETS.intrinsics,
        byteLength: 72,
        dtype: 'float32',
        shape: [2, 3, 3],
      },
    },
  }
}

function makeFixture(
  mutateManifest?: (manifest: FixtureManifest) => void,
  version = 1,
): ArrayBuffer {
  const manifest = fixtureManifest()
  mutateManifest?.(manifest)
  const header = new TextEncoder().encode(JSON.stringify(manifest))
  if (header.byteLength > OFFSETS.positions - 16) {
    throw new Error('Test manifest no longer fits before the first section.')
  }

  const buffer = new ArrayBuffer(PAYLOAD_BYTES)
  const bytes = new Uint8Array(buffer)
  const preamble = new DataView(buffer)
  bytes.set(MAGIC)
  preamble.setUint32(8, version, true)
  preamble.setUint32(12, header.byteLength, true)
  bytes.set(header, 16)

  new Float32Array(buffer, OFFSETS.positions, POSITIONS.length).set(POSITIONS)
  new Uint8Array(buffer, OFFSETS.colors, 6).set([255, 128, 0, 12, 34, 56])
  new Float32Array(buffer, OFFSETS.dynamicScore, 2).set([0.1, 0.9])
  new Uint16Array(buffer, OFFSETS.sourceView, 2).set([0, 1])

  const poses = new Float32Array(32)
  poses.set([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], 0)
  poses.set([1, 0, 0, 2, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], 16)
  new Float32Array(buffer, OFFSETS.cameraPose, poses.length).set(poses)

  const intrinsics = [100, 0, 50, 0, 100, 40, 0, 0, 1]
  new Float32Array(buffer, OFFSETS.intrinsics, 18).set([...intrinsics, ...intrinsics])
  return buffer
}

function parseErrorCode(buffer: ArrayBuffer): Omx4dParseErrorCode {
  try {
    parseOmx4d(buffer)
  } catch (error) {
    expect(error).toBeInstanceOf(Omx4dParseError)
    return (error as Omx4dParseError).code
  }
  throw new Error('Expected parseOmx4d to reject the fixture.')
}

describe('parseOmx4d', () => {
  it('parses a valid payload into zero-copy, transferable typed arrays and metadata', () => {
    const payload = makeFixture()
    const dataset = parseOmx4d(payload)

    expect(dataset.manifest.name).toBe('two-frame fixture')
    expect(dataset.manifest.bounds).toEqual({ min: [0, 1, 2], max: [13, 14, 15] })
    expect(dataset.metadata).toMatchObject({
      formatVersion: 1,
      frameCount: 2,
      pointCount: 2,
      sourceViewCount: 2,
      payloadByteLength: PAYLOAD_BYTES,
    })
    expect(dataset.positions).toBeInstanceOf(Float32Array)
    expect(dataset.colors).toBeInstanceOf(Uint8Array)
    expect(dataset.dynamicScore).toBeInstanceOf(Float32Array)
    expect(dataset.sourceView).toBeInstanceOf(Uint16Array)
    expect(dataset.cameraPose).toBeInstanceOf(Float32Array)
    expect(dataset.intrinsics).toBeInstanceOf(Float32Array)
    expect(Array.from(dataset.positions)).toEqual(POSITIONS)
    expect(Array.from(dataset.colors)).toEqual([255, 128, 0, 12, 34, 56])
    expect(dataset.dynamicScore[0]).toBeCloseTo(0.1)
    expect(dataset.dynamicScore[1]).toBeCloseTo(0.9)
    expect(Array.from(dataset.sourceView)).toEqual([0, 1])
    expect(Array.from(dataset.framePositions(1))).toEqual([10, 11, 12, 13, 14, 15])
    expect(Array.from(getOmx4dFramePositions(dataset, 0))).toEqual([0, 1, 2, 3, 4, 5])
    expect(dataset.positions.buffer).toBe(payload)
    expect(dataset.transferables).toEqual([payload])
    expect(getOmx4dTransferables(dataset)).toEqual([payload])
    expect(Object.keys(dataset)).not.toContain('framePositions')
    expect(() => dataset.framePositions(2)).toThrow(RangeError)

    const clone = structuredClone(dataset, { transfer: dataset.transferables })
    expect(Array.from(clone.positions.subarray(6))).toEqual([10, 11, 12, 13, 14, 15])
    expect('framePositions' in clone).toBe(false)
    expect(payload.byteLength).toBe(0)
  })

  it('rejects truncated, malformed, and invalid-length payload envelopes', () => {
    expect(parseErrorCode(new ArrayBuffer(15))).toBe('PAYLOAD_TOO_SMALL')

    const badMagic = makeFixture()
    new Uint8Array(badMagic)[0] = 0
    expect(parseErrorCode(badMagic)).toBe('INVALID_MAGIC')

    const badHeaderLength = makeFixture()
    new DataView(badHeaderLength).setUint32(12, badHeaderLength.byteLength, true)
    expect(parseErrorCode(badHeaderLength)).toBe('INVALID_HEADER_LENGTH')
  })

  it('rejects unsupported envelope versions', () => {
    expect(parseErrorCode(makeFixture(undefined, 2))).toBe('UNSUPPORTED_VERSION')
  })

  it('rejects misaligned and out-of-bounds absolute section ranges', () => {
    expect(
      parseErrorCode(
        makeFixture((manifest) => {
          manifest.attributes.positions.offset += 1
        }),
      ),
    ).toBe('INVALID_ATTRIBUTE')

    expect(
      parseErrorCode(
        makeFixture((manifest) => {
          manifest.attributes.intrinsics.offset = PAYLOAD_BYTES
        }),
      ),
    ).toBe('SECTION_OUT_OF_BOUNDS')
  })

  it('rejects unsupported dtypes and mismatched byte lengths', () => {
    expect(
      parseErrorCode(
        makeFixture((manifest) => {
          manifest.attributes.colors.dtype = 'float32'
        }),
      ),
    ).toBe('INVALID_ATTRIBUTE')

    expect(
      parseErrorCode(
        makeFixture((manifest) => {
          manifest.attributes.positions.byteLength -= 4
        }),
      ),
    ).toBe('INVALID_ATTRIBUTE')
  })

  it('rejects invalid manifest and attribute value ranges', () => {
    expect(
      parseErrorCode(
        makeFixture((manifest) => {
          manifest.bounds.min[0] = manifest.bounds.max[0] + 1
        }),
      ),
    ).toBe('INVALID_METADATA')

    const invalidScore = makeFixture()
    new Float32Array(invalidScore, OFFSETS.dynamicScore, 2)[1] = 1.1
    expect(parseErrorCode(invalidScore)).toBe('INVALID_ATTRIBUTE_DATA')

    const invalidSource = makeFixture()
    new Uint16Array(invalidSource, OFFSETS.sourceView, 2)[1] = 2
    expect(parseErrorCode(invalidSource)).toBe('INVALID_ATTRIBUTE_DATA')
  })
})
