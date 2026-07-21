import { describe, expect, it, vi } from 'vitest'
import type { TensorDescriptor } from './types'
import { validateOmnixTensorMap } from './schema'
import { samplePointIdentities } from './sampling'

function descriptor(shape: number[], key: string): TensorDescriptor {
  const stride = new Array<number>(shape.length)
  let elements = 1
  for (let index = shape.length - 1; index >= 0; index -= 1) {
    stride[index] = elements
    elements *= shape[index]
  }
  return {
    storage: {
      key,
      dtype: 'float32',
      location: 'cpu',
      elementCount: elements,
      byteLength: elements * 4,
    },
    storageOffset: 0,
    shape,
    stride,
    requiresGrad: false,
  }
}

function schema(width = 8) {
  return validateOmnixTensorMap(new Map([
    ['trajectory', descriptor([1, 2, 1, width, 3], '0')],
    ['camera_pose', descriptor([1, 3, 4], '1')],
    ['intrinsics', descriptor([1, 3, 3], '2')],
    ['pts3d_dynamic_score', descriptor([1, 1, width], '3')],
  ]))
}

function lineFrame(width: number): Float32Array {
  const result = new Float32Array(width * 3)
  for (let point = 0; point < width; point += 1) result[point * 3] = point
  return result
}

describe('samplePointIdentities', () => {
  it('reserves the highest-scoring dynamic identity and voxel-samples the remainder', async () => {
    const result = await samplePointIdentities(
      schema(),
      new Float32Array([0.1, 0.9, 0.8, 0.7, 0.6, 0.2, 0.3, 0.4]),
      4,
      async () => lineFrame(8),
    )

    expect(Array.from(result.identities)).toEqual([1, 2, 4, 6])
    expect(result).toMatchObject({
      selectedPointCount: 4,
      dynamicSelectedPointCount: 1,
      validCandidateCount: 8,
    })
  })

  it('uses stable lower-identity tie breaking for dynamic scores', async () => {
    const result = await samplePointIdentities(
      schema(),
      new Float32Array([0.1, 0.9, 0.9, 0.7, 0.6, 0.2, 0.3, 0.4]),
      4,
      async () => lineFrame(8),
    )

    expect(result.identities).toContain(1)
    expect(result.dynamicSelectedPointCount).toBe(1)
  })

  it('falls back to deterministic midpoint sampling for degenerate geometry', async () => {
    const result = await samplePointIdentities(
      schema(),
      new Float32Array(8),
      3,
      async () => new Float32Array(8 * 3),
    )

    expect(Array.from(result.identities)).toEqual([1, 4, 6])
  })

  it('fast-paths full resolution without reading frame zero', async () => {
    const reader = vi.fn(async () => lineFrame(8))
    const result = await samplePointIdentities(
      schema(),
      new Float32Array([0, 1, 0, 1, 0, 1, 0, 1]),
      8,
      reader,
    )

    expect(Array.from(result.identities)).toEqual([0, 1, 2, 3, 4, 5, 6, 7])
    expect(result.dynamicSelectedPointCount).toBe(2)
    expect(reader).not.toHaveBeenCalled()
  })
})
