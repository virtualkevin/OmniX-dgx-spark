import { describe, expect, it } from 'vitest'
import type { TensorDescriptor } from './types'
import {
  OmnixSchemaError,
  validateAndConvertCameras,
  validateDynamicScores,
  validateOmnixTensorMap,
} from './schema'

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

function validMap(): Map<string, TensorDescriptor> {
  return new Map([
    ['trajectory', descriptor([2, 3, 2, 4, 3], '0')],
    ['camera_pose', descriptor([2, 3, 4], '1')],
    ['intrinsics', descriptor([2, 3, 3], '2')],
    ['pts3d_dynamic_score', descriptor([2, 2, 4], '3')],
  ])
}

describe('validateOmnixTensorMap', () => {
  it('accepts only the exact contiguous OmniX tensor contract', () => {
    const schema = validateOmnixTensorMap(validMap())

    expect(schema).toMatchObject({
      sourceViewCount: 2,
      frameCount: 3,
      height: 2,
      width: 4,
      identitiesPerView: 8,
      identityCount: 16,
    })
  })

  it('accepts the repository full-resolution 32-view, 32-frame tensor sizes', () => {
    const tensors = new Map([
      ['trajectory', descriptor([32, 32, 280, 504, 3], '0')],
      ['camera_pose', descriptor([32, 3, 4], '1')],
      ['intrinsics', descriptor([32, 3, 3], '2')],
      ['pts3d_dynamic_score', descriptor([32, 280, 504], '3')],
    ])

    const schema = validateOmnixTensorMap(tensors)

    expect(schema).toMatchObject({
      sourceViewCount: 32,
      frameCount: 32,
      height: 280,
      width: 504,
      identityCount: 4_515_840,
      totalTensorBytes: 1_752_148_608,
    })
  })

  it('rejects extra keys, shared storage, and non-contiguous strides', () => {
    const extra = validMap()
    extra.set('extra', descriptor([1], '4'))
    expect(() => validateOmnixTensorMap(extra)).toThrowError(OmnixSchemaError)

    const shared = validMap()
    shared.set('intrinsics', descriptor([2, 3, 3], '1'))
    expect(() => validateOmnixTensorMap(shared)).toThrow(/independent storages/)

    const strided = validMap()
    const trajectory = strided.get('trajectory')!
    strided.set('trajectory', { ...trajectory, stride: [1, 1, 1, 1, 1] })
    expect(() => validateOmnixTensorMap(strided)).toThrow(/contiguous strides/)
  })
})

describe('tensor value validation', () => {
  it('validates calibration and converts OpenCV coordinates into the Three.js basis', () => {
    const pose = new Float32Array([
      1, 0, 0, 1,
      0, 1, 0, 2,
      0, 0, 1, 3,
    ])
    const intrinsic = new Float32Array([
      500, 0, 252,
      0, 500, 140,
      0, 0, 1,
    ])

    const converted = validateAndConvertCameras(pose, intrinsic, 1)
    expect(Array.from(converted.cameraPose)).toEqual([
      1, -0, -0, 1,
      -0, 1, 0, -2,
      -0, 0, 1, -3,
      0, -0, -0, 1,
    ])
    expect(converted.intrinsics).toEqual(intrinsic)
  })

  it('rejects non-finite and out-of-range dynamic scores', () => {
    expect(() => validateDynamicScores(new Float32Array([0, 0.5, 1]))).not.toThrow()
    expect(() => validateDynamicScores(new Float32Array([0, Number.NaN]))).toThrow(/NaN/)
    expect(() => validateDynamicScores(new Float32Array([1.1]))).toThrow(/\[0, 1\]/)
  })
})
