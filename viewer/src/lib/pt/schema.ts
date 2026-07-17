import type { TensorDescriptor } from './types'

const FLOAT32_BYTES = Float32Array.BYTES_PER_ELEMENT

export const OMX_MAX_SOURCE_VIEWS = 64
export const OMX_MAX_FRAMES = 600
export const OMX_MAX_SOURCE_PIXELS = 16_000_000
export const OMX_MAX_TENSOR_BYTES = 1024 * 1024 * 1024

const REQUIRED_KEYS = [
  'trajectory',
  'camera_pose',
  'intrinsics',
  'pts3d_dynamic_score',
] as const

type RequiredKey = (typeof REQUIRED_KEYS)[number]

export type OmnixSchemaErrorCode =
  | 'INVALID_SCHEMA_KEYS'
  | 'INVALID_TENSOR_DESCRIPTOR'
  | 'UNSUPPORTED_TENSOR'
  | 'INVALID_TENSOR_SHAPE'
  | 'RESOURCE_LIMIT_EXCEEDED'
  | 'NON_FINITE_TENSOR'
  | 'INVALID_DYNAMIC_SCORE'
  | 'INVALID_INTRINSICS'
  | 'INVALID_CAMERA_POSE'

export class OmnixSchemaError extends Error {
  readonly code: OmnixSchemaErrorCode

  constructor(code: OmnixSchemaErrorCode, message: string) {
    super(message)
    this.name = 'OmnixSchemaError'
    this.code = code
  }
}

export interface OmnixTensorSchema {
  trajectory: TensorDescriptor
  cameraPose: TensorDescriptor
  intrinsics: TensorDescriptor
  dynamicScore: TensorDescriptor
  sourceViewCount: number
  frameCount: number
  height: number
  width: number
  identitiesPerView: number
  identityCount: number
  totalTensorBytes: number
}

function fail(code: OmnixSchemaErrorCode, message: string): never {
  throw new OmnixSchemaError(code, message)
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

function checkedProduct(values: readonly number[], label: string): number {
  let result = 1
  for (const value of values) {
    if (!Number.isSafeInteger(value) || value <= 0) {
      fail('INVALID_TENSOR_SHAPE', `${label} must have positive, safe-integer dimensions.`)
    }
    if (result > Number.MAX_SAFE_INTEGER / value) {
      fail('RESOURCE_LIMIT_EXCEEDED', `${label} is too large for this browser.`)
    }
    result *= value
  }
  return result
}

function contiguousStride(shape: readonly number[]): number[] {
  const result = new Array<number>(shape.length)
  let stride = 1
  for (let index = shape.length - 1; index >= 0; index -= 1) {
    result[index] = stride
    stride *= shape[index]
  }
  return result
}

function assertTensorDescriptor(value: unknown, field: RequiredKey): TensorDescriptor {
  if (!isRecord(value) || !isRecord(value.storage)) {
    fail('INVALID_TENSOR_DESCRIPTOR', `'${field}' is not a supported tensor descriptor.`)
  }

  const descriptor = value as unknown as TensorDescriptor
  const storage = descriptor.storage
  if (
    storage.dtype !== 'float32' ||
    storage.location !== 'cpu' ||
    typeof storage.key !== 'string' ||
    storage.key.length === 0 ||
    !Number.isSafeInteger(storage.elementCount) ||
    storage.elementCount <= 0 ||
    !Number.isSafeInteger(storage.byteLength) ||
    storage.byteLength <= 0
  ) {
    fail('UNSUPPORTED_TENSOR', `'${field}' must use a CPU float32 storage.`)
  }
  if (
    !Number.isSafeInteger(descriptor.storageOffset) ||
    descriptor.storageOffset !== 0 ||
    !Array.isArray(descriptor.shape) ||
    !Array.isArray(descriptor.stride) ||
    descriptor.requiresGrad !== false
  ) {
    fail(
      'UNSUPPORTED_TENSOR',
      `'${field}' must be a detached, zero-offset, contiguous tensor.`,
    )
  }

  const elementCount = checkedProduct(descriptor.shape, `'${field}'`)
  if (descriptor.stride.length !== descriptor.shape.length) {
    fail('UNSUPPORTED_TENSOR', `'${field}' has an incompatible stride rank.`)
  }
  const expectedStride = contiguousStride(descriptor.shape)
  if (
    descriptor.stride.some(
      (stride, index) => !Number.isSafeInteger(stride) || stride !== expectedStride[index],
    )
  ) {
    fail('UNSUPPORTED_TENSOR', `'${field}' must use canonical contiguous strides.`)
  }
  if (
    storage.elementCount !== elementCount ||
    storage.byteLength !== elementCount * FLOAT32_BYTES
  ) {
    fail('INVALID_TENSOR_DESCRIPTOR', `'${field}' storage size does not match its shape.`)
  }
  return descriptor
}

function sameShape(actual: readonly number[], expected: readonly number[]): boolean {
  return actual.length === expected.length && actual.every((value, index) => value === expected[index])
}

/** Validate the exact plain-tensor dictionary produced by OmniX inference. */
export function validateOmnixTensorMap(
  raw: ReadonlyMap<string, TensorDescriptor>,
): OmnixTensorSchema {
  const keys = [...raw.keys()].sort()
  const expectedKeys = [...REQUIRED_KEYS].sort()
  if (
    keys.length !== expectedKeys.length ||
    keys.some((key, index) => key !== expectedKeys[index])
  ) {
    fail(
      'INVALID_SCHEMA_KEYS',
      'The .pt root must contain exactly the four supported OmniX prediction tensors.',
    )
  }

  const trajectory = assertTensorDescriptor(raw.get('trajectory'), 'trajectory')
  const cameraPose = assertTensorDescriptor(raw.get('camera_pose'), 'camera_pose')
  const intrinsics = assertTensorDescriptor(raw.get('intrinsics'), 'intrinsics')
  const dynamicScore = assertTensorDescriptor(
    raw.get('pts3d_dynamic_score'),
    'pts3d_dynamic_score',
  )

  if (trajectory.shape.length !== 5 || trajectory.shape[4] !== 3) {
    fail('INVALID_TENSOR_SHAPE', "'trajectory' must have shape [S, T, H, W, 3].")
  }
  const [sourceViewCount, frameCount, height, width] = trajectory.shape
  if (!sameShape(cameraPose.shape, [sourceViewCount, 3, 4])) {
    fail('INVALID_TENSOR_SHAPE', "'camera_pose' must have shape [S, 3, 4].")
  }
  if (!sameShape(intrinsics.shape, [sourceViewCount, 3, 3])) {
    fail('INVALID_TENSOR_SHAPE', "'intrinsics' must have shape [S, 3, 3].")
  }
  if (!sameShape(dynamicScore.shape, [sourceViewCount, height, width])) {
    fail(
      'INVALID_TENSOR_SHAPE',
      "'pts3d_dynamic_score' must have shape [S, H, W].",
    )
  }

  if (sourceViewCount > OMX_MAX_SOURCE_VIEWS) {
    fail('RESOURCE_LIMIT_EXCEEDED', 'The source-view count exceeds the browser limit.')
  }
  if (frameCount > OMX_MAX_FRAMES) {
    fail('RESOURCE_LIMIT_EXCEEDED', 'The trajectory frame count exceeds the browser limit.')
  }
  const identitiesPerView = checkedProduct([height, width], 'Source image')
  const identityCount = checkedProduct([sourceViewCount, height, width], 'Source pixels')
  if (identityCount > OMX_MAX_SOURCE_PIXELS) {
    fail('RESOURCE_LIMIT_EXCEEDED', 'The source pixel count exceeds the browser limit.')
  }

  const storageKeys = new Set([
    trajectory.storage.key,
    cameraPose.storage.key,
    intrinsics.storage.key,
    dynamicScore.storage.key,
  ])
  if (storageKeys.size !== REQUIRED_KEYS.length) {
    fail('UNSUPPORTED_TENSOR', 'The four OmniX tensors must use independent storages.')
  }

  const totalTensorBytes =
    trajectory.storage.byteLength +
    cameraPose.storage.byteLength +
    intrinsics.storage.byteLength +
    dynamicScore.storage.byteLength
  if (!Number.isSafeInteger(totalTensorBytes) || totalTensorBytes > OMX_MAX_TENSOR_BYTES) {
    fail('RESOURCE_LIMIT_EXCEEDED', 'The decoded tensor data exceeds the browser limit.')
  }

  return {
    trajectory,
    cameraPose,
    intrinsics,
    dynamicScore,
    sourceViewCount,
    frameCount,
    height,
    width,
    identitiesPerView,
    identityCount,
    totalTensorBytes,
  }
}

export function assertFiniteFloat32(name: string, values: Float32Array): void {
  for (let index = 0; index < values.length; index += 1) {
    if (!Number.isFinite(values[index])) {
      fail('NON_FINITE_TENSOR', `'${name}' contains NaN or infinite values.`)
    }
  }
}

export function validateDynamicScores(values: Float32Array): void {
  assertFiniteFloat32('pts3d_dynamic_score', values)
  for (let index = 0; index < values.length; index += 1) {
    if (values[index] < -1e-6 || values[index] > 1 + 1e-6) {
      fail('INVALID_DYNAMIC_SCORE', "'pts3d_dynamic_score' must contain values in [0, 1].")
    }
  }
}

function close(actual: number, expected: number, absolute = 1e-4, relative = 1e-4): boolean {
  return Math.abs(actual - expected) <= absolute + relative * Math.abs(expected)
}

/** Validate calibration tensors and return Three.js-basis camera matrices. */
export function validateAndConvertCameras(
  rawCameraPose: Float32Array,
  rawIntrinsics: Float32Array,
  sourceViewCount: number,
): { cameraPose: Float32Array; intrinsics: Float32Array } {
  if (
    rawCameraPose.length !== sourceViewCount * 12 ||
    rawIntrinsics.length !== sourceViewCount * 9
  ) {
    fail('INVALID_TENSOR_SHAPE', 'Camera tensor storage lengths do not match the schema.')
  }
  assertFiniteFloat32('camera_pose', rawCameraPose)
  assertFiniteFloat32('intrinsics', rawIntrinsics)

  const cameraPose = new Float32Array(sourceViewCount * 16)
  const intrinsics = new Float32Array(rawIntrinsics)
  const basis = [1, -1, -1, 1] as const

  for (let view = 0; view < sourceViewCount; view += 1) {
    const poseOffset = view * 12
    const intrinsicOffset = view * 9
    const fx = rawIntrinsics[intrinsicOffset]
    const fy = rawIntrinsics[intrinsicOffset + 4]
    if (
      fx <= 0 ||
      fy <= 0 ||
      !close(rawIntrinsics[intrinsicOffset + 6], 0) ||
      !close(rawIntrinsics[intrinsicOffset + 7], 0) ||
      !close(rawIntrinsics[intrinsicOffset + 8], 1)
    ) {
      fail('INVALID_INTRINSICS', 'Camera intrinsics are not valid homogeneous matrices.')
    }

    const a = rawCameraPose[poseOffset]
    const b = rawCameraPose[poseOffset + 1]
    const c = rawCameraPose[poseOffset + 2]
    const d = rawCameraPose[poseOffset + 4]
    const e = rawCameraPose[poseOffset + 5]
    const f = rawCameraPose[poseOffset + 6]
    const g = rawCameraPose[poseOffset + 8]
    const h = rawCameraPose[poseOffset + 9]
    const i = rawCameraPose[poseOffset + 10]
    const determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if (determinant <= 0.5 || determinant >= 1.5) {
      fail('INVALID_CAMERA_POSE', 'Camera rotations must be right-handed and non-singular.')
    }
    const rotation = [a, b, c, d, e, f, g, h, i]
    for (let row = 0; row < 3; row += 1) {
      for (let column = 0; column < 3; column += 1) {
        let gram = 0
        for (let axis = 0; axis < 3; axis += 1) {
          gram += rotation[axis * 3 + row] * rotation[axis * 3 + column]
        }
        if (!close(gram, row === column ? 1 : 0, 5e-2, 5e-2)) {
          fail('INVALID_CAMERA_POSE', 'Camera rotations must be approximately orthonormal.')
        }
      }
    }

    const outputOffset = view * 16
    for (let row = 0; row < 4; row += 1) {
      for (let column = 0; column < 4; column += 1) {
        const raw = row < 3 ? (column < 4 ? rawCameraPose[poseOffset + row * 4 + column] : 0) : column === 3 ? 1 : 0
        cameraPose[outputOffset + row * 4 + column] = raw * basis[row] * basis[column]
      }
    }
  }

  return { cameraPose, intrinsics }
}
