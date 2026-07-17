const MAGIC = new Uint8Array([0x4f, 0x4d, 0x58, 0x34, 0x44, 0x0d, 0x0a, 0x1a])

export const OMX4D_FORMAT_VERSION = 1
export const OMX4D_PREAMBLE_BYTES = 16

const MAX_HEADER_BYTES = 4 * 1024 * 1024

export type Omx4dDtype = 'float32' | 'uint8' | 'uint16'
export type Omx4dAttributeName =
  | 'positions'
  | 'colors'
  | 'dynamicScore'
  | 'sourceView'
  | 'cameraPose'
  | 'intrinsics'

export interface Omx4dAttributeDescriptor {
  offset: number
  byteLength: number
  dtype: Omx4dDtype
  shape: number[]
}

export interface Omx4dBounds {
  min: [number, number, number]
  max: [number, number, number]
}

export interface Omx4dAttributeManifest {
  positions: Omx4dAttributeDescriptor
  colors: Omx4dAttributeDescriptor
  dynamicScore: Omx4dAttributeDescriptor
  sourceView: Omx4dAttributeDescriptor
  cameraPose: Omx4dAttributeDescriptor
  intrinsics: Omx4dAttributeDescriptor
}

export interface Omx4dManifest {
  schemaVersion: number
  name: string
  fps: number
  frameCount: number
  durationSeconds: number
  sourceViewCount: number
  pointCount: number
  coordinateSystem: string
  units: string
  primitive: 'points'
  bounds: Omx4dBounds
  sampling: Record<string, unknown>
  warnings?: string[]
  attributes: Omx4dAttributeManifest
  [key: string]: unknown
}

export interface Omx4dMetadata {
  formatVersion: typeof OMX4D_FORMAT_VERSION
  schemaVersion: number
  name: string
  fps: number
  frameCount: number
  durationSeconds: number
  sourceViewCount: number
  pointCount: number
  coordinateSystem: string
  units: string
  primitive: 'points'
  bounds: Omx4dBounds
  sampling: Record<string, unknown>
  warnings: string[]
  headerByteLength: number
  payloadByteLength: number
}

export interface Omx4dTypedAttributes {
  positions: Float32Array
  colors: Uint8Array
  dynamicScore: Float32Array
  sourceView: Uint16Array
  cameraPose: Float32Array
  intrinsics: Float32Array
}

/** Parsed data, including direct typed-array aliases used by the renderer. */
export interface Omx4dDataset {
  manifest: Omx4dManifest
  metadata: Omx4dMetadata
  attributes: Omx4dTypedAttributes
  positions: Float32Array
  colors: Uint8Array
  dynamicScore: Float32Array
  sourceView: Uint16Array
  cameraPose: Float32Array
  intrinsics: Float32Array
  transferables: ArrayBuffer[]
  framePositions(frameIndex: number): Float32Array
}

export type Omx4dParseErrorCode =
  | 'PAYLOAD_TOO_SMALL'
  | 'INVALID_MAGIC'
  | 'UNSUPPORTED_VERSION'
  | 'INVALID_HEADER_LENGTH'
  | 'INVALID_HEADER_ENCODING'
  | 'INVALID_MANIFEST'
  | 'INVALID_METADATA'
  | 'MISSING_ATTRIBUTE'
  | 'INVALID_ATTRIBUTE'
  | 'SECTION_OUT_OF_BOUNDS'
  | 'OVERLAPPING_SECTIONS'
  | 'INVALID_ATTRIBUTE_DATA'

export class Omx4dParseError extends Error {
  readonly code: Omx4dParseErrorCode

  constructor(code: Omx4dParseErrorCode, message: string, options?: ErrorOptions) {
    super(message, options)
    this.name = 'Omx4dParseError'
    this.code = code
  }
}

interface AttributeSpec {
  dtype: Omx4dDtype
  shape: (manifest: CoreManifest) => number[]
}

interface CoreManifest {
  schemaVersion: number
  name: string
  fps: number
  frameCount: number
  durationSeconds: number
  sourceViewCount: number
  pointCount: number
  coordinateSystem: string
  units: string
  primitive: 'points'
  bounds: Omx4dBounds
  sampling: Record<string, unknown>
  warnings: string[]
}

const ATTRIBUTE_NAMES: Omx4dAttributeName[] = [
  'positions',
  'colors',
  'dynamicScore',
  'sourceView',
  'cameraPose',
  'intrinsics',
]

const ATTRIBUTE_SPECS: Record<Omx4dAttributeName, AttributeSpec> = {
  positions: {
    dtype: 'float32',
    shape: ({ frameCount, pointCount }) => [frameCount, pointCount, 3],
  },
  colors: {
    dtype: 'uint8',
    shape: ({ pointCount }) => [pointCount, 3],
  },
  dynamicScore: {
    dtype: 'float32',
    shape: ({ pointCount }) => [pointCount],
  },
  sourceView: {
    dtype: 'uint16',
    shape: ({ pointCount }) => [pointCount],
  },
  cameraPose: {
    dtype: 'float32',
    shape: ({ sourceViewCount }) => [sourceViewCount, 4, 4],
  },
  intrinsics: {
    dtype: 'float32',
    shape: ({ sourceViewCount }) => [sourceViewCount, 3, 3],
  },
}

const BYTES_PER_ELEMENT: Record<Omx4dDtype, number> = {
  float32: Float32Array.BYTES_PER_ELEMENT,
  uint8: Uint8Array.BYTES_PER_ELEMENT,
  uint16: Uint16Array.BYTES_PER_ELEMENT,
}

function fail(code: Omx4dParseErrorCode, message: string, cause?: unknown): never {
  throw new Omx4dParseError(code, message, cause === undefined ? undefined : { cause })
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

function align8(value: number): number {
  return Math.ceil(value / 8) * 8
}

function requiredString(
  value: Record<string, unknown>,
  field: string,
): string {
  const result = value[field]
  if (typeof result !== 'string' || result.trim().length === 0) {
    fail('INVALID_METADATA', `Manifest field "${field}" must be a non-empty string.`)
  }
  return result
}

function requiredPositiveNumber(
  value: Record<string, unknown>,
  field: string,
): number {
  const result = value[field]
  if (typeof result !== 'number' || !Number.isFinite(result) || result <= 0) {
    fail('INVALID_METADATA', `Manifest field "${field}" must be a positive finite number.`)
  }
  return result
}

function requiredPositiveInteger(
  value: Record<string, unknown>,
  field: string,
): number {
  const result = value[field]
  if (!Number.isSafeInteger(result) || (result as number) <= 0) {
    fail('INVALID_METADATA', `Manifest field "${field}" must be a positive safe integer.`)
  }
  return result as number
}

function finiteVector(value: unknown, field: string): [number, number, number] {
  if (
    !Array.isArray(value) ||
    value.length !== 3 ||
    value.some((component) => typeof component !== 'number' || !Number.isFinite(component))
  ) {
    fail('INVALID_METADATA', `Manifest field "${field}" must contain three finite numbers.`)
  }
  return [value[0] as number, value[1] as number, value[2] as number]
}

function requiredBounds(value: unknown): Omx4dBounds {
  if (!isRecord(value)) {
    fail('INVALID_METADATA', 'Manifest field "bounds" must be an object.')
  }

  const min = finiteVector(value.min, 'bounds.min')
  const max = finiteVector(value.max, 'bounds.max')
  if (min.some((component, index) => component > max[index])) {
    fail('INVALID_METADATA', 'Manifest bounds must have min <= max on every axis.')
  }
  return { min, max }
}

function requiredSampling(value: unknown): Record<string, unknown> {
  if (!isRecord(value)) {
    fail('INVALID_METADATA', 'Manifest field "sampling" must be an object.')
  }
  return value
}

function optionalWarnings(value: unknown): string[] {
  if (value === undefined) return []
  if (!Array.isArray(value) || value.some((warning) => typeof warning !== 'string')) {
    fail('INVALID_METADATA', 'Manifest field "warnings" must be an array of strings.')
  }
  return value.slice() as string[]
}

function validateCoreManifest(value: Record<string, unknown>): CoreManifest {
  const schemaVersion = requiredPositiveInteger(value, 'schemaVersion')
  if (schemaVersion !== OMX4D_FORMAT_VERSION) {
    fail(
      'INVALID_METADATA',
      `Unsupported manifest schema version ${schemaVersion}; expected ${OMX4D_FORMAT_VERSION}.`,
    )
  }

  const frameCount = requiredPositiveInteger(value, 'frameCount')
  const fps = requiredPositiveNumber(value, 'fps')
  const durationSeconds = requiredPositiveNumber(value, 'durationSeconds')
  const expectedDuration = frameCount / fps
  const durationTolerance = Math.max(1e-6, expectedDuration * 1e-6)
  if (Math.abs(durationSeconds - expectedDuration) > durationTolerance) {
    fail(
      'INVALID_METADATA',
      `Manifest durationSeconds must equal frameCount / fps (${expectedDuration}).`,
    )
  }

  if (value.primitive !== 'points') {
    fail('INVALID_METADATA', 'Manifest field "primitive" must be "points".')
  }

  return {
    schemaVersion,
    name: requiredString(value, 'name'),
    fps,
    frameCount,
    durationSeconds,
    sourceViewCount: requiredPositiveInteger(value, 'sourceViewCount'),
    pointCount: requiredPositiveInteger(value, 'pointCount'),
    coordinateSystem: requiredString(value, 'coordinateSystem'),
    units: requiredString(value, 'units'),
    primitive: 'points',
    bounds: requiredBounds(value.bounds),
    sampling: requiredSampling(value.sampling),
    warnings: optionalWarnings(value.warnings),
  }
}

function checkedElementCount(shape: number[], attributeName: string): number {
  let count = 1
  for (const dimension of shape) {
    if (!Number.isSafeInteger(dimension) || dimension <= 0) {
      fail(
        'INVALID_ATTRIBUTE',
        `Attribute "${attributeName}" shape must contain only positive safe integers.`,
      )
    }
    if (count > Number.MAX_SAFE_INTEGER / dimension) {
      fail('INVALID_ATTRIBUTE', `Attribute "${attributeName}" shape is too large.`)
    }
    count *= dimension
  }
  return count
}

function sameShape(actual: number[], expected: number[]): boolean {
  return actual.length === expected.length && actual.every((value, index) => value === expected[index])
}

function validateDescriptor(
  attributeName: Omx4dAttributeName,
  value: unknown,
  core: CoreManifest,
  minimumSectionOffset: number,
  payloadByteLength: number,
): Omx4dAttributeDescriptor {
  if (!isRecord(value)) {
    fail('MISSING_ATTRIBUTE', `Manifest is missing attribute "${attributeName}".`)
  }

  const spec = ATTRIBUTE_SPECS[attributeName]
  if (value.dtype !== spec.dtype) {
    fail(
      'INVALID_ATTRIBUTE',
      `Attribute "${attributeName}" must use dtype "${spec.dtype}".`,
    )
  }
  if (!Array.isArray(value.shape)) {
    fail('INVALID_ATTRIBUTE', `Attribute "${attributeName}" must provide a shape array.`)
  }

  const shape = value.shape.slice()
  if (!sameShape(shape as number[], spec.shape(core))) {
    fail(
      'INVALID_ATTRIBUTE',
      `Attribute "${attributeName}" has shape ${JSON.stringify(shape)}; expected ${JSON.stringify(spec.shape(core))}.`,
    )
  }

  const offset = value.offset
  const byteLength = value.byteLength
  if (!Number.isSafeInteger(offset) || (offset as number) < minimumSectionOffset) {
    fail(
      'SECTION_OUT_OF_BOUNDS',
      `Attribute "${attributeName}" offset must be an absolute payload offset at or after ${minimumSectionOffset}.`,
    )
  }
  if ((offset as number) % 8 !== 0) {
    fail('INVALID_ATTRIBUTE', `Attribute "${attributeName}" offset must be 8-byte aligned.`)
  }
  if (!Number.isSafeInteger(byteLength) || (byteLength as number) <= 0) {
    fail('INVALID_ATTRIBUTE', `Attribute "${attributeName}" byteLength must be a positive safe integer.`)
  }

  const elementCount = checkedElementCount(shape as number[], attributeName)
  const bytesPerElement = BYTES_PER_ELEMENT[spec.dtype]
  if (elementCount > Number.MAX_SAFE_INTEGER / bytesPerElement) {
    fail('INVALID_ATTRIBUTE', `Attribute "${attributeName}" byte length is too large.`)
  }
  const expectedByteLength = elementCount * bytesPerElement
  if (byteLength !== expectedByteLength) {
    fail(
      'INVALID_ATTRIBUTE',
      `Attribute "${attributeName}" byteLength is ${String(byteLength)}; expected ${expectedByteLength}.`,
    )
  }

  const end = (offset as number) + expectedByteLength
  if (!Number.isSafeInteger(end) || end > payloadByteLength) {
    fail(
      'SECTION_OUT_OF_BOUNDS',
      `Attribute "${attributeName}" extends beyond the ${payloadByteLength}-byte payload.`,
    )
  }

  return {
    offset: offset as number,
    byteLength: expectedByteLength,
    dtype: spec.dtype,
    shape: shape as number[],
  }
}

function validateAttributes(
  value: unknown,
  core: CoreManifest,
  minimumSectionOffset: number,
  payloadByteLength: number,
): Omx4dAttributeManifest {
  if (!isRecord(value)) {
    fail('INVALID_MANIFEST', 'Manifest field "attributes" must be an object.')
  }

  const descriptors = Object.fromEntries(
    ATTRIBUTE_NAMES.map((name) => [
      name,
      validateDescriptor(name, value[name], core, minimumSectionOffset, payloadByteLength),
    ]),
  ) as unknown as Omx4dAttributeManifest

  const ranges = ATTRIBUTE_NAMES.map((name) => ({
    name,
    start: descriptors[name].offset,
    end: descriptors[name].offset + descriptors[name].byteLength,
  })).sort((left, right) => left.start - right.start)

  for (let index = 1; index < ranges.length; index += 1) {
    const previous = ranges[index - 1]
    const current = ranges[index]
    if (current.start < previous.end) {
      fail(
        'OVERLAPPING_SECTIONS',
        `Attributes "${previous.name}" and "${current.name}" overlap.`,
      )
    }
  }

  return descriptors
}

function float32View(buffer: ArrayBuffer, descriptor: Omx4dAttributeDescriptor): Float32Array {
  return new Float32Array(
    buffer,
    descriptor.offset,
    descriptor.byteLength / Float32Array.BYTES_PER_ELEMENT,
  )
}

function createTypedAttributes(
  buffer: ArrayBuffer,
  descriptors: Omx4dAttributeManifest,
): Omx4dTypedAttributes {
  return {
    positions: float32View(buffer, descriptors.positions),
    colors: new Uint8Array(buffer, descriptors.colors.offset, descriptors.colors.byteLength),
    dynamicScore: float32View(buffer, descriptors.dynamicScore),
    sourceView: new Uint16Array(
      buffer,
      descriptors.sourceView.offset,
      descriptors.sourceView.byteLength / Uint16Array.BYTES_PER_ELEMENT,
    ),
    cameraPose: float32View(buffer, descriptors.cameraPose),
    intrinsics: float32View(buffer, descriptors.intrinsics),
  }
}

function validateFinite(values: Float32Array, attributeName: Omx4dAttributeName): void {
  for (let index = 0; index < values.length; index += 1) {
    if (!Number.isFinite(values[index])) {
      fail(
        'INVALID_ATTRIBUTE_DATA',
        `Attribute "${attributeName}" contains a non-finite value at element ${index}.`,
      )
    }
  }
}

function validateAttributeData(attributes: Omx4dTypedAttributes, sourceViewCount: number): void {
  validateFinite(attributes.positions, 'positions')
  validateFinite(attributes.dynamicScore, 'dynamicScore')
  validateFinite(attributes.cameraPose, 'cameraPose')
  validateFinite(attributes.intrinsics, 'intrinsics')

  for (let index = 0; index < attributes.dynamicScore.length; index += 1) {
    const score = attributes.dynamicScore[index]
    if (score < 0 || score > 1) {
      fail(
        'INVALID_ATTRIBUTE_DATA',
        `Attribute "dynamicScore" must stay in [0, 1]; element ${index} is ${score}.`,
      )
    }
  }

  for (let index = 0; index < attributes.sourceView.length; index += 1) {
    const view = attributes.sourceView[index]
    if (view >= sourceViewCount) {
      fail(
        'INVALID_ATTRIBUTE_DATA',
        `Attribute "sourceView" element ${index} is ${view}, outside [0, ${sourceViewCount}).`,
      )
    }
  }
}

function readManifest(buffer: ArrayBuffer, headerByteLength: number): Record<string, unknown> {
  let text: string
  try {
    text = new TextDecoder('utf-8', { fatal: true }).decode(
      new Uint8Array(buffer, OMX4D_PREAMBLE_BYTES, headerByteLength),
    )
  } catch (error) {
    fail('INVALID_HEADER_ENCODING', 'OMX4D manifest is not valid UTF-8.', error)
  }

  let value: unknown
  try {
    value = JSON.parse(text)
  } catch (error) {
    fail('INVALID_MANIFEST', 'OMX4D manifest is not valid JSON.', error)
  }
  if (!isRecord(value)) {
    fail('INVALID_MANIFEST', 'OMX4D manifest must be a JSON object.')
  }
  return value
}

/**
 * Parse and validate a complete OMX4D v1 payload.
 *
 * Typed arrays are zero-copy views over `buffer`. Transfer the buffers returned by
 * `getOmx4dTransferables` when sending this value from a Web Worker.
 */
export function parseOmx4d(buffer: ArrayBuffer): Omx4dDataset {
  if (!(buffer instanceof ArrayBuffer)) {
    fail('PAYLOAD_TOO_SMALL', 'OMX4D input must be an ArrayBuffer.')
  }
  if (buffer.byteLength < OMX4D_PREAMBLE_BYTES) {
    fail(
      'PAYLOAD_TOO_SMALL',
      `OMX4D payload must contain at least ${OMX4D_PREAMBLE_BYTES} bytes.`,
    )
  }

  const bytes = new Uint8Array(buffer)
  if (MAGIC.some((expected, index) => bytes[index] !== expected)) {
    fail('INVALID_MAGIC', 'OMX4D payload has an invalid magic signature.')
  }

  const preamble = new DataView(buffer, 0, OMX4D_PREAMBLE_BYTES)
  const formatVersion = preamble.getUint32(8, true)
  if (formatVersion !== OMX4D_FORMAT_VERSION) {
    fail(
      'UNSUPPORTED_VERSION',
      `Unsupported OMX4D format version ${formatVersion}; expected ${OMX4D_FORMAT_VERSION}.`,
    )
  }

  const headerByteLength = preamble.getUint32(12, true)
  if (
    headerByteLength === 0 ||
    headerByteLength > MAX_HEADER_BYTES ||
    headerByteLength > buffer.byteLength - OMX4D_PREAMBLE_BYTES
  ) {
    fail('INVALID_HEADER_LENGTH', 'OMX4D manifest byte length is invalid or out of bounds.')
  }

  const rawManifest = readManifest(buffer, headerByteLength)
  const core = validateCoreManifest(rawManifest)
  const attributes = validateAttributes(
    rawManifest.attributes,
    core,
    align8(OMX4D_PREAMBLE_BYTES + headerByteLength),
    buffer.byteLength,
  )
  const manifest: Omx4dManifest = {
    ...rawManifest,
    ...core,
    attributes,
  } as Omx4dManifest
  const typedAttributes = createTypedAttributes(buffer, attributes)
  validateAttributeData(typedAttributes, core.sourceViewCount)

  const metadata: Omx4dMetadata = {
    formatVersion: OMX4D_FORMAT_VERSION,
    schemaVersion: core.schemaVersion,
    name: core.name,
    fps: core.fps,
    frameCount: core.frameCount,
    durationSeconds: core.durationSeconds,
    sourceViewCount: core.sourceViewCount,
    pointCount: core.pointCount,
    coordinateSystem: core.coordinateSystem,
    units: core.units,
    primitive: core.primitive,
    bounds: core.bounds,
    sampling: core.sampling,
    warnings: core.warnings,
    headerByteLength,
    payloadByteLength: buffer.byteLength,
  }

  const dataset: Omx4dDataset = {
    manifest,
    metadata,
    attributes: typedAttributes,
    positions: typedAttributes.positions,
    colors: typedAttributes.colors,
    dynamicScore: typedAttributes.dynamicScore,
    sourceView: typedAttributes.sourceView,
    cameraPose: typedAttributes.cameraPose,
    intrinsics: typedAttributes.intrinsics,
    transferables: [buffer],
    framePositions(frameIndex) {
      if (!Number.isInteger(frameIndex) || frameIndex < 0 || frameIndex >= core.frameCount) {
        throw new RangeError(
          `frameIndex must be an integer in [0, ${core.frameCount}); received ${frameIndex}.`,
        )
      }
      const valuesPerFrame = core.pointCount * 3
      const start = frameIndex * valuesPerFrame
      return typedAttributes.positions.subarray(start, start + valuesPerFrame)
    },
  }

  Object.defineProperty(dataset, 'framePositions', {
    value: dataset.framePositions,
    enumerable: false,
  })
  return dataset
}

/** Return each unique backing ArrayBuffer exactly once for `postMessage` transfer lists. */
export function getOmx4dTransferables(dataset: Omx4dDataset): ArrayBuffer[] {
  const result: ArrayBuffer[] = []
  const seen = new Set<ArrayBuffer>()
  for (const value of Object.values(dataset.attributes)) {
    const buffer = value.buffer as ArrayBuffer
    if (!seen.has(buffer)) {
      seen.add(buffer)
      result.push(buffer)
    }
  }
  return result
}

/** Return a zero-copy xyz view for one discrete inference frame. */
export function getOmx4dFramePositions(
  dataset: Omx4dDataset,
  frameIndex: number,
): Float32Array {
  if (!Number.isInteger(frameIndex) || frameIndex < 0 || frameIndex >= dataset.metadata.frameCount) {
    throw new RangeError(
      `frameIndex must be an integer in [0, ${dataset.metadata.frameCount}); received ${frameIndex}.`,
    )
  }
  const valuesPerFrame = dataset.metadata.pointCount * 3
  const start = frameIndex * valuesPerFrame
  return dataset.attributes.positions.subarray(start, start + valuesPerFrame)
}
