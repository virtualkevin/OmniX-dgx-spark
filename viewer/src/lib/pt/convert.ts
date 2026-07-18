import type { ViewerDataset } from '../dataset'
import { MAX_BROWSER_POINT_BUDGET } from '../limits'
import type {
  Omx4dAttributeDescriptor,
  Omx4dAttributeManifest,
  Omx4dManifest,
} from '../omx4d'
import { parseTorchPickle } from './pickle'
import {
  SAMPLING_METHOD,
  SamplingCancelledError,
  samplePointIdentities,
} from './sampling'
import {
  assertFiniteFloat32,
  validateAndConvertCameras,
  validateDynamicScores,
  validateOmnixTensorMap,
  type OmnixTensorSchema,
} from './schema'
import { openTorchZip, type TorchZipArchive, type ZipEntry } from './zip'

const MIB = 1024 * 1024
const GIB = 1024 * MIB
const MAX_PICKLE_BYTES = MIB
const MAX_OUTPUT_BYTES = 2 * GIB
const FLOAT32_BYTES = Float32Array.BYTES_PER_ELEMENT

const HOST_IS_LITTLE_ENDIAN = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1

export type PtConversionPhase =
  | 'archive'
  | 'metadata'
  | 'sampling'
  | 'trajectory'
  | 'finalizing'

export interface PtConversionProgress {
  phase: PtConversionPhase
  completed: number
  total: number
  /** Worker/UI-compatible aliases derived from the exact counters above. */
  stage: PtConversionPhase
  progress: number
  message: string
}

export interface PtConversionOptions {
  pointBudget: number
  fps: number
  name: string
}

export interface PtConversionCallbacks {
  onProgress?: (progress: PtConversionProgress) => void
  isCancelled?: () => boolean
}

export type PtConversionErrorCode =
  | 'INVALID_OPTIONS'
  | 'STORAGE_MISMATCH'
  | 'CRC_MISMATCH'
  | 'RESOURCE_LIMIT'
  | 'NON_FINITE_TRAJECTORY'
  | 'CANCELLED'

export class PtConversionError extends Error {
  readonly code: PtConversionErrorCode

  constructor(code: PtConversionErrorCode, message: string, options?: ErrorOptions) {
    super(message, options)
    this.name = 'PtConversionError'
    this.code = code
  }
}

function fail(code: PtConversionErrorCode, message: string, cause?: unknown): never {
  throw new PtConversionError(code, message, cause === undefined ? undefined : { cause })
}

function checkpoint(callbacks: PtConversionCallbacks | undefined): void {
  if (callbacks?.isCancelled?.()) fail('CANCELLED', 'The .pt conversion was cancelled.')
}

function emit(
  callbacks: PtConversionCallbacks | undefined,
  phase: PtConversionPhase,
  completed: number,
  total: number,
): void {
  const messages: Record<PtConversionPhase, string> = {
    archive: 'Inspecting the torch.save archive…',
    metadata: 'Validating tensor metadata…',
    sampling: 'Selecting stable point identities…',
    trajectory: 'Reading trajectory frames…',
    finalizing: 'Preparing the renderer…',
  }
  callbacks?.onProgress?.({
    phase,
    completed,
    total,
    stage: phase,
    progress: total > 0 ? Math.min(1, Math.max(0, completed / total)) : 0,
    message: messages[phase],
  })
}

function validateOptions(options: PtConversionOptions): PtConversionOptions {
  if (!Number.isSafeInteger(options.pointBudget) || options.pointBudget <= 0) {
    fail('INVALID_OPTIONS', 'Point budget must be a positive safe integer.')
  }
  if (options.pointBudget > MAX_BROWSER_POINT_BUDGET) {
    fail('INVALID_OPTIONS', 'Point budget must not exceed the browser maximum of 500,000.')
  }
  if (!Number.isFinite(options.fps) || options.fps < 0.1 || options.fps > 240) {
    fail('INVALID_OPTIONS', 'Playback FPS must be between 0.1 and 240.')
  }
  const basename = options.name.replace(/\\/g, '/').split('/').pop() ?? ''
  const stem = basename.replace(/\.[^.]*$/, '')
  const name = stem.replace(/[^A-Za-z0-9._ -]+/g, '_').replace(/^[ ._]+|[ ._]+$/g, '')
  return {
    pointBudget: options.pointBudget,
    fps: options.fps,
    name: (name || 'OmniX predictions').slice(0, 80),
  }
}

const CRC32_TABLE = (() => {
  const table = new Uint32Array(256)
  for (let index = 0; index < table.length; index += 1) {
    let value = index
    for (let bit = 0; bit < 8; bit += 1) {
      value = (value & 1) !== 0 ? 0xedb88320 ^ (value >>> 1) : value >>> 1
    }
    table[index] = value >>> 0
  }
  return table
})()

class Crc32 {
  private value = 0xffffffff

  update(bytes: Uint8Array): void {
    let value = this.value
    for (let index = 0; index < bytes.length; index += 1) {
      value = CRC32_TABLE[(value ^ bytes[index]) & 0xff] ^ (value >>> 8)
    }
    this.value = value
  }

  digest(): number {
    return (this.value ^ 0xffffffff) >>> 0
  }
}

function fullEntryName(archive: TorchZipArchive, relativeName: string): string {
  return archive.prefix.length === 0 ? relativeName : `${archive.prefix}/${relativeName}`
}

function requireEntry(archive: TorchZipArchive, relativeName: string): ZipEntry {
  const entry = archive.entries.get(fullEntryName(archive, relativeName))
  if (entry === undefined) fail('STORAGE_MISMATCH', 'The .pt archive is missing tensor data.')
  return entry
}

function verifyCrc(entry: ZipEntry, bytes: Uint8Array): void {
  const crc = new Crc32()
  crc.update(bytes)
  if (crc.digest() !== (entry.crc32 >>> 0)) {
    fail('CRC_MISMATCH', 'A .pt archive member failed its integrity check.')
  }
}

async function readVerifiedEntry(
  archive: TorchZipArchive,
  relativeName: string,
  maxBytes: number,
): Promise<Uint8Array> {
  const entry = requireEntry(archive, relativeName)
  const bytes = await archive.readEntry(relativeName, maxBytes)
  verifyCrc(entry, bytes)
  return bytes
}

function asFloat32(bytes: Uint8Array): Float32Array {
  if (bytes.byteLength % FLOAT32_BYTES !== 0) {
    fail('STORAGE_MISMATCH', 'A float32 storage has an invalid byte length.')
  }
  if (HOST_IS_LITTLE_ENDIAN && bytes.byteOffset % FLOAT32_BYTES === 0) {
    return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / FLOAT32_BYTES)
  }
  const result = new Float32Array(bytes.byteLength / FLOAT32_BYTES)
  const data = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  for (let index = 0; index < result.length; index += 1) {
    result[index] = data.getFloat32(index * FLOAT32_BYTES, true)
  }
  return result
}

function storageName(key: string): string {
  return `data/${key}`
}

function validateStorageMembers(archive: TorchZipArchive, schema: OmnixTensorSchema): void {
  const descriptors = [
    schema.trajectory,
    schema.cameraPose,
    schema.intrinsics,
    schema.dynamicScore,
  ]
  const expected = new Set(descriptors.map((descriptor) => storageName(descriptor.storage.key)))
  const dataPrefix = fullEntryName(archive, 'data/')
  const actual = [...archive.entries.keys()]
    .filter((name) => name.startsWith(dataPrefix))
    .map((name) => name.slice(archive.prefix.length === 0 ? 0 : archive.prefix.length + 1))
  if (actual.length !== expected.size || actual.some((name) => !expected.has(name))) {
    fail('STORAGE_MISMATCH', 'The .pt tensor storage set does not match its metadata.')
  }

  for (const descriptor of descriptors) {
    const entry = requireEntry(archive, storageName(descriptor.storage.key))
    if (
      entry.uncompressedSize !== descriptor.storage.byteLength ||
      entry.compressedSize !== descriptor.storage.byteLength ||
      entry.dataOffset % 64 !== 0
    ) {
      fail('STORAGE_MISMATCH', 'A .pt tensor storage does not match its descriptor.')
    }
  }
}

function projectedOutputBytes(schema: OmnixTensorSchema, pointCount: number): number {
  const positions = schema.frameCount * pointCount * 3 * FLOAT32_BYTES
  const attributes = pointCount * (3 + FLOAT32_BYTES + Uint16Array.BYTES_PER_ELEMENT)
  const calibration = schema.sourceViewCount * (16 + 9) * FLOAT32_BYTES
  const total = positions + attributes + calibration
  if (!Number.isSafeInteger(total) || total > MAX_OUTPUT_BYTES) {
    fail('RESOURCE_LIMIT', 'The requested point budget exceeds the browser output limit.')
  }
  return total
}

function hsvToRgb(hue: number, saturation: number, value: number): [number, number, number] {
  const sector = Math.floor(hue * 6)
  const fraction = hue * 6 - sector
  const p = value * (1 - saturation)
  const q = value * (1 - fraction * saturation)
  const t = value * (1 - (1 - fraction) * saturation)
  const candidates: [number, number, number][] = [
    [value, t, p],
    [q, value, p],
    [p, value, t],
    [p, q, value],
    [t, p, value],
    [value, p, q],
  ]
  return candidates[sector % 6]
}

function sourcePalette(sourceViewCount: number): Uint8Array {
  const result = new Uint8Array(sourceViewCount * 3)
  const goldenRatio = 0.6180339887498949
  for (let source = 0; source < sourceViewCount; source += 1) {
    const [red, green, blue] = hsvToRgb((0.08 + source * goldenRatio) % 1, 0.62, 0.95)
    result[source * 3] = Math.round(red * 255)
    result[source * 3 + 1] = Math.round(green * 255)
    result[source * 3 + 2] = Math.round(blue * 255)
  }
  return result
}

function buildPointAttributes(
  identities: Uint32Array,
  dynamicScore: Float32Array,
  schema: OmnixTensorSchema,
): { colors: Uint8Array; selectedScore: Float32Array; sourceView: Uint16Array } {
  const colors = new Uint8Array(identities.length * 3)
  const selectedScore = new Float32Array(identities.length)
  const sourceView = new Uint16Array(identities.length)
  const palette = sourcePalette(schema.sourceViewCount)
  for (let index = 0; index < identities.length; index += 1) {
    const identity = identities[index]
    const source = Math.floor(identity / schema.identitiesPerView)
    const score = dynamicScore[identity]
    const blend = Math.min(1, Math.max(0, score)) * 0.45
    selectedScore[index] = score
    sourceView[index] = source
    for (let channel = 0; channel < 3; channel += 1) {
      const base = palette[source * 3 + channel]
      const accent = channel === 0 ? 255 : channel === 1 ? 72 : 86
      colors[index * 3 + channel] = Math.round(base * (1 - blend) + accent * blend)
    }
  }
  return { colors, selectedScore, sourceView }
}

function align8(value: number): number {
  return Math.ceil(value / 8) * 8
}

function descriptor(
  offset: number,
  byteLength: number,
  dtype: Omx4dAttributeDescriptor['dtype'],
  shape: number[],
): Omx4dAttributeDescriptor {
  return { offset, byteLength, dtype, shape }
}

function logicalAttributes(
  schema: OmnixTensorSchema,
  pointCount: number,
): Omx4dAttributeManifest {
  let offset = 0
  const positionsBytes = schema.frameCount * pointCount * 3 * FLOAT32_BYTES
  const positions = descriptor(offset, positionsBytes, 'float32', [schema.frameCount, pointCount, 3])
  offset = align8(offset + positionsBytes)
  const colorsBytes = pointCount * 3
  const colors = descriptor(offset, colorsBytes, 'uint8', [pointCount, 3])
  offset = align8(offset + colorsBytes)
  const dynamicBytes = pointCount * FLOAT32_BYTES
  const dynamicScore = descriptor(offset, dynamicBytes, 'float32', [pointCount])
  offset = align8(offset + dynamicBytes)
  const sourceBytes = pointCount * Uint16Array.BYTES_PER_ELEMENT
  const sourceView = descriptor(offset, sourceBytes, 'uint16', [pointCount])
  offset = align8(offset + sourceBytes)
  const cameraBytes = schema.sourceViewCount * 16 * FLOAT32_BYTES
  const cameraPose = descriptor(offset, cameraBytes, 'float32', [schema.sourceViewCount, 4, 4])
  offset = align8(offset + cameraBytes)
  const intrinsicsBytes = schema.sourceViewCount * 9 * FLOAT32_BYTES
  const intrinsics = descriptor(offset, intrinsicsBytes, 'float32', [schema.sourceViewCount, 3, 3])
  return { positions, colors, dynamicScore, sourceView, cameraPose, intrinsics }
}

async function gatherTrajectory(
  archive: TorchZipArchive,
  schema: OmnixTensorSchema,
  identities: Uint32Array,
  positions: Float32Array,
  callbacks?: PtConversionCallbacks,
): Promise<{ min: [number, number, number]; max: [number, number, number] }> {
  const trajectoryName = storageName(schema.trajectory.storage.key)
  const trajectoryEntry = requireEntry(archive, trajectoryName)
  const frameElements = schema.identitiesPerView * 3
  const frameBytes = frameElements * FLOAT32_BYTES
  const sourceOffsets = new Uint32Array(schema.sourceViewCount + 1)
  let identityCursor = 0
  for (let source = 0; source < schema.sourceViewCount; source += 1) {
    sourceOffsets[source] = identityCursor
    const nextSourceIdentity = (source + 1) * schema.identitiesPerView
    while (identityCursor < identities.length && identities[identityCursor] < nextSourceIdentity) {
      identityCursor += 1
    }
  }
  sourceOffsets[schema.sourceViewCount] = identities.length

  const minimum: [number, number, number] = [Infinity, Infinity, Infinity]
  const maximum: [number, number, number] = [-Infinity, -Infinity, -Infinity]
  const crc = new Crc32()
  const totalFrames = schema.sourceViewCount * schema.frameCount
  let completedFrames = 0

  for (let source = 0; source < schema.sourceViewCount; source += 1) {
    const selectionStart = sourceOffsets[source]
    const selectionEnd = sourceOffsets[source + 1]
    const identityBase = source * schema.identitiesPerView
    for (let frame = 0; frame < schema.frameCount; frame += 1) {
      checkpoint(callbacks)
      const frameIndex = source * schema.frameCount + frame
      const bytes = await archive.sliceEntry(trajectoryName, frameIndex * frameBytes, frameBytes)
      crc.update(bytes)
      const values = asFloat32(bytes)
      try {
        assertFiniteFloat32('trajectory', values)
      } catch (error) {
        fail('NON_FINITE_TRAJECTORY', 'The trajectory contains NaN or infinite values.', error)
      }
      for (let selected = selectionStart; selected < selectionEnd; selected += 1) {
        const pixel = identities[selected] - identityBase
        const inputOffset = pixel * 3
        const outputOffset = (frame * identities.length + selected) * 3
        const x = values[inputOffset]
        const y = -values[inputOffset + 1]
        const z = -values[inputOffset + 2]
        positions[outputOffset] = x
        positions[outputOffset + 1] = y
        positions[outputOffset + 2] = z
        if (x < minimum[0]) minimum[0] = x
        if (y < minimum[1]) minimum[1] = y
        if (z < minimum[2]) minimum[2] = z
        if (x > maximum[0]) maximum[0] = x
        if (y > maximum[1]) maximum[1] = y
        if (z > maximum[2]) maximum[2] = z
      }
      completedFrames += 1
      emit(callbacks, 'trajectory', completedFrames, totalFrames)
    }
  }
  if (crc.digest() !== (trajectoryEntry.crc32 >>> 0)) {
    fail('CRC_MISMATCH', 'The trajectory storage failed its integrity check.')
  }
  return { min: minimum, max: maximum }
}

/** Convert an exact OmniX torch.save artifact entirely inside the browser. */
export async function convertOmnixPt(
  file: File | Blob,
  requestedOptions: PtConversionOptions,
  callbacks?: PtConversionCallbacks,
): Promise<ViewerDataset> {
  const options = validateOptions(requestedOptions)
  checkpoint(callbacks)
  emit(callbacks, 'archive', 0, 1)
  const archive = await openTorchZip(file)
  emit(callbacks, 'archive', 1, 1)

  checkpoint(callbacks)
  emit(callbacks, 'metadata', 0, 1)
  const pickleBytes = await readVerifiedEntry(archive, 'data.pkl', MAX_PICKLE_BYTES)
  const tensors = parseTorchPickle(pickleBytes)
  const schema = validateOmnixTensorMap(tensors)
  validateStorageMembers(archive, schema)
  const pointCount = Math.min(options.pointBudget, schema.identityCount)
  projectedOutputBytes(schema, pointCount)

  const [dynamicBytes, cameraBytes, intrinsicsBytes] = await Promise.all([
    readVerifiedEntry(
      archive,
      storageName(schema.dynamicScore.storage.key),
      schema.dynamicScore.storage.byteLength,
    ),
    readVerifiedEntry(
      archive,
      storageName(schema.cameraPose.storage.key),
      schema.cameraPose.storage.byteLength,
    ),
    readVerifiedEntry(
      archive,
      storageName(schema.intrinsics.storage.key),
      schema.intrinsics.storage.byteLength,
    ),
  ])
  const dynamicScore = asFloat32(dynamicBytes)
  validateDynamicScores(dynamicScore)
  const { cameraPose, intrinsics } = validateAndConvertCameras(
    asFloat32(cameraBytes),
    asFloat32(intrinsicsBytes),
    schema.sourceViewCount,
  )
  emit(callbacks, 'metadata', 1, 1)

  checkpoint(callbacks)
  const frameBytes = schema.identitiesPerView * 3 * FLOAT32_BYTES
  const trajectoryName = storageName(schema.trajectory.storage.key)
  const sampling = await samplePointIdentities(
    schema,
    dynamicScore,
    options.pointBudget,
    async (sourceView) => {
      const offset = sourceView * schema.frameCount * frameBytes
      return asFloat32(await archive.sliceEntry(trajectoryName, offset, frameBytes))
    },
    {
      isCancelled: callbacks?.isCancelled,
      onProgress: (progress) => {
        const passOffset = progress.pass === 'bounds' ? 0 : schema.sourceViewCount
        emit(
          callbacks,
          'sampling',
          passOffset + progress.completed,
          schema.sourceViewCount * 2,
        )
      },
    },
  ).catch((error: unknown) => {
    if (error instanceof SamplingCancelledError) {
      fail('CANCELLED', 'The .pt conversion was cancelled.', error)
    }
    throw error
  })

  let positions: Float32Array
  try {
    positions = new Float32Array(schema.frameCount * sampling.selectedPointCount * 3)
  } catch (error) {
    fail('RESOURCE_LIMIT', 'The browser could not allocate the requested point trajectory.', error)
  }
  const { colors, selectedScore, sourceView } = buildPointAttributes(
    sampling.identities,
    dynamicScore,
    schema,
  )
  const bounds = await gatherTrajectory(
    archive,
    schema,
    sampling.identities,
    positions,
    callbacks,
  )

  checkpoint(callbacks)
  emit(callbacks, 'finalizing', 0, 1)
  const manifest: Omx4dManifest = {
    schemaVersion: 1,
    name: options.name,
    fps: options.fps,
    frameCount: schema.frameCount,
    durationSeconds: schema.frameCount / options.fps,
    sourceViewCount: schema.sourceViewCount,
    pointCount: sampling.selectedPointCount,
    coordinateSystem: 'threejs-right-handed-y-up',
    units: 'unknown',
    primitive: 'points',
    bounds,
    sampling: {
      method: SAMPLING_METHOD,
      requestedPointCount: sampling.requestedPointCount,
      selectedPointCount: sampling.selectedPointCount,
      validCandidateCount: sampling.validCandidateCount,
      dynamicThreshold: sampling.dynamicThreshold,
      dynamicReservedFraction: sampling.dynamicReservedFraction,
      dynamicSelectedPointCount: sampling.dynamicSelectedPointCount,
    },
    warnings: [
      'Source RGB was not supplied; colors use a stable source/dynamic palette.',
      'The .pt format has no timing metadata; playback uses the selected FPS.',
      'World-space units are not recorded in the OmniX output.',
    ],
    attributes: logicalAttributes(schema, sampling.selectedPointCount),
  }
  const dataset: ViewerDataset = {
    manifest,
    positions,
    colors,
    dynamicScore: selectedScore,
    sourceView,
    cameraPose,
    intrinsics,
  }
  emit(callbacks, 'finalizing', 1, 1)
  return dataset
}
