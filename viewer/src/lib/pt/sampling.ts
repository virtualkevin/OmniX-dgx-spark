import { assertFiniteFloat32, type OmnixTensorSchema } from './schema'

export const DYNAMIC_THRESHOLD = 0.5
export const DYNAMIC_RESERVED_FRACTION = 0.25
export const SAMPLING_METHOD = 'dynamic-reserved-voxel-v1'

const CANCEL_CHECK_INTERVAL = 256 * 1024
const MAX_VOXEL_CELLS = 32_000_000

export interface SamplingProgress {
  pass: 'bounds' | 'voxels'
  completed: number
  total: number
}

export interface SamplingCallbacks {
  onProgress?: (progress: SamplingProgress) => void
  isCancelled?: () => boolean
}

export interface SamplingResult {
  identities: Uint32Array
  requestedPointCount: number
  selectedPointCount: number
  validCandidateCount: number
  dynamicSelectedPointCount: number
  dynamicThreshold: number
  dynamicReservedFraction: number
}

export type FrameZeroReader = (sourceView: number) => Promise<Float32Array>

export class SamplingCancelledError extends Error {
  constructor() {
    super('Point sampling was cancelled.')
    this.name = 'SamplingCancelledError'
  }
}

function checkpoint(callbacks: SamplingCallbacks | undefined): void {
  if (callbacks?.isCancelled?.()) throw new SamplingCancelledError()
}

function roundHalfEven(value: number): number {
  const lower = Math.floor(value)
  const fraction = value - lower
  if (fraction < 0.5) return lower
  if (fraction > 0.5) return lower + 1
  return lower % 2 === 0 ? lower : lower + 1
}

function isBetter(
  score: number,
  identity: number,
  otherScore: number,
  otherIdentity: number,
): boolean {
  return score > otherScore || (score === otherScore && identity < otherIdentity)
}

function isWorse(
  score: number,
  identity: number,
  otherScore: number,
  otherIdentity: number,
): boolean {
  return score < otherScore || (score === otherScore && identity > otherIdentity)
}

/** Fixed-capacity heap whose root is the worst currently selected candidate. */
class DynamicTopHeap {
  private readonly scores: Float32Array
  private readonly identities: Uint32Array
  private count = 0

  constructor(readonly capacity: number) {
    this.scores = new Float32Array(capacity)
    this.identities = new Uint32Array(capacity)
  }

  get size(): number {
    return this.count
  }

  consider(score: number, identity: number): void {
    if (this.capacity === 0) return
    if (this.count < this.capacity) {
      const index = this.count
      this.count += 1
      this.scores[index] = score
      this.identities[index] = identity
      this.bubbleUp(index)
      return
    }
    if (!isBetter(score, identity, this.scores[0], this.identities[0])) return
    this.scores[0] = score
    this.identities[0] = identity
    this.bubbleDown(0)
  }

  mark(target: Uint8Array): void {
    for (let index = 0; index < this.count; index += 1) {
      target[this.identities[index]] = 1
    }
  }

  private swap(left: number, right: number): void {
    const score = this.scores[left]
    const identity = this.identities[left]
    this.scores[left] = this.scores[right]
    this.identities[left] = this.identities[right]
    this.scores[right] = score
    this.identities[right] = identity
  }

  private bubbleUp(start: number): void {
    let index = start
    while (index > 0) {
      const parent = Math.floor((index - 1) / 2)
      if (
        !isWorse(
          this.scores[index],
          this.identities[index],
          this.scores[parent],
          this.identities[parent],
        )
      ) {
        break
      }
      this.swap(index, parent)
      index = parent
    }
  }

  private bubbleDown(start: number): void {
    let index = start
    for (;;) {
      const left = index * 2 + 1
      if (left >= this.count) return
      const right = left + 1
      let worseChild = left
      if (
        right < this.count &&
        isWorse(
          this.scores[right],
          this.identities[right],
          this.scores[left],
          this.identities[left],
        )
      ) {
        worseChild = right
      }
      if (
        !isWorse(
          this.scores[worseChild],
          this.identities[worseChild],
          this.scores[index],
          this.identities[index],
        )
      ) {
        return
      }
      this.swap(index, worseChild)
      index = worseChild
    }
  }
}

function frameForSource(
  values: Float32Array,
  expectedElements: number,
  sourceView: number,
): Float32Array {
  if (values.length !== expectedElements) {
    throw new Error(`Trajectory frame ${sourceView} has an invalid storage length.`)
  }
  assertFiniteFloat32('trajectory', values)
  return values
}

function markEvenlySpaced(
  selected: Uint8Array,
  availableCount: number,
  count: number,
): void {
  if (count <= 0) return
  if (count > availableCount) throw new Error('Sampler requested more candidates than available.')
  let targetIndex = 0
  let targetOrdinal = Math.floor((targetIndex + 0.5) * (availableCount / count))
  let ordinal = 0
  for (let identity = 0; identity < selected.length && targetIndex < count; identity += 1) {
    if (selected[identity] !== 0) continue
    if (ordinal === targetOrdinal) {
      selected[identity] = 1
      targetIndex += 1
      targetOrdinal =
        targetIndex < count
          ? Math.floor((targetIndex + 0.5) * (availableCount / count))
          : -1
    }
    ordinal += 1
  }
  if (targetIndex !== count) throw new Error('Sampler could not fill the requested point count.')
}

function sortedSelectedIdentities(selected: Uint8Array, count: number): Uint32Array {
  const result = new Uint32Array(count)
  let cursor = 0
  for (let identity = 0; identity < selected.length; identity += 1) {
    if (selected[identity] === 0) continue
    if (cursor >= result.length) throw new Error('Sampler selected too many point identities.')
    result[cursor] = identity
    cursor += 1
  }
  if (cursor !== result.length) throw new Error('Sampler selected too few point identities.')
  return result
}

/**
 * Select stable source-pixel identities once for every trajectory frame.
 * Frame zero is requested at most twice per source and is never retained.
 */
export async function samplePointIdentities(
  schema: OmnixTensorSchema,
  dynamicScore: Float32Array,
  pointBudget: number,
  readFrameZero: FrameZeroReader,
  callbacks?: SamplingCallbacks,
): Promise<SamplingResult> {
  if (!Number.isSafeInteger(pointBudget) || pointBudget <= 0) {
    throw new RangeError('Point budget must be a positive safe integer.')
  }
  if (dynamicScore.length !== schema.identityCount) {
    throw new Error('Dynamic-score storage length does not match the trajectory schema.')
  }
  checkpoint(callbacks)

  const selectedPointCount = Math.min(pointBudget, schema.identityCount)
  const reserveTarget = Math.min(
    selectedPointCount,
    roundHalfEven(selectedPointCount * DYNAMIC_RESERVED_FRACTION),
  )

  let dynamicCandidateCount = 0
  if (selectedPointCount === schema.identityCount) {
    const identities = new Uint32Array(schema.identityCount)
    for (let identity = 0; identity < schema.identityCount; identity += 1) {
      if (dynamicScore[identity] >= DYNAMIC_THRESHOLD) dynamicCandidateCount += 1
      identities[identity] = identity
      if (identity % CANCEL_CHECK_INTERVAL === 0) checkpoint(callbacks)
    }
    return {
      identities,
      requestedPointCount: pointBudget,
      selectedPointCount,
      validCandidateCount: schema.identityCount,
      dynamicSelectedPointCount: Math.min(reserveTarget, dynamicCandidateCount),
      dynamicThreshold: DYNAMIC_THRESHOLD,
      dynamicReservedFraction: DYNAMIC_RESERVED_FRACTION,
    }
  }

  const heap = new DynamicTopHeap(reserveTarget)
  for (let identity = 0; identity < dynamicScore.length; identity += 1) {
    const score = dynamicScore[identity]
    if (score >= DYNAMIC_THRESHOLD) {
      dynamicCandidateCount += 1
      heap.consider(score, identity)
    }
    if (identity % CANCEL_CHECK_INTERVAL === 0) checkpoint(callbacks)
  }

  const selected = new Uint8Array(schema.identityCount)
  heap.mark(selected)
  const dynamicSelectedPointCount = heap.size
  const spatialTarget = selectedPointCount - dynamicSelectedPointCount
  const spatialCandidateCount = schema.identityCount - dynamicSelectedPointCount
  const expectedFrameElements = schema.identitiesPerView * 3

  const minimum = [Infinity, Infinity, Infinity]
  const maximum = [-Infinity, -Infinity, -Infinity]
  for (let source = 0; source < schema.sourceViewCount; source += 1) {
    checkpoint(callbacks)
    const frame = frameForSource(await readFrameZero(source), expectedFrameElements, source)
    const identityBase = source * schema.identitiesPerView
    for (let pixel = 0; pixel < schema.identitiesPerView; pixel += 1) {
      const identity = identityBase + pixel
      if (selected[identity] !== 0) continue
      const offset = pixel * 3
      for (let axis = 0; axis < 3; axis += 1) {
        const value = frame[offset + axis]
        if (value < minimum[axis]) minimum[axis] = value
        if (value > maximum[axis]) maximum[axis] = value
      }
    }
    callbacks?.onProgress?.({ pass: 'bounds', completed: source + 1, total: schema.sourceViewCount })
  }

  const extent = maximum.map((value, axis) => Math.fround(value - minimum[axis]))
  const active = extent.map((value) => value > 1e-12)
  const activeCount = active.filter(Boolean).length
  if (activeCount === 0) {
    markEvenlySpaced(selected, spatialCandidateCount, spatialTarget)
    return {
      identities: sortedSelectedIdentities(selected, selectedPointCount),
      requestedPointCount: pointBudget,
      selectedPointCount,
      validCandidateCount: schema.identityCount,
      dynamicSelectedPointCount,
      dynamicThreshold: DYNAMIC_THRESHOLD,
      dynamicReservedFraction: DYNAMIC_RESERVED_FRACTION,
    }
  }

  let cellsPerAxis = Math.max(1, Math.ceil((spatialTarget * 2) ** (1 / activeCount)))
  const dimensions = active.map((enabled) => (enabled ? cellsPerAxis : 1))
  let cellCount = dimensions[0] * dimensions[1] * dimensions[2]
  if (cellCount > MAX_VOXEL_CELLS) {
    cellsPerAxis = Math.max(1, Math.floor(MAX_VOXEL_CELLS ** (1 / activeCount)))
    for (let axis = 0; axis < 3; axis += 1) dimensions[axis] = active[axis] ? cellsPerAxis : 1
    cellCount = dimensions[0] * dimensions[1] * dimensions[2]
  }

  const firstByCell = new Int32Array(cellCount)
  firstByCell.fill(-1)
  let representativeCount = 0
  for (let source = 0; source < schema.sourceViewCount; source += 1) {
    checkpoint(callbacks)
    const frame = frameForSource(await readFrameZero(source), expectedFrameElements, source)
    const identityBase = source * schema.identitiesPerView
    for (let pixel = 0; pixel < schema.identitiesPerView; pixel += 1) {
      const identity = identityBase + pixel
      if (selected[identity] !== 0) continue
      const offset = pixel * 3
      const cell = [0, 0, 0]
      for (let axis = 0; axis < 3; axis += 1) {
        if (!active[axis]) continue
        const difference = Math.fround(frame[offset + axis] - minimum[axis])
        const normalized = Math.min(1, Math.max(0, Math.fround(difference / extent[axis])))
        const scaled = Math.fround(normalized * dimensions[axis])
        cell[axis] = Math.min(dimensions[axis] - 1, Math.floor(scaled))
      }
      const key = (cell[0] * dimensions[1] + cell[1]) * dimensions[2] + cell[2]
      if (firstByCell[key] === -1) {
        firstByCell[key] = identity
        representativeCount += 1
      }
    }
    callbacks?.onProgress?.({ pass: 'voxels', completed: source + 1, total: schema.sourceViewCount })
  }

  const representatives = new Uint32Array(representativeCount)
  let representativeCursor = 0
  for (let key = 0; key < firstByCell.length; key += 1) {
    if (firstByCell[key] < 0) continue
    representatives[representativeCursor] = firstByCell[key]
    representativeCursor += 1
  }

  if (representativeCount >= spatialTarget) {
    for (let index = 0; index < spatialTarget; index += 1) {
      const position = Math.floor((index + 0.5) * (representativeCount / spatialTarget))
      selected[representatives[position]] = 1
    }
  } else {
    for (let index = 0; index < representatives.length; index += 1) {
      selected[representatives[index]] = 1
    }
    const fillCount = spatialTarget - representativeCount
    markEvenlySpaced(selected, spatialCandidateCount - representativeCount, fillCount)
  }

  return {
    identities: sortedSelectedIdentities(selected, selectedPointCount),
    requestedPointCount: pointBudget,
    selectedPointCount,
    validCandidateCount: schema.identityCount,
    dynamicSelectedPointCount,
    dynamicThreshold: DYNAMIC_THRESHOLD,
    dynamicReservedFraction: DYNAMIC_RESERVED_FRACTION,
  }
}
