import {
  PtParseError,
  type PtParseErrorCode,
  type StorageReference,
  type TensorDescriptor,
} from './types'
import { MAX_BROWSER_TENSOR_BYTES } from '../limits'

export { PtParseError } from './types'
export type { StorageReference, TensorDescriptor } from './types'

const MAX_PICKLE_BYTES = 1024 * 1024
const MAX_OPCODE_COUNT = 10_000
const MAX_STACK_DEPTH = 256
const MAX_MEMO_ENTRIES = 512
const MAX_STRING_BYTES = 256
const MAX_STORAGE_ELEMENTS = MAX_BROWSER_TENSOR_BYTES / Float32Array.BYTES_PER_ELEMENT
const MAX_TOTAL_TENSOR_BYTES = MAX_BROWSER_TENSOR_BYTES
const MAX_SOURCE_VIEWS = 64
const MAX_FRAMES = 600
const MAX_SOURCE_PIXELS = 16_000_000

const REQUIRED_KEYS = [
  'trajectory',
  'camera_pose',
  'intrinsics',
  'pts3d_dynamic_score',
] as const

type RequiredKey = (typeof REQUIRED_KEYS)[number]
type GlobalId = 'rebuild-tensor-v2' | 'float-storage' | 'ordered-dict'

interface GlobalValue {
  readonly kind: 'global'
  readonly id: GlobalId
}

interface TupleValue {
  readonly kind: 'tuple'
  readonly items: unknown[]
}

interface DictValue {
  readonly kind: 'dict'
  readonly entries: Map<string, unknown>
}

interface StorageValue {
  readonly kind: 'storage'
  readonly reference: StorageReference
}

interface TensorValue {
  readonly kind: 'tensor'
  readonly descriptor: TensorDescriptor
}

const MARK = Symbol('pickle-mark')
const EMPTY_HOOKS = Symbol('empty-hooks')

const GLOBALS = new Map<string, GlobalValue>([
  ['torch._utils\n_rebuild_tensor_v2', { kind: 'global', id: 'rebuild-tensor-v2' }],
  ['torch\nFloatStorage', { kind: 'global', id: 'float-storage' }],
  ['collections\nOrderedDict', { kind: 'global', id: 'ordered-dict' }],
])

function fail(code: PtParseErrorCode, message: string, cause?: unknown): never {
  throw new PtParseError(code, message, cause === undefined ? undefined : { cause })
}

function isTuple(value: unknown): value is TupleValue {
  return typeof value === 'object' && value !== null && (value as TupleValue).kind === 'tuple'
}

function isDict(value: unknown): value is DictValue {
  return typeof value === 'object' && value !== null && (value as DictValue).kind === 'dict'
}

function isGlobal(value: unknown, id?: GlobalId): value is GlobalValue {
  return (
    typeof value === 'object' &&
    value !== null &&
    (value as GlobalValue).kind === 'global' &&
    (id === undefined || (value as GlobalValue).id === id)
  )
}

function isStorage(value: unknown): value is StorageValue {
  return typeof value === 'object' && value !== null && (value as StorageValue).kind === 'storage'
}

function isTensor(value: unknown): value is TensorValue {
  return typeof value === 'object' && value !== null && (value as TensorValue).kind === 'tensor'
}

function safeInteger(value: unknown): value is number {
  return typeof value === 'number' && Number.isSafeInteger(value)
}

function positiveIntegerTuple(value: unknown, field: string): number[] {
  if (
    !isTuple(value) ||
    value.items.length === 0 ||
    value.items.length > 5 ||
    value.items.some((item) => !safeInteger(item) || item <= 0)
  ) {
    fail('INVALID_TENSOR_SCHEMA', `The ${field} tuple is invalid.`)
  }
  return value.items as number[]
}

function exactShape(actual: readonly number[], expected: readonly number[]): boolean {
  return actual.length === expected.length && actual.every((value, index) => value === expected[index])
}

class RestrictedPickleReader {
  private readonly bytes: Uint8Array
  private readonly data: DataView
  private readonly stack: unknown[] = []
  private readonly memo = new Map<number, unknown>()
  private readonly storageKeys = new Map<string, StorageReference>()
  private cursor = 0
  private opcodeCount = 0
  private protocolSeen = false

  constructor(bytes: Uint8Array) {
    this.bytes = bytes
    this.data = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  }

  run(): Map<string, TensorDescriptor> {
    while (this.cursor < this.bytes.byteLength) {
      this.opcodeCount += 1
      if (this.opcodeCount > MAX_OPCODE_COUNT) {
        fail('RESOURCE_LIMIT', 'The pickle metadata contains too many operations.')
      }
      const opcode = this.readUint8()
      switch (opcode) {
        case 0x80: // PROTO
          this.readProtocol()
          break
        case 0x7d: // EMPTY_DICT
          this.requireProtocol()
          this.push({ kind: 'dict', entries: new Map() } satisfies DictValue)
          break
        case 0x28: // MARK
          this.requireProtocol()
          this.push(MARK)
          break
        case 0x58: // BINUNICODE
          this.requireProtocol()
          this.push(this.readUnicode())
          break
        case 0x63: // GLOBAL
          this.requireProtocol()
          this.push(this.readGlobal())
          break
        case 0x71: // BINPUT
          this.requireProtocol()
          this.memoize(this.readUint8())
          break
        case 0x68: // BINGET
          this.requireProtocol()
          this.recall(this.readUint8())
          break
        case 0x4a: // BININT
          this.requireProtocol()
          this.push(this.readInt32())
          break
        case 0x4b: // BININT1
          this.requireProtocol()
          this.push(this.readUint8())
          break
        case 0x4d: // BININT2
          this.requireProtocol()
          this.push(this.readUint16())
          break
        case 0x74: // TUPLE
          this.requireProtocol()
          this.buildMarkedTuple()
          break
        case 0x87: // TUPLE3
          this.requireProtocol()
          this.buildTuple3()
          break
        case 0x51: // BINPERSID
          this.requireProtocol()
          this.buildStorageReference()
          break
        case 0x89: // NEWFALSE
          this.requireProtocol()
          this.push(false)
          break
        case 0x29: // EMPTY_TUPLE
          this.requireProtocol()
          this.push({ kind: 'tuple', items: [] } satisfies TupleValue)
          break
        case 0x52: // REDUCE
          this.requireProtocol()
          this.reduce()
          break
        case 0x75: // SETITEMS
          this.requireProtocol()
          this.setItems()
          break
        case 0x2e: // STOP
          this.requireProtocol()
          return this.finish()
        default:
          fail('UNSUPPORTED_PICKLE', 'The pickle uses an opcode outside the safe OmniX subset.')
      }
    }
    fail('INVALID_PICKLE', 'The pickle metadata ended before STOP.')
  }

  private readProtocol(): void {
    if (this.protocolSeen || this.opcodeCount !== 1 || this.stack.length !== 0) {
      fail('INVALID_PICKLE', 'The pickle protocol marker is misplaced.')
    }
    if (this.readUint8() !== 2) {
      fail('UNSUPPORTED_PICKLE', 'Only torch.save pickle protocol 2 is supported.')
    }
    this.protocolSeen = true
  }

  private requireProtocol(): void {
    if (!this.protocolSeen) {
      fail('INVALID_PICKLE', 'The pickle is missing its protocol marker.')
    }
  }

  private readUint8(): number {
    if (this.cursor + 1 > this.bytes.byteLength) {
      fail('INVALID_PICKLE', 'The pickle metadata is truncated.')
    }
    return this.bytes[this.cursor++]
  }

  private readUint16(): number {
    if (this.cursor + 2 > this.bytes.byteLength) {
      fail('INVALID_PICKLE', 'The pickle metadata is truncated.')
    }
    const value = this.data.getUint16(this.cursor, true)
    this.cursor += 2
    return value
  }

  private readUint32(): number {
    if (this.cursor + 4 > this.bytes.byteLength) {
      fail('INVALID_PICKLE', 'The pickle metadata is truncated.')
    }
    const value = this.data.getUint32(this.cursor, true)
    this.cursor += 4
    return value
  }

  private readInt32(): number {
    if (this.cursor + 4 > this.bytes.byteLength) {
      fail('INVALID_PICKLE', 'The pickle metadata is truncated.')
    }
    const value = this.data.getInt32(this.cursor, true)
    this.cursor += 4
    return value
  }

  private readUnicode(): string {
    const length = this.readUint32()
    if (length > MAX_STRING_BYTES || this.cursor + length > this.bytes.byteLength) {
      fail('RESOURCE_LIMIT', 'A pickle string is too large or truncated.')
    }
    const encoded = this.bytes.subarray(this.cursor, this.cursor + length)
    this.cursor += length
    try {
      return new TextDecoder('utf-8', { fatal: true }).decode(encoded)
    } catch (error) {
      fail('INVALID_PICKLE', 'A pickle string has invalid UTF-8 encoding.', error)
    }
  }

  private readAsciiLine(): string {
    const start = this.cursor
    while (this.cursor < this.bytes.byteLength && this.bytes[this.cursor] !== 0x0a) {
      const byte = this.bytes[this.cursor]
      if (byte < 0x20 || byte > 0x7e || this.cursor - start >= MAX_STRING_BYTES) {
        fail('INVALID_PICKLE', 'A pickle global name is malformed.')
      }
      this.cursor += 1
    }
    if (this.cursor >= this.bytes.byteLength || this.cursor === start) {
      fail('INVALID_PICKLE', 'A pickle global name is truncated or empty.')
    }
    const result = String.fromCharCode(...this.bytes.subarray(start, this.cursor))
    this.cursor += 1
    return result
  }

  private readGlobal(): GlobalValue {
    const moduleName = this.readAsciiLine()
    const symbolName = this.readAsciiLine()
    const result = GLOBALS.get(`${moduleName}\n${symbolName}`)
    if (result === undefined) {
      fail('UNSAFE_PICKLE', 'The pickle references a global outside the safe OmniX allowlist.')
    }
    return result
  }

  private push(value: unknown): void {
    this.stack.push(value)
    if (this.stack.length > MAX_STACK_DEPTH) {
      fail('RESOURCE_LIMIT', 'The pickle metadata stack is too deep.')
    }
  }

  private pop(): unknown {
    if (this.stack.length === 0) {
      fail('INVALID_PICKLE', 'The pickle metadata stack underflowed.')
    }
    return this.stack.pop()
  }

  private memoize(index: number): void {
    if (this.stack.length === 0 || index !== this.memo.size || index >= MAX_MEMO_ENTRIES) {
      fail('INVALID_PICKLE', 'The pickle memo table is malformed.')
    }
    this.memo.set(index, this.stack[this.stack.length - 1])
  }

  private recall(index: number): void {
    const value = this.memo.get(index)
    if (value === undefined) {
      fail('INVALID_PICKLE', 'The pickle references an unknown memo entry.')
    }
    this.push(value)
  }

  private lastMark(): number {
    for (let index = this.stack.length - 1; index >= 0; index -= 1) {
      if (this.stack[index] === MARK) {
        return index
      }
    }
    fail('INVALID_PICKLE', 'The pickle contains an unmatched MARK.')
  }

  private buildMarkedTuple(): void {
    const markIndex = this.lastMark()
    const items = this.stack.slice(markIndex + 1)
    this.stack.length = markIndex
    this.push({ kind: 'tuple', items } satisfies TupleValue)
  }

  private buildTuple3(): void {
    if (this.stack.length < 3) {
      fail('INVALID_PICKLE', 'The pickle metadata stack underflowed.')
    }
    const third = this.pop()
    const second = this.pop()
    const first = this.pop()
    this.push({ kind: 'tuple', items: [first, second, third] } satisfies TupleValue)
  }

  private buildStorageReference(): void {
    const persistentId = this.pop()
    if (
      !isTuple(persistentId) ||
      persistentId.items.length !== 5 ||
      persistentId.items[0] !== 'storage' ||
      !isGlobal(persistentId.items[1], 'float-storage') ||
      typeof persistentId.items[2] !== 'string' ||
      !/^(0|[1-9][0-9]*)$/.test(persistentId.items[2]) ||
      persistentId.items[3] !== 'cpu' ||
      !safeInteger(persistentId.items[4]) ||
      persistentId.items[4] <= 0 ||
      persistentId.items[4] > MAX_STORAGE_ELEMENTS
    ) {
      fail('INVALID_TENSOR_SCHEMA', 'A tensor storage reference is invalid or unsupported.')
    }
    const key = persistentId.items[2]
    if (this.storageKeys.has(key)) {
      fail('INVALID_TENSOR_SCHEMA', 'Shared or duplicate tensor storages are not supported.')
    }
    const elementCount = persistentId.items[4]
    const reference: StorageReference = Object.freeze({
      key,
      dtype: 'float32',
      location: 'cpu',
      elementCount,
      byteLength: elementCount * Float32Array.BYTES_PER_ELEMENT,
    })
    this.storageKeys.set(key, reference)
    this.push({ kind: 'storage', reference } satisfies StorageValue)
  }

  private reduce(): void {
    const args = this.pop()
    const callable = this.pop()
    if (!isTuple(args)) {
      fail('UNSAFE_PICKLE', 'A pickle REDUCE call has invalid arguments.')
    }
    if (isGlobal(callable, 'ordered-dict')) {
      if (args.items.length !== 0) {
        fail('UNSAFE_PICKLE', 'Only an empty tensor hooks dictionary is supported.')
      }
      this.push(EMPTY_HOOKS)
      return
    }
    if (!isGlobal(callable, 'rebuild-tensor-v2') || args.items.length !== 6) {
      fail('UNSAFE_PICKLE', 'The pickle attempts an unsupported REDUCE call.')
    }

    const [storage, storageOffset, rawShape, rawStride, requiresGrad, hooks] = args.items
    if (
      !isStorage(storage) ||
      storageOffset !== 0 ||
      requiresGrad !== false ||
      hooks !== EMPTY_HOOKS
    ) {
      fail('INVALID_TENSOR_SCHEMA', 'A tensor rebuild descriptor is invalid or unsupported.')
    }
    const shape = positiveIntegerTuple(rawShape, 'tensor shape')
    const stride = positiveIntegerTuple(rawStride, 'tensor stride')
    if (shape.length !== stride.length) {
      fail('INVALID_TENSOR_SCHEMA', 'Tensor shape and stride ranks do not match.')
    }

    let contiguousStride = 1n
    for (let index = shape.length - 1; index >= 0; index -= 1) {
      if (BigInt(stride[index]) !== contiguousStride) {
        fail('INVALID_TENSOR_SCHEMA', 'Only contiguous OmniX tensors are supported.')
      }
      contiguousStride *= BigInt(shape[index])
      if (contiguousStride > BigInt(MAX_STORAGE_ELEMENTS)) {
        fail('RESOURCE_LIMIT', 'A tensor descriptor exceeds the browser storage limit.')
      }
    }
    if (contiguousStride !== BigInt(storage.reference.elementCount)) {
      fail('INVALID_TENSOR_SCHEMA', 'A tensor shape does not match its storage size.')
    }

    const descriptor: TensorDescriptor = Object.freeze({
      storage: storage.reference,
      storageOffset: 0,
      shape: Object.freeze([...shape]),
      stride: Object.freeze([...stride]),
      requiresGrad: false,
    })
    this.push({ kind: 'tensor', descriptor } satisfies TensorValue)
  }

  private setItems(): void {
    const markIndex = this.lastMark()
    if (markIndex === 0 || (this.stack.length - markIndex - 1) % 2 !== 0) {
      fail('INVALID_PICKLE', 'A pickle dictionary update is malformed.')
    }
    const dictionary = this.stack[markIndex - 1]
    if (!isDict(dictionary)) {
      fail('UNSAFE_PICKLE', 'Only the root OmniX tensor dictionary is supported.')
    }
    for (let index = markIndex + 1; index < this.stack.length; index += 2) {
      const key = this.stack[index]
      const value = this.stack[index + 1]
      if (typeof key !== 'string' || dictionary.entries.has(key)) {
        fail('INVALID_TENSOR_SCHEMA', 'The root tensor dictionary has invalid keys.')
      }
      dictionary.entries.set(key, value)
    }
    this.stack.length = markIndex
  }

  private finish(): Map<string, TensorDescriptor> {
    if (this.cursor !== this.bytes.byteLength || this.stack.length !== 1) {
      fail('INVALID_PICKLE', 'The pickle has trailing data or an invalid final stack.')
    }
    const root = this.stack[0]
    if (!isDict(root) || root.entries.size !== REQUIRED_KEYS.length) {
      fail('INVALID_TENSOR_SCHEMA', 'The pickle root must contain exactly four OmniX tensors.')
    }

    const tensors = new Map<RequiredKey, TensorDescriptor>()
    for (const key of REQUIRED_KEYS) {
      const value = root.entries.get(key)
      if (!isTensor(value)) {
        fail('INVALID_TENSOR_SCHEMA', 'The pickle root does not match the OmniX tensor schema.')
      }
      tensors.set(key, value.descriptor)
    }
    if ([...root.entries.keys()].some((key) => !REQUIRED_KEYS.includes(key as RequiredKey))) {
      fail('INVALID_TENSOR_SCHEMA', 'The pickle root contains unsupported keys.')
    }

    const trajectory = tensors.get('trajectory')!
    const cameraPose = tensors.get('camera_pose')!
    const intrinsics = tensors.get('intrinsics')!
    const dynamicScore = tensors.get('pts3d_dynamic_score')!
    if (trajectory.shape.length !== 5 || trajectory.shape[4] !== 3) {
      fail('INVALID_TENSOR_SCHEMA', 'The trajectory tensor shape is invalid.')
    }
    const [sourceViews, frames, height, width] = trajectory.shape
    if (
      !exactShape(cameraPose.shape, [sourceViews, 3, 4]) ||
      !exactShape(intrinsics.shape, [sourceViews, 3, 3]) ||
      !exactShape(dynamicScore.shape, [sourceViews, height, width])
    ) {
      fail('INVALID_TENSOR_SCHEMA', 'The OmniX tensor shapes are incompatible.')
    }
    const sourcePixels = BigInt(sourceViews) * BigInt(height) * BigInt(width)
    if (
      sourceViews > MAX_SOURCE_VIEWS ||
      frames > MAX_FRAMES ||
      sourcePixels > BigInt(MAX_SOURCE_PIXELS)
    ) {
      fail('RESOURCE_LIMIT', 'The OmniX tensor dimensions exceed browser ingestion limits.')
    }

    const storageKeys = new Set<string>()
    let totalBytes = 0
    for (const descriptor of tensors.values()) {
      if (storageKeys.has(descriptor.storage.key)) {
        fail('INVALID_TENSOR_SCHEMA', 'Shared tensor storages are not supported.')
      }
      storageKeys.add(descriptor.storage.key)
      totalBytes += descriptor.storage.byteLength
    }
    if (storageKeys.size !== 4 || totalBytes > MAX_TOTAL_TENSOR_BYTES) {
      fail('RESOURCE_LIMIT', 'The OmniX tensor storage exceeds browser ingestion limits.')
    }
    return new Map(tensors)
  }
}

/** Parse only the inert tensor metadata emitted by this repository's torch.save call. */
export function parseTorchPickle(bytes: Uint8Array): Map<string, TensorDescriptor> {
  if (!(bytes instanceof Uint8Array) || bytes.byteLength === 0) {
    fail('INVALID_PICKLE', 'The torch.save pickle metadata is empty.')
  }
  if (bytes.byteLength > MAX_PICKLE_BYTES) {
    fail('RESOURCE_LIMIT', 'The torch.save pickle metadata exceeds the 1 MiB limit.')
  }
  return new RestrictedPickleReader(bytes).run()
}
