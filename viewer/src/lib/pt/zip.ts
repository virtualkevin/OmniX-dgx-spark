import { PtParseError, type PtParseErrorCode } from './types'
import { MAX_BROWSER_DATASET_FILE_BYTES } from '../limits'

export { PtParseError } from './types'

const EOCD_SIGNATURE = 0x06054b50
const ZIP64_EOCD_SIGNATURE = 0x06064b50
const ZIP64_LOCATOR_SIGNATURE = 0x07064b50
const CENTRAL_HEADER_SIGNATURE = 0x02014b50
const LOCAL_HEADER_SIGNATURE = 0x04034b50

const EOCD_BYTES = 22
const MAX_EOCD_SEARCH_BYTES = EOCD_BYTES + 0xffff
const MAX_CENTRAL_DIRECTORY_BYTES = 16 * 1024 * 1024
const MAX_ENTRY_COUNT = 512
const DEFAULT_READ_LIMIT = 64 * 1024 * 1024
const TORCH_FLAGS = 0x0808

export interface ZipEntry {
  readonly name: string
  readonly flags: number
  readonly compressionMethod: 0
  readonly crc32: number
  readonly compressedSize: number
  readonly uncompressedSize: number
  readonly localHeaderOffset: number
  readonly dataOffset: number
}

export interface TorchZipArchive {
  readonly file: Blob
  /** The PyTorch archive directory without a trailing slash. */
  readonly prefix: string
  /** Entries keyed by their complete archive path. */
  readonly entries: ReadonlyMap<string, ZipEntry>
  /** Reads stored bytes after range checks; callers may verify them against ZipEntry.crc32. */
  readEntry(relativeOrFullName: string, maxBytes?: number): Promise<Uint8Array>
  sliceEntry(
    relativeOrFullName: string,
    start: number,
    length: number,
  ): Promise<Uint8Array>
}

interface EndOfCentralDirectory {
  readonly offset: number
  readonly entryCount: number
  readonly centralDirectoryOffset: number
  readonly centralDirectorySize: number
}

interface CentralEntry extends Omit<ZipEntry, 'dataOffset' | 'compressionMethod'> {
  readonly compressionMethod: number
}

function fail(code: PtParseErrorCode, message: string, cause?: unknown): never {
  throw new PtParseError(code, message, cause === undefined ? undefined : { cause })
}

function view(bytes: Uint8Array): DataView {
  return new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
}

function checkedNumber(value: bigint, code: PtParseErrorCode, message: string): number {
  if (value < 0n || value > BigInt(Number.MAX_SAFE_INTEGER)) {
    fail(code, message)
  }
  return Number(value)
}

function checkedAdd(
  left: number,
  right: number,
  code: PtParseErrorCode,
  message: string,
): number {
  if (!Number.isSafeInteger(left) || !Number.isSafeInteger(right) || left < 0 || right < 0) {
    fail(code, message)
  }
  const result = left + right
  if (!Number.isSafeInteger(result)) {
    fail(code, message)
  }
  return result
}

async function readRange(
  file: Blob,
  offset: number,
  length: number,
  code: PtParseErrorCode = 'INVALID_ZIP',
): Promise<Uint8Array> {
  const end = checkedAdd(offset, length, code, 'The .pt archive contains an invalid byte range.')
  if (end > file.size) {
    fail(code, 'The .pt archive is truncated or contains an invalid byte range.')
  }
  try {
    const buffer = await file.slice(offset, end).arrayBuffer()
    if (buffer.byteLength !== length) {
      fail(code, 'The .pt archive is truncated or unreadable.')
    }
    return new Uint8Array(buffer)
  } catch (error) {
    if (error instanceof PtParseError) {
      throw error
    }
    fail(code, 'The .pt archive could not be read.', error)
  }
}

function findEocd(tail: Uint8Array, tailOffset: number, fileSize: number): number {
  const data = view(tail)
  for (let index = tail.byteLength - EOCD_BYTES; index >= 0; index -= 1) {
    if (data.getUint32(index, true) !== EOCD_SIGNATURE) {
      continue
    }
    const commentLength = data.getUint16(index + 20, true)
    if (tailOffset + index + EOCD_BYTES + commentLength === fileSize) {
      return tailOffset + index
    }
  }
  fail('INVALID_ZIP', 'The file is not a supported ZIP-based torch.save archive.')
}

async function parseZip64Eocd(
  file: Blob,
  standardEocdOffset: number,
): Promise<Omit<EndOfCentralDirectory, 'offset'>> {
  if (standardEocdOffset < 20) {
    fail('INVALID_ZIP', 'The ZIP64 end records are missing or malformed.')
  }
  const locatorOffset = standardEocdOffset - 20
  const locator = await readRange(file, locatorOffset, 20)
  const locatorView = view(locator)
  if (
    locatorView.getUint32(0, true) !== ZIP64_LOCATOR_SIGNATURE ||
    locatorView.getUint32(4, true) !== 0 ||
    locatorView.getUint32(16, true) !== 1
  ) {
    fail('UNSUPPORTED_ZIP', 'Multi-part or malformed ZIP64 archives are not supported.')
  }

  const zip64Offset = checkedNumber(
    locatorView.getBigUint64(8, true),
    'RESOURCE_LIMIT',
    'The ZIP64 archive is too large for this browser.',
  )
  const record = await readRange(file, zip64Offset, 56)
  const recordView = view(record)
  if (
    recordView.getUint32(0, true) !== ZIP64_EOCD_SIGNATURE ||
    recordView.getBigUint64(4, true) < 44n
  ) {
    fail('INVALID_ZIP', 'The ZIP64 end record is malformed.')
  }
  const recordEnd = BigInt(zip64Offset) + 12n + recordView.getBigUint64(4, true)
  if (recordEnd > BigInt(locatorOffset)) {
    fail('INVALID_ZIP', 'The ZIP64 end record has an invalid length.')
  }
  if (recordView.getUint32(16, true) !== 0 || recordView.getUint32(20, true) !== 0) {
    fail('UNSUPPORTED_ZIP', 'Multi-part ZIP archives are not supported.')
  }

  const entriesOnDisk = recordView.getBigUint64(24, true)
  const totalEntries = recordView.getBigUint64(32, true)
  if (entriesOnDisk !== totalEntries) {
    fail('UNSUPPORTED_ZIP', 'Multi-part ZIP archives are not supported.')
  }
  return {
    entryCount: checkedNumber(
      totalEntries,
      'RESOURCE_LIMIT',
      'The .pt archive contains too many entries.',
    ),
    centralDirectorySize: checkedNumber(
      recordView.getBigUint64(40, true),
      'RESOURCE_LIMIT',
      'The ZIP central directory is too large for this browser.',
    ),
    centralDirectoryOffset: checkedNumber(
      recordView.getBigUint64(48, true),
      'RESOURCE_LIMIT',
      'The ZIP archive is too large for this browser.',
    ),
  }
}

async function readEocd(file: Blob): Promise<EndOfCentralDirectory> {
  const tailLength = Math.min(file.size, MAX_EOCD_SEARCH_BYTES)
  const tailOffset = file.size - tailLength
  const tail = await readRange(file, tailOffset, tailLength)
  const eocdOffset = findEocd(tail, tailOffset, file.size)
  const relativeOffset = eocdOffset - tailOffset
  const data = view(tail)

  const diskNumber = data.getUint16(relativeOffset + 4, true)
  const centralDisk = data.getUint16(relativeOffset + 6, true)
  const entriesOnDisk = data.getUint16(relativeOffset + 8, true)
  const totalEntries = data.getUint16(relativeOffset + 10, true)
  const centralDirectorySize = data.getUint32(relativeOffset + 12, true)
  const centralDirectoryOffset = data.getUint32(relativeOffset + 16, true)

  const needsZip64 =
    diskNumber === 0xffff ||
    centralDisk === 0xffff ||
    entriesOnDisk === 0xffff ||
    totalEntries === 0xffff ||
    centralDirectorySize === 0xffffffff ||
    centralDirectoryOffset === 0xffffffff

  let result: Omit<EndOfCentralDirectory, 'offset'>
  if (needsZip64) {
    result = await parseZip64Eocd(file, eocdOffset)
  } else {
    if (diskNumber !== 0 || centralDisk !== 0 || entriesOnDisk !== totalEntries) {
      fail('UNSUPPORTED_ZIP', 'Multi-part ZIP archives are not supported.')
    }
    result = { entryCount: totalEntries, centralDirectorySize, centralDirectoryOffset }
  }

  if (result.entryCount <= 0 || result.entryCount > MAX_ENTRY_COUNT) {
    fail('RESOURCE_LIMIT', 'The .pt archive contains too many entries.')
  }
  if (
    result.centralDirectorySize <= 0 ||
    result.centralDirectorySize > MAX_CENTRAL_DIRECTORY_BYTES
  ) {
    fail('RESOURCE_LIMIT', 'The ZIP central directory is too large for this browser.')
  }
  const centralEnd = checkedAdd(
    result.centralDirectoryOffset,
    result.centralDirectorySize,
    'INVALID_ZIP',
    'The ZIP central directory has an invalid range.',
  )
  if (centralEnd > eocdOffset) {
    fail('INVALID_ZIP', 'The ZIP central directory has an invalid range.')
  }
  return { offset: eocdOffset, ...result }
}

function decodeEntryName(bytes: Uint8Array): string {
  let name: string
  try {
    name = new TextDecoder('utf-8', { fatal: true }).decode(bytes)
  } catch (error) {
    fail('UNSAFE_ZIP_ENTRY', 'The .pt archive contains an invalid entry name.', error)
  }
  const parts = name.split('/')
  if (
    name.length === 0 ||
    name.startsWith('/') ||
    name.includes('\\') ||
    name.includes('\0') ||
    parts.some((part) => part.length === 0 || part === '.' || part === '..')
  ) {
    fail('UNSAFE_ZIP_ENTRY', 'The .pt archive contains an unsafe entry name.')
  }
  return name
}

function readZip64CentralValues(
  extra: Uint8Array,
  rawUncompressedSize: number,
  rawCompressedSize: number,
  rawLocalOffset: number,
  rawDisk: number,
): {
  uncompressedSize: number
  compressedSize: number
  localHeaderOffset: number
  disk: number
} {
  let cursor = 0
  let zip64: Uint8Array | undefined
  const extraView = view(extra)
  while (cursor < extra.byteLength) {
    if (cursor + 4 > extra.byteLength) {
      fail('INVALID_ZIP', 'A ZIP central-directory extra field is malformed.')
    }
    const id = extraView.getUint16(cursor, true)
    const length = extraView.getUint16(cursor + 2, true)
    const end = cursor + 4 + length
    if (end > extra.byteLength) {
      fail('INVALID_ZIP', 'A ZIP central-directory extra field is malformed.')
    }
    if (id === 0x0001) {
      if (zip64 !== undefined) {
        fail('INVALID_ZIP', 'The ZIP64 metadata is ambiguous.')
      }
      zip64 = extra.subarray(cursor + 4, end)
    }
    cursor = end
  }

  const needsZip64 =
    rawUncompressedSize === 0xffffffff ||
    rawCompressedSize === 0xffffffff ||
    rawLocalOffset === 0xffffffff ||
    rawDisk === 0xffff
  if (!needsZip64) {
    return {
      uncompressedSize: rawUncompressedSize,
      compressedSize: rawCompressedSize,
      localHeaderOffset: rawLocalOffset,
      disk: rawDisk,
    }
  }
  if (zip64 === undefined) {
    fail('INVALID_ZIP', 'A ZIP64 central-directory entry is missing metadata.')
  }

  const zip64View = view(zip64)
  let zip64Cursor = 0
  const nextUint64 = (): number => {
    if (zip64Cursor + 8 > zip64.byteLength) {
      fail('INVALID_ZIP', 'A ZIP64 central-directory entry is truncated.')
    }
    const value = checkedNumber(
      zip64View.getBigUint64(zip64Cursor, true),
      'RESOURCE_LIMIT',
      'A ZIP entry is too large for this browser.',
    )
    zip64Cursor += 8
    return value
  }

  const uncompressedSize =
    rawUncompressedSize === 0xffffffff ? nextUint64() : rawUncompressedSize
  const compressedSize = rawCompressedSize === 0xffffffff ? nextUint64() : rawCompressedSize
  const localHeaderOffset = rawLocalOffset === 0xffffffff ? nextUint64() : rawLocalOffset
  let disk = rawDisk
  if (rawDisk === 0xffff) {
    if (zip64Cursor + 4 > zip64.byteLength) {
      fail('INVALID_ZIP', 'A ZIP64 central-directory entry is truncated.')
    }
    disk = zip64View.getUint32(zip64Cursor, true)
  }
  return { uncompressedSize, compressedSize, localHeaderOffset, disk }
}

function parseCentralDirectory(
  bytes: Uint8Array,
  entryCount: number,
): CentralEntry[] {
  const entries: CentralEntry[] = []
  const names = new Set<string>()
  const data = view(bytes)
  let cursor = 0

  for (let index = 0; index < entryCount; index += 1) {
    if (cursor + 46 > bytes.byteLength || data.getUint32(cursor, true) !== CENTRAL_HEADER_SIGNATURE) {
      fail('INVALID_ZIP', 'The ZIP central directory is truncated or malformed.')
    }
    const flags = data.getUint16(cursor + 8, true)
    const compressionMethod = data.getUint16(cursor + 10, true)
    const crc32 = data.getUint32(cursor + 16, true)
    const rawCompressedSize = data.getUint32(cursor + 20, true)
    const rawUncompressedSize = data.getUint32(cursor + 24, true)
    const nameLength = data.getUint16(cursor + 28, true)
    const extraLength = data.getUint16(cursor + 30, true)
    const commentLength = data.getUint16(cursor + 32, true)
    const rawDisk = data.getUint16(cursor + 34, true)
    const rawLocalOffset = data.getUint32(cursor + 42, true)
    const recordLength = 46 + nameLength + extraLength + commentLength
    if (cursor + recordLength > bytes.byteLength) {
      fail('INVALID_ZIP', 'The ZIP central directory is truncated or malformed.')
    }
    if (flags !== TORCH_FLAGS || compressionMethod !== 0 || commentLength !== 0) {
      fail(
        'UNSUPPORTED_ZIP',
        'Only uncompressed, unencrypted PyTorch ZIP entries are supported.',
      )
    }
    const name = decodeEntryName(bytes.subarray(cursor + 46, cursor + 46 + nameLength))
    if (names.has(name)) {
      fail('UNSAFE_ZIP_ENTRY', 'The .pt archive contains duplicate entry names.')
    }
    names.add(name)

    const extraStart = cursor + 46 + nameLength
    const values = readZip64CentralValues(
      bytes.subarray(extraStart, extraStart + extraLength),
      rawUncompressedSize,
      rawCompressedSize,
      rawLocalOffset,
      rawDisk,
    )
    if (values.disk !== 0) {
      fail('UNSUPPORTED_ZIP', 'Multi-part ZIP archives are not supported.')
    }
    if (values.compressedSize !== values.uncompressedSize) {
      fail('UNSUPPORTED_ZIP', 'Compressed PyTorch ZIP entries are not supported.')
    }
    entries.push({
      name,
      flags,
      compressionMethod,
      crc32,
      compressedSize: values.compressedSize,
      uncompressedSize: values.uncompressedSize,
      localHeaderOffset: values.localHeaderOffset,
    })
    cursor += recordLength
  }
  if (cursor !== bytes.byteLength) {
    fail('INVALID_ZIP', 'The ZIP central directory contains trailing records.')
  }
  return entries
}

async function validateLocalEntries(
  file: Blob,
  centralEntries: CentralEntry[],
  centralDirectoryOffset: number,
): Promise<Map<string, ZipEntry>> {
  const entries = new Map<string, ZipEntry>()
  const physicalRanges: Array<{ start: number; end: number }> = []

  for (const entry of centralEntries) {
    if (entry.localHeaderOffset >= centralDirectoryOffset) {
      fail('INVALID_ZIP', 'A ZIP entry points outside the payload area.')
    }
    const header = await readRange(file, entry.localHeaderOffset, 30)
    const headerView = view(header)
    if (
      headerView.getUint32(0, true) !== LOCAL_HEADER_SIGNATURE ||
      headerView.getUint16(6, true) !== entry.flags ||
      headerView.getUint16(8, true) !== entry.compressionMethod
    ) {
      fail('INVALID_ZIP', 'A ZIP local header does not match the central directory.')
    }
    // PyTorch writes sizes and CRC after the stored body because bit 3 is set.
    if (
      headerView.getUint32(14, true) !== 0 ||
      headerView.getUint32(18, true) !== 0 ||
      headerView.getUint32(22, true) !== 0
    ) {
      fail('UNSUPPORTED_ZIP', 'The ZIP local-header layout is not a supported torch.save layout.')
    }
    const nameLength = headerView.getUint16(26, true)
    const extraLength = headerView.getUint16(28, true)
    const localName = decodeEntryName(
      await readRange(file, entry.localHeaderOffset + 30, nameLength),
    )
    if (localName !== entry.name) {
      fail('INVALID_ZIP', 'A ZIP local entry name does not match the central directory.')
    }
    const dataOffset = checkedAdd(
      entry.localHeaderOffset,
      30 + nameLength + extraLength,
      'INVALID_ZIP',
      'A ZIP local header has an invalid length.',
    )
    const dataEnd = checkedAdd(
      dataOffset,
      entry.compressedSize,
      'INVALID_ZIP',
      'A ZIP entry has an invalid payload range.',
    )
    if (dataEnd > centralDirectoryOffset) {
      fail('INVALID_ZIP', 'A ZIP entry overlaps the central directory.')
    }
    physicalRanges.push({ start: entry.localHeaderOffset, end: dataEnd })
    entries.set(entry.name, {
      ...entry,
      compressionMethod: 0,
      dataOffset,
    })
  }

  physicalRanges.sort((left, right) => left.start - right.start)
  for (let index = 1; index < physicalRanges.length; index += 1) {
    if (physicalRanges[index].start < physicalRanges[index - 1].end) {
      fail('INVALID_ZIP', 'ZIP entry payload ranges overlap.')
    }
  }
  return entries
}

function derivePrefix(entries: ReadonlyMap<string, ZipEntry>): string {
  const pickleEntries = [...entries.keys()].filter(
    (name) => name === 'data.pkl' || name.endsWith('/data.pkl'),
  )
  if (pickleEntries.length !== 1) {
    fail('MISSING_TORCH_ENTRY', 'The archive must contain exactly one PyTorch data.pkl entry.')
  }
  const pickleName = pickleEntries[0]
  const prefix = pickleName === 'data.pkl' ? '' : pickleName.slice(0, -'/data.pkl'.length)
  const prefixWithSlash = prefix.length === 0 ? '' : `${prefix}/`
  for (const name of entries.keys()) {
    if (prefix.length > 0 ? !name.startsWith(prefixWithSlash) : name.includes('/')) {
      fail('UNSAFE_ZIP_ENTRY', 'All torch.save entries must share one archive prefix.')
    }
  }
  for (const relativeName of ['data.pkl', 'byteorder', 'version']) {
    if (!entries.has(`${prefixWithSlash}${relativeName}`)) {
      fail('MISSING_TORCH_ENTRY', 'The archive is missing required torch.save metadata.')
    }
  }
  return prefix
}

function resolveEntry(
  entries: ReadonlyMap<string, ZipEntry>,
  prefix: string,
  relativeOrFullName: string,
): ZipEntry {
  const direct = entries.get(relativeOrFullName)
  if (direct !== undefined) {
    return direct
  }
  const relative = prefix.length === 0 ? relativeOrFullName : `${prefix}/${relativeOrFullName}`
  const entry = entries.get(relative)
  if (entry === undefined) {
    fail('MISSING_TORCH_ENTRY', 'The requested torch.save entry is missing.')
  }
  return entry
}

function decodeMarker(bytes: Uint8Array): string {
  try {
    return new TextDecoder('ascii', { fatal: true }).decode(bytes).trim()
  } catch (error) {
    fail('INVALID_ZIP', 'A torch.save format marker has invalid encoding.', error)
  }
}

export async function openTorchZip(file: Blob): Promise<TorchZipArchive> {
  if (!Number.isSafeInteger(file.size) || file.size < EOCD_BYTES) {
    fail('INVALID_ZIP', 'The file is empty or too small to be a torch.save archive.')
  }
  if (file.size > MAX_BROWSER_DATASET_FILE_BYTES) {
    fail('RESOURCE_LIMIT', 'The selected .pt file exceeds the 2 GiB browser limit.')
  }

  const eocd = await readEocd(file)
  const centralBytes = await readRange(
    file,
    eocd.centralDirectoryOffset,
    eocd.centralDirectorySize,
  )
  const centralEntries = parseCentralDirectory(centralBytes, eocd.entryCount)
  const entries = await validateLocalEntries(
    file,
    centralEntries,
    eocd.centralDirectoryOffset,
  )
  const prefix = derivePrefix(entries)

  const archive: TorchZipArchive = {
    file,
    prefix,
    entries,
    async readEntry(relativeOrFullName, maxBytes = DEFAULT_READ_LIMIT) {
      if (
        !Number.isSafeInteger(maxBytes) ||
        maxBytes <= 0 ||
        maxBytes > MAX_BROWSER_DATASET_FILE_BYTES
      ) {
        fail('INVALID_ENTRY_RANGE', 'The requested entry size limit is invalid.')
      }
      const entry = resolveEntry(entries, prefix, relativeOrFullName)
      if (entry.uncompressedSize > maxBytes) {
        fail('ENTRY_TOO_LARGE', 'The requested torch.save entry exceeds its read limit.')
      }
      return readRange(file, entry.dataOffset, entry.uncompressedSize)
    },
    async sliceEntry(relativeOrFullName, start, length) {
      const entry = resolveEntry(entries, prefix, relativeOrFullName)
      if (
        !Number.isSafeInteger(start) ||
        !Number.isSafeInteger(length) ||
        start < 0 ||
        length < 0 ||
        start + length > entry.uncompressedSize
      ) {
        fail('INVALID_ENTRY_RANGE', 'The requested torch.save entry slice is out of bounds.')
      }
      return readRange(file, entry.dataOffset + start, length, 'INVALID_ENTRY_RANGE')
    },
  }

  const byteorder = decodeMarker(await archive.readEntry('byteorder', 16))
  const version = decodeMarker(await archive.readEntry('version', 16))
  if (byteorder !== 'little' || version !== '3') {
    fail('UNSUPPORTED_ZIP', 'Only little-endian torch.save format version 3 is supported.')
  }
  const formatVersionName = prefix.length === 0 ? '.format_version' : `${prefix}/.format_version`
  if (entries.has(formatVersionName)) {
    const formatVersion = decodeMarker(await archive.readEntry('.format_version', 16))
    if (formatVersion !== '1') {
      fail('UNSUPPORTED_ZIP', 'This torch.save format version is not supported.')
    }
  }
  const alignmentName =
    prefix.length === 0 ? '.storage_alignment' : `${prefix}/.storage_alignment`
  if (entries.has(alignmentName)) {
    const alignment = decodeMarker(await archive.readEntry('.storage_alignment', 16))
    if (alignment !== '64') {
      fail('UNSUPPORTED_ZIP', 'This torch.save storage alignment is not supported.')
    }
  }
  return archive
}
