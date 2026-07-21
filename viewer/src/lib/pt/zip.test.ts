import { describe, expect, it } from 'vitest'
import { openTorchZip } from './zip'
import { PtParseError, type PtParseErrorCode } from './types'

const encoder = new TextEncoder()

interface FixtureEntry {
  name: string
  data: Uint8Array
  method?: number
  flags?: number
}

interface WrittenEntry extends FixtureEntry {
  offset: number
}

function header(length: number): { bytes: Uint8Array; data: DataView } {
  const bytes = new Uint8Array(length)
  return { bytes, data: new DataView(bytes.buffer) }
}

function blobPart(bytes: Uint8Array): ArrayBuffer {
  return new Uint8Array(bytes).buffer
}

function zipFixture(
  entries: FixtureEntry[] = [
    { name: 'fixture/data.pkl', data: new Uint8Array([0x80, 0x02, 0x2e]) },
    { name: 'fixture/byteorder', data: new TextEncoder().encode('little') },
    { name: 'fixture/version', data: new TextEncoder().encode('3\n') },
    { name: 'fixture/data/0', data: new Uint8Array([10, 20, 30, 40, 50]) },
  ],
  zip64Sentinels = false,
): Blob {
  const parts: BlobPart[] = []
  const written: WrittenEntry[] = []
  let archiveOffset = 0

  for (const entry of entries) {
    const name = encoder.encode(entry.name)
    const local = header(30)
    local.data.setUint32(0, 0x04034b50, true)
    local.data.setUint16(6, entry.flags ?? 0x0808, true)
    local.data.setUint16(8, entry.method ?? 0, true)
    local.data.setUint16(26, name.byteLength, true)
    const descriptor = header(16)
    descriptor.data.setUint32(0, 0x08074b50, true)
    descriptor.data.setUint32(8, entry.data.byteLength, true)
    descriptor.data.setUint32(12, entry.data.byteLength, true)
    written.push({ ...entry, offset: archiveOffset })
    parts.push(blobPart(local.bytes), blobPart(name), blobPart(entry.data), blobPart(descriptor.bytes))
    archiveOffset += 30 + name.byteLength + entry.data.byteLength + 16
  }

  const centralOffset = archiveOffset
  for (const entry of written) {
    const name = encoder.encode(entry.name)
    const central = header(46)
    central.data.setUint32(0, 0x02014b50, true)
    central.data.setUint16(8, entry.flags ?? 0x0808, true)
    central.data.setUint16(10, entry.method ?? 0, true)
    central.data.setUint32(20, entry.data.byteLength, true)
    central.data.setUint32(24, entry.data.byteLength, true)
    central.data.setUint16(28, name.byteLength, true)
    central.data.setUint32(42, entry.offset, true)
    parts.push(blobPart(central.bytes), blobPart(name))
    archiveOffset += 46 + name.byteLength
  }
  const centralSize = archiveOffset - centralOffset

  const zip64Offset = archiveOffset
  const zip64 = header(56)
  zip64.data.setUint32(0, 0x06064b50, true)
  zip64.data.setBigUint64(4, 44n, true)
  zip64.data.setUint16(12, 0x031e, true)
  zip64.data.setUint16(14, 45, true)
  zip64.data.setBigUint64(24, BigInt(entries.length), true)
  zip64.data.setBigUint64(32, BigInt(entries.length), true)
  zip64.data.setBigUint64(40, BigInt(centralSize), true)
  zip64.data.setBigUint64(48, BigInt(centralOffset), true)
  const locator = header(20)
  locator.data.setUint32(0, 0x07064b50, true)
  locator.data.setBigUint64(8, BigInt(zip64Offset), true)
  locator.data.setUint32(16, 1, true)
  const eocd = header(22)
  eocd.data.setUint32(0, 0x06054b50, true)
  eocd.data.setUint16(8, zip64Sentinels ? 0xffff : entries.length, true)
  eocd.data.setUint16(10, zip64Sentinels ? 0xffff : entries.length, true)
  eocd.data.setUint32(12, zip64Sentinels ? 0xffffffff : centralSize, true)
  eocd.data.setUint32(16, zip64Sentinels ? 0xffffffff : centralOffset, true)
  parts.push(blobPart(zip64.bytes), blobPart(locator.bytes), blobPart(eocd.bytes))
  return new Blob(parts)
}

async function errorCode(blob: Blob): Promise<PtParseErrorCode> {
  try {
    await openTorchZip(blob)
  } catch (error) {
    expect(error).toBeInstanceOf(PtParseError)
    return (error as PtParseError).code
  }
  throw new Error('Expected openTorchZip to reject the fixture.')
}

describe('openTorchZip', () => {
  it.each([false, true])(
    'opens stored torch.save entries through standard and ZIP64 EOCD metadata (%s)',
    async (zip64Sentinels) => {
      const archive = await openTorchZip(zipFixture(undefined, zip64Sentinels))

      expect(archive.prefix).toBe('fixture')
      expect([...archive.entries.keys()]).toEqual([
        'fixture/data.pkl',
        'fixture/byteorder',
        'fixture/version',
        'fixture/data/0',
      ])
      expect(Array.from(await archive.readEntry('data.pkl'))).toEqual([0x80, 0x02, 0x2e])
      expect(Array.from(await archive.readEntry('fixture/data/0'))).toEqual([10, 20, 30, 40, 50])
      expect(Array.from(await archive.sliceEntry('data/0', 1, 3))).toEqual([20, 30, 40])
      expect(archive.entries.get('fixture/data/0')).toMatchObject({
        flags: 0x0808,
        compressionMethod: 0,
        compressedSize: 5,
        uncompressedSize: 5,
      })
    },
  )

  it('enforces read limits and entry slice bounds before allocating', async () => {
    const archive = await openTorchZip(zipFixture())
    await expect(archive.readEntry('data/0', 4)).rejects.toMatchObject({
      code: 'ENTRY_TOO_LARGE',
    })
    await expect(archive.sliceEntry('data/0', 4, 2)).rejects.toMatchObject({
      code: 'INVALID_ENTRY_RANGE',
    })
    await expect(archive.sliceEntry('data/0', -1, 1)).rejects.toMatchObject({
      code: 'INVALID_ENTRY_RANGE',
    })
  })

  it('rejects compressed, encrypted, duplicate, and unsafe entries', async () => {
    expect(
      await errorCode(
        zipFixture([
          { name: 'fixture/data.pkl', data: new Uint8Array([1]), method: 8 },
          { name: 'fixture/byteorder', data: encoder.encode('little') },
          { name: 'fixture/version', data: encoder.encode('3') },
        ]),
      ),
    ).toBe('UNSUPPORTED_ZIP')

    expect(
      await errorCode(
        zipFixture([
          { name: 'fixture/data.pkl', data: new Uint8Array([1]), flags: 0x0809 },
          { name: 'fixture/byteorder', data: encoder.encode('little') },
          { name: 'fixture/version', data: encoder.encode('3') },
        ]),
      ),
    ).toBe('UNSUPPORTED_ZIP')

    expect(
      await errorCode(
        zipFixture([
          { name: 'fixture/data.pkl', data: new Uint8Array([1]) },
          { name: 'fixture/data.pkl', data: new Uint8Array([2]) },
          { name: 'fixture/byteorder', data: encoder.encode('little') },
          { name: 'fixture/version', data: encoder.encode('3') },
        ]),
      ),
    ).toBe('UNSAFE_ZIP_ENTRY')

    expect(
      await errorCode(
        zipFixture([
          { name: 'fixture/../data.pkl', data: new Uint8Array([1]) },
          { name: 'fixture/byteorder', data: encoder.encode('little') },
          { name: 'fixture/version', data: encoder.encode('3') },
        ]),
      ),
    ).toBe('UNSAFE_ZIP_ENTRY')
  })

  it('rejects missing metadata, unsupported markers, and truncated archives', async () => {
    expect(
      await errorCode(
        zipFixture([
          { name: 'fixture/data.pkl', data: new Uint8Array([1]) },
          { name: 'fixture/version', data: encoder.encode('3') },
        ]),
      ),
    ).toBe('MISSING_TORCH_ENTRY')

    expect(
      await errorCode(
        zipFixture([
          { name: 'fixture/data.pkl', data: new Uint8Array([1]) },
          { name: 'fixture/byteorder', data: encoder.encode('big') },
          { name: 'fixture/version', data: encoder.encode('3') },
        ]),
      ),
    ).toBe('UNSUPPORTED_ZIP')

    const valid = zipFixture()
    expect(await errorCode(valid.slice(0, valid.size - 1))).toBe('INVALID_ZIP')
  })
})
