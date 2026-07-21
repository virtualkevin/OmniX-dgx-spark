import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { TensorDescriptor } from './types'
import type { TorchZipArchive, ZipEntry } from './zip'

const mocks = vi.hoisted(() => ({
  openTorchZip: vi.fn(),
  parseTorchPickle: vi.fn(),
}))

vi.mock('./zip', async (importOriginal) => ({
  ...(await importOriginal<typeof import('./zip')>()),
  openTorchZip: mocks.openTorchZip,
}))

vi.mock('./pickle', async (importOriginal) => ({
  ...(await importOriginal<typeof import('./pickle')>()),
  parseTorchPickle: mocks.parseTorchPickle,
}))

import { convertOmnixPt } from './convert'

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

function crc32(bytes: Uint8Array): number {
  let crc = 0xffffffff
  for (const byte of bytes) {
    crc ^= byte
    for (let bit = 0; bit < 8; bit += 1) {
      crc = (crc & 1) !== 0 ? 0xedb88320 ^ (crc >>> 1) : crc >>> 1
    }
  }
  return (crc ^ 0xffffffff) >>> 0
}

function floatBytes(values: number[]): Uint8Array<ArrayBuffer> {
  return new Uint8Array(new Float32Array(values).buffer)
}

function fakeArchive(data: ReadonlyMap<string, Uint8Array>): TorchZipArchive {
  const entries = new Map<string, ZipEntry>()
  let index = 1
  for (const [relativeName, bytes] of data) {
    entries.set(`fixture/${relativeName}`, {
      name: `fixture/${relativeName}`,
      flags: 0x0808,
      compressionMethod: 0,
      crc32: crc32(bytes),
      compressedSize: bytes.byteLength,
      uncompressedSize: bytes.byteLength,
      localHeaderOffset: index * 32,
      dataOffset: index * 64,
    })
    index += 1
  }

  function resolve(name: string): Uint8Array {
    const relative = name.startsWith('fixture/') ? name.slice('fixture/'.length) : name
    const bytes = data.get(relative)
    if (bytes === undefined) throw new Error('Missing fake entry')
    return bytes
  }

  return {
    file: new Blob(),
    prefix: 'fixture',
    entries,
    async readEntry(name, maxBytes = Number.MAX_SAFE_INTEGER) {
      const bytes = resolve(name)
      if (bytes.byteLength > maxBytes) throw new Error('Fake read limit exceeded')
      return bytes.slice()
    },
    async sliceEntry(name, start, length) {
      return resolve(name).slice(start, start + length)
    },
  }
}

describe('convertOmnixPt', () => {
  it('rejects point budgets above the realtime browser ceiling before reading', async () => {
    await expect(convertOmnixPt(
      new Blob(),
      { pointBudget: 500_001, fps: 15, name: 'too-large.pt' },
    )).rejects.toMatchObject({
      code: 'INVALID_OPTIONS',
      message: 'Point budget must not exceed the browser maximum of 500,000.',
    })
  })

  beforeEach(() => vi.clearAllMocks())

  it('streams tensor members into a renderer dataset without materializing the file', async () => {
    const tensors = new Map<string, TensorDescriptor>([
      ['trajectory', descriptor([1, 2, 1, 4, 3], '0')],
      ['camera_pose', descriptor([1, 3, 4], '1')],
      ['intrinsics', descriptor([1, 3, 3], '2')],
      ['pts3d_dynamic_score', descriptor([1, 1, 4], '3')],
    ])
    const data = new Map<string, Uint8Array>([
      ['data.pkl', new Uint8Array([0x80, 0x02, 0x2e])],
      ['data/0', floatBytes([
        0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5,
        10, 11, 12, 11, 12, 13, 12, 13, 14, 13, 14, 15,
      ])],
      ['data/1', floatBytes([
        1, 0, 0, 1,
        0, 1, 0, 2,
        0, 0, 1, 3,
      ])],
      ['data/2', floatBytes([
        500, 0, 2,
        0, 500, 1,
        0, 0, 1,
      ])],
      ['data/3', floatBytes([0, 0.25, 0.5, 1])],
    ])
    mocks.openTorchZip.mockResolvedValue(fakeArchive(data))
    mocks.parseTorchPickle.mockReturnValue(tensors)
    const progress = vi.fn()

    const dataset = await convertOmnixPt(
      new Blob([new Uint8Array([1, 2, 3])]),
      { pointBudget: 4, fps: 15, name: 'folder/demo.pt' },
      { onProgress: progress },
    )

    expect(dataset.manifest).toMatchObject({
      name: 'demo',
      frameCount: 2,
      pointCount: 4,
      fps: 15,
      bounds: { min: [0, -14, -15], max: [13, -1, -2] },
    })
    expect(Array.from(dataset.positions.slice(0, 6))).toEqual([0, -1, -2, 1, -2, -3])
    expect(Array.from(dataset.dynamicScore)).toEqual([0, 0.25, 0.5, 1])
    expect(Array.from(dataset.sourceView)).toEqual([0, 0, 0, 0])
    expect(Array.from(dataset.cameraPose)).toEqual([
      1, -0, -0, 1,
      -0, 1, 0, -2,
      -0, 0, 1, -3,
      0, -0, -0, 1,
    ])
    expect(progress).toHaveBeenLastCalledWith(expect.objectContaining({
      stage: 'finalizing',
      progress: 1,
    }))
  })
})
