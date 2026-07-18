/// <reference lib="webworker" />

import { parseOmx4d } from '../lib/omx4d'
import { MAX_BROWSER_DATASET_FILE_BYTES } from '../lib/limits'
import { convertOmnixPt } from '../lib/pt/convert'
import type { ViewerDataset } from '../lib/dataset'
import type { PtDecodeOptions, PtDecodeProgress } from '../lib/decoder-client'

interface DecodeOmx4dRequest {
  id: number
  kind: 'decodeOmx4d'
  buffer: ArrayBuffer
}

interface DecodeOmx4dFileRequest {
  id: number
  kind: 'decodeOmx4dFile'
  file: File
}

interface DecodePtRequest {
  id: number
  kind: 'decodePt'
  file: File
  options: PtDecodeOptions
}

interface CancelRequest {
  id: number
  kind: 'cancel'
}

type WorkerRequest = DecodeOmx4dRequest | DecodeOmx4dFileRequest | DecodePtRequest | CancelRequest

interface DecodeSuccess {
  id: number
  kind: 'success'
  dataset: ViewerDataset
}

interface DecodeFailure {
  id: number
  kind: 'failure'
  error: string
}

interface DecodeProgress {
  id: number
  kind: 'progress'
  progress: PtDecodeProgress
}

const cancelled = new Set<number>()
const active = new Set<number>()

function transferablesFor(dataset: ViewerDataset): ArrayBuffer[] {
  return Array.from(new Set([
    dataset.positions.buffer,
    dataset.colors.buffer,
    dataset.dynamicScore.buffer,
    dataset.sourceView.buffer,
    dataset.cameraPose.buffer,
    dataset.intrinsics.buffer,
  ])) as ArrayBuffer[]
}

function omx4dDataset(buffer: ArrayBuffer): ViewerDataset {
  const decoded = parseOmx4d(buffer)
  return {
    manifest: decoded.manifest,
    positions: decoded.positions,
    colors: decoded.colors,
    dynamicScore: decoded.dynamicScore,
    sourceView: decoded.sourceView,
    cameraPose: decoded.cameraPose,
    intrinsics: decoded.intrinsics,
  }
}

async function readFileBuffer(file: File, id: number): Promise<ArrayBuffer> {
  const bytes = new Uint8Array(file.size)
  const reader = file.stream().getReader()
  let offset = 0

  try {
    while (true) {
      if (cancelled.has(id)) {
        await reader.cancel()
        throw new DOMException('Decode cancelled.', 'AbortError')
      }
      const { done, value } = await reader.read()
      if (cancelled.has(id)) {
        await reader.cancel()
        throw new DOMException('Decode cancelled.', 'AbortError')
      }
      if (done) break
      if (offset + value.byteLength > bytes.byteLength) {
        throw new Error('The selected .omx4d file changed while it was being read.')
      }
      bytes.set(value, offset)
      offset += value.byteLength
    }
  } finally {
    reader.releaseLock()
  }

  if (offset !== bytes.byteLength) {
    throw new Error('The selected .omx4d file ended before its declared size.')
  }
  return bytes.buffer
}

async function handleRequest(
  request: DecodeOmx4dRequest | DecodeOmx4dFileRequest | DecodePtRequest,
): Promise<void> {
  const { id } = request
  try {
    if (cancelled.has(id)) return
    let dataset: ViewerDataset
    if (request.kind === 'decodeOmx4d') {
      if (request.buffer.byteLength > MAX_BROWSER_DATASET_FILE_BYTES) {
        throw new Error('The selected .omx4d payload exceeds the 2 GiB browser limit.')
      }
      dataset = omx4dDataset(request.buffer)
    } else if (request.kind === 'decodeOmx4dFile') {
      if (request.file.size > MAX_BROWSER_DATASET_FILE_BYTES) {
        throw new Error('The selected .omx4d file exceeds the 2 GiB browser limit.')
      }
      const buffer = await readFileBuffer(request.file, id)
      if (cancelled.has(id)) return
      dataset = omx4dDataset(buffer)
    } else {
      let lastProgressPhase: string | undefined
      dataset = await convertOmnixPt(request.file, request.options, {
        isCancelled: () => cancelled.has(id),
        onProgress: (progress) => {
          if (cancelled.has(id)) return
          const phaseChanged = progress.phase !== lastProgressPhase
          const intervalReached = progress.completed % 8 === 0
          const phaseComplete = progress.completed === progress.total
          if (!phaseChanged && !intervalReached && !phaseComplete) return
          lastProgressPhase = progress.phase
          const messages: Record<PtDecodeProgress['stage'], string> = {
            archive: 'Reading the PyTorch archive directory',
            metadata: 'Validating tensor metadata',
            sampling: 'Selecting stable point identities',
            trajectory: 'Reading trajectory frames locally',
            finalizing: 'Preparing the renderer payload',
          }
          const normalized: PtDecodeProgress = {
            stage: progress.phase,
            progress: progress.total > 0 ? progress.completed / progress.total : 0,
            message: messages[progress.phase],
          }
          const response: DecodeProgress = { id, kind: 'progress', progress: normalized }
          self.postMessage(response)
        },
      })
    }
    if (cancelled.has(id)) return
    const response: DecodeSuccess = { id, kind: 'success', dataset }
    self.postMessage(response, { transfer: transferablesFor(dataset) })
  } catch (error) {
    if (cancelled.has(id)) return
    const response: DecodeFailure = {
      id,
      kind: 'failure',
      error: error instanceof Error ? error.message : 'The selected file could not be decoded.',
    }
    self.postMessage(response)
  } finally {
    cancelled.delete(id)
    active.delete(id)
  }
}

let decodeQueue = Promise.resolve()

self.onmessage = (event: MessageEvent<WorkerRequest>) => {
  const request = event.data
  if (request.kind === 'cancel') {
    if (active.has(request.id)) cancelled.add(request.id)
    return
  }
  active.add(request.id)
  decodeQueue = decodeQueue.then(() => handleRequest(request))
}

export {}
