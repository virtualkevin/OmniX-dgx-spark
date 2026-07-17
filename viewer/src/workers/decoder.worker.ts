/// <reference lib="webworker" />

import { parseOmx4d } from '../lib/omx4d'
import { convertOmnixPt } from '../lib/pt/convert'
import type { ViewerDataset } from '../lib/dataset'
import type { PtDecodeOptions, PtDecodeProgress } from '../lib/decoder-client'

interface DecodeOmx4dRequest {
  id: number
  kind: 'decodeOmx4d'
  buffer: ArrayBuffer
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

type WorkerRequest = DecodeOmx4dRequest | DecodePtRequest | CancelRequest

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

async function handleRequest(request: DecodeOmx4dRequest | DecodePtRequest): Promise<void> {
  const { id } = request
  try {
    const dataset = request.kind === 'decodeOmx4d'
      ? omx4dDataset(request.buffer)
      : await convertOmnixPt(request.file, request.options, {
        isCancelled: () => cancelled.has(id),
        onProgress: (progress) => {
          if (cancelled.has(id)) return
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

self.onmessage = (event: MessageEvent<WorkerRequest>) => {
  const request = event.data
  if (request.kind === 'cancel') {
    if (active.has(request.id)) cancelled.add(request.id)
    return
  }
  active.add(request.id)
  void handleRequest(request)
}

export {}
