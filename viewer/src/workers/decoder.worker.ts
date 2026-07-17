/// <reference lib="webworker" />

import { parseOmx4d } from '../lib/omx4d'
import type { ViewerDataset } from '../lib/dataset'

interface DecodeRequest {
  id: number
  buffer: ArrayBuffer
}

interface DecodeSuccess {
  id: number
  ok: true
  dataset: ViewerDataset
}

interface DecodeFailure {
  id: number
  ok: false
  error: string
}

self.onmessage = (event: MessageEvent<DecodeRequest>) => {
  const { id, buffer } = event.data
  try {
    const decoded = parseOmx4d(buffer)
    const dataset: ViewerDataset = {
      manifest: decoded.manifest,
      positions: decoded.positions,
      colors: decoded.colors,
      dynamicScore: decoded.dynamicScore,
      sourceView: decoded.sourceView,
      cameraPose: decoded.cameraPose,
      intrinsics: decoded.intrinsics,
    }
    const transferables = Array.from(new Set([
      dataset.positions.buffer,
      dataset.colors.buffer,
      dataset.dynamicScore.buffer,
      dataset.sourceView.buffer,
      dataset.cameraPose.buffer,
      dataset.intrinsics.buffer,
    ])) as ArrayBuffer[]
    const response: DecodeSuccess = { id, ok: true, dataset }
    self.postMessage(response, { transfer: transferables })
  } catch (error) {
    const response: DecodeFailure = {
      id,
      ok: false,
      error: error instanceof Error ? error.message : 'The renderer payload could not be decoded.',
    }
    self.postMessage(response)
  }
}

export {}
