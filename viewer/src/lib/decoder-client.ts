import type { ViewerDataset } from './dataset'

export type PtDecodeStage = 'archive' | 'metadata' | 'sampling' | 'trajectory' | 'finalizing'

export interface PtDecodeProgress {
  stage: PtDecodeStage
  progress: number
  message: string
}

export interface PtDecodeOptions {
  pointBudget: number
  fps: number
  name: string
}

interface WorkerProgress {
  id: number
  kind: 'progress'
  progress: PtDecodeProgress
}

interface WorkerSuccess {
  id: number
  kind: 'success'
  dataset: ViewerDataset
}

interface WorkerFailure {
  id: number
  kind: 'failure'
  error: string
}

type WorkerResponse = WorkerProgress | WorkerSuccess | WorkerFailure

interface PendingDecode {
  resolve: (dataset: ViewerDataset) => void
  reject: (error: Error) => void
  onProgress?: (progress: PtDecodeProgress) => void
  cleanup?: () => void
}

export interface PtDecodeCallbacks {
  signal?: AbortSignal
  onProgress?: (progress: PtDecodeProgress) => void
}

export interface DatasetDecodeCallbacks {
  signal?: AbortSignal
}

export class DecoderClient {
  private readonly worker = new Worker(new URL('../workers/decoder.worker.ts', import.meta.url), { type: 'module' })
  private readonly pending = new Map<number, PendingDecode>()
  private nextId = 1

  constructor() {
    this.worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
      const result = event.data
      const request = this.pending.get(result.id)
      if (!request) return
      if (result.kind === 'progress') {
        request.onProgress?.(result.progress)
        return
      }
      this.pending.delete(result.id)
      request.cleanup?.()
      if (result.kind === 'success') request.resolve(result.dataset)
      else request.reject(new Error(result.error))
    }
    this.worker.onerror = (event) => {
      const error = new Error(event.message || 'The decoder worker stopped unexpectedly.')
      for (const request of this.pending.values()) {
        request.cleanup?.()
        request.reject(error)
      }
      this.pending.clear()
    }
  }

  decode(
    buffer: ArrayBuffer,
    callbacks: DatasetDecodeCallbacks = {},
  ): Promise<ViewerDataset> {
    const id = this.nextId++
    return new Promise((resolve, reject) => {
      if (callbacks.signal?.aborted) {
        reject(new DOMException('Decode cancelled.', 'AbortError'))
        return
      }

      const onAbort = () => {
        const request = this.pending.get(id)
        if (!request) return
        this.pending.delete(id)
        request.cleanup?.()
        this.worker.postMessage({ id, kind: 'cancel' })
        reject(new DOMException('Decode cancelled.', 'AbortError'))
      }
      callbacks.signal?.addEventListener('abort', onAbort, { once: true })
      this.pending.set(id, {
        resolve,
        reject,
        cleanup: () => callbacks.signal?.removeEventListener('abort', onAbort),
      })
      this.worker.postMessage({ id, kind: 'decodeOmx4d', buffer }, [buffer])
    })
  }

  decodeOmx4dFile(
    file: File,
    callbacks: DatasetDecodeCallbacks = {},
  ): Promise<ViewerDataset> {
    const id = this.nextId++
    return new Promise((resolve, reject) => {
      if (callbacks.signal?.aborted) {
        reject(new DOMException('Decode cancelled.', 'AbortError'))
        return
      }

      const onAbort = () => {
        const request = this.pending.get(id)
        if (!request) return
        this.pending.delete(id)
        request.cleanup?.()
        this.worker.postMessage({ id, kind: 'cancel' })
        reject(new DOMException('Decode cancelled.', 'AbortError'))
      }
      callbacks.signal?.addEventListener('abort', onAbort, { once: true })
      this.pending.set(id, {
        resolve,
        reject,
        cleanup: () => callbacks.signal?.removeEventListener('abort', onAbort),
      })
      this.worker.postMessage({ id, kind: 'decodeOmx4dFile', file })
    })
  }

  decodePt(
    file: File,
    options: PtDecodeOptions,
    callbacks: PtDecodeCallbacks = {},
  ): Promise<ViewerDataset> {
    const id = this.nextId++
    return new Promise((resolve, reject) => {
      if (callbacks.signal?.aborted) {
        reject(new DOMException('Decode cancelled.', 'AbortError'))
        return
      }

      const onAbort = () => {
        const request = this.pending.get(id)
        if (!request) return
        this.pending.delete(id)
        request.cleanup?.()
        this.worker.postMessage({ id, kind: 'cancel' })
        reject(new DOMException('Decode cancelled.', 'AbortError'))
      }
      callbacks.signal?.addEventListener('abort', onAbort, { once: true })
      this.pending.set(id, {
        resolve,
        reject,
        onProgress: callbacks.onProgress,
        cleanup: () => callbacks.signal?.removeEventListener('abort', onAbort),
      })
      this.worker.postMessage({ id, kind: 'decodePt', file, options })
    })
  }

  dispose(): void {
    this.worker.terminate()
    for (const request of this.pending.values()) {
      request.cleanup?.()
      request.reject(new Error('Decode cancelled.'))
    }
    this.pending.clear()
  }
}
