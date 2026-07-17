import type { ViewerDataset } from './dataset'

interface WorkerSuccess {
  id: number
  ok: true
  dataset: ViewerDataset
}

interface WorkerFailure {
  id: number
  ok: false
  error: string
}

type WorkerResponse = WorkerSuccess | WorkerFailure

interface PendingDecode {
  resolve: (dataset: ViewerDataset) => void
  reject: (error: Error) => void
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
      this.pending.delete(result.id)
      if (result.ok) request.resolve(result.dataset)
      else request.reject(new Error(result.error))
    }
    this.worker.onerror = (event) => {
      const error = new Error(event.message || 'The decoder worker stopped unexpectedly.')
      for (const request of this.pending.values()) request.reject(error)
      this.pending.clear()
    }
  }

  decode(buffer: ArrayBuffer): Promise<ViewerDataset> {
    const id = this.nextId++
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject })
      this.worker.postMessage({ id, buffer }, [buffer])
    })
  }

  dispose(): void {
    this.worker.terminate()
    for (const request of this.pending.values()) request.reject(new Error('Decode cancelled.'))
    this.pending.clear()
  }
}
