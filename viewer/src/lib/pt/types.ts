export type PtParseErrorCode =
  | 'INVALID_ZIP'
  | 'UNSUPPORTED_ZIP'
  | 'UNSAFE_ZIP_ENTRY'
  | 'MISSING_TORCH_ENTRY'
  | 'ENTRY_TOO_LARGE'
  | 'INVALID_ENTRY_RANGE'
  | 'INVALID_PICKLE'
  | 'UNSUPPORTED_PICKLE'
  | 'UNSAFE_PICKLE'
  | 'INVALID_TENSOR_SCHEMA'
  | 'RESOURCE_LIMIT'

/** A deliberately user-safe parse failure with no attacker-controlled text. */
export class PtParseError extends Error {
  readonly code: PtParseErrorCode

  constructor(code: PtParseErrorCode, message: string, options?: ErrorOptions) {
    super(message, options)
    this.name = 'PtParseError'
    this.code = code
  }
}

export interface StorageReference {
  readonly key: string
  readonly dtype: 'float32'
  readonly location: 'cpu'
  readonly elementCount: number
  readonly byteLength: number
}

export interface TensorDescriptor {
  readonly storage: StorageReference
  readonly storageOffset: number
  readonly shape: readonly number[]
  readonly stride: readonly number[]
  readonly requiresGrad: false
}
