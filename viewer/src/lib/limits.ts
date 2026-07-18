export const GIB = 1024 ** 3

/**
 * The current 32-view, 32-frame OmniX inference artifacts are about 1.75 GB.
 * Keep one shared ceiling so the UI, ZIP reader, and tensor validators cannot
 * silently disagree about whether a real repository output is supported.
 */
export const MAX_BROWSER_DATASET_FILE_BYTES = 2 * GIB
export const MAX_BROWSER_TENSOR_BYTES = 2 * GIB
export const MAX_BROWSER_POINT_BUDGET = 500_000
