import type { Omx4dManifest } from './omx4d'

export interface ViewerDataset {
  manifest: Omx4dManifest
  positions: Float32Array
  colors: Uint8Array
  dynamicScore: Float32Array
  sourceView: Uint16Array
  cameraPose: Float32Array
  intrinsics: Float32Array
}

export type ColorMode = 'rgb' | 'dynamic' | 'source' | 'depth'

export interface RenderSettings {
  colorMode: ColorMode
  dynamicThreshold: number
  selectedView: number
  pointSize: number
  trails: boolean
  grid: boolean
  cameraFrusta: boolean
}
