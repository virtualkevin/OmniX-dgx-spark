export const PLAYBACK_RATES = [0.25, 0.5, 1, 1.5, 2] as const

export interface PlayerState {
  frame: number
  frameCount: number
  fps: number
  playing: boolean
  loop: boolean
  rate: number
}

export type PlayerAction =
  | { type: 'load'; frameCount: number; fps: number }
  | { type: 'seek'; frame: number }
  | { type: 'step'; delta: number }
  | { type: 'setPlaying'; playing: boolean }
  | { type: 'togglePlaying' }
  | { type: 'toggleLoop' }
  | { type: 'setRate'; rate: number }
  | { type: 'setFps'; fps: number }
  | { type: 'restart' }

export const initialPlayerState: PlayerState = {
  frame: 0,
  frameCount: 1,
  fps: 15,
  playing: false,
  loop: true,
  rate: 1,
}

export function clampFrame(frame: number, frameCount: number): number {
  const last = Math.max(0, Math.floor(frameCount) - 1)
  if (!Number.isFinite(frame)) return 0
  return Math.min(last, Math.max(0, Math.floor(frame)))
}

export function clampFps(fps: number): number {
  if (!Number.isFinite(fps)) return 15
  return Math.min(120, Math.max(1, fps))
}

export function frameAtTime(timeSeconds: number, fps: number, frameCount: number): number {
  if (!Number.isFinite(timeSeconds)) return 0
  return clampFrame(Math.floor(Math.max(0, timeSeconds) * clampFps(fps)), frameCount)
}

export function frameAtMediaTime(
  mediaTimeSeconds: number,
  syncOffsetSeconds: number,
  fps: number,
  frameCount: number,
): number {
  return frameAtTime(mediaTimeSeconds - syncOffsetSeconds, fps, frameCount)
}

export function timeForFrame(frame: number, fps: number): number {
  return Math.max(0, frame) / clampFps(fps)
}

export function durationSeconds(frameCount: number, fps: number): number {
  return Math.max(0, frameCount) / clampFps(fps)
}

export function formatTime(seconds: number): string {
  const safe = Math.max(0, Number.isFinite(seconds) ? seconds : 0)
  const minutes = Math.floor(safe / 60)
  const remainder = safe - minutes * 60
  return `${String(minutes).padStart(2, '0')}:${remainder.toFixed(2).padStart(5, '0')}`
}

export function playerReducer(state: PlayerState, action: PlayerAction): PlayerState {
  switch (action.type) {
    case 'load':
      return {
        ...state,
        frame: 0,
        frameCount: Math.max(1, Math.floor(action.frameCount)),
        fps: clampFps(action.fps),
        playing: false,
      }
    case 'seek':
      return { ...state, frame: clampFrame(action.frame, state.frameCount) }
    case 'step':
      return {
        ...state,
        frame: clampFrame(state.frame + action.delta, state.frameCount),
        playing: false,
      }
    case 'setPlaying':
      return { ...state, playing: action.playing }
    case 'togglePlaying':
      return { ...state, playing: !state.playing }
    case 'toggleLoop':
      return { ...state, loop: !state.loop }
    case 'setRate':
      return {
        ...state,
        rate: PLAYBACK_RATES.includes(action.rate as (typeof PLAYBACK_RATES)[number])
          ? action.rate
          : 1,
      }
    case 'setFps':
      return { ...state, fps: clampFps(action.fps) }
    case 'restart':
      return { ...state, frame: 0, playing: false }
  }
}
