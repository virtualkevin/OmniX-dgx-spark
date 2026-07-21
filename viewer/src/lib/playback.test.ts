import { describe, expect, it } from 'vitest'
import {
  clampFrame,
  durationSeconds,
  frameAtMediaTime,
  frameAtTime,
  initialPlayerState,
  playerReducer,
} from './playback'

describe('playback math', () => {
  it('clamps frame and time inputs', () => {
    expect(clampFrame(-2, 16)).toBe(0)
    expect(clampFrame(99, 16)).toBe(15)
    expect(frameAtTime(0.999, 15, 16)).toBe(14)
    expect(durationSeconds(16, 15)).toBeCloseTo(1.0667, 3)
  })

  it('applies media sync offset before choosing a frame', () => {
    expect(frameAtMediaTime(1.5, 0.5, 15, 100)).toBe(15)
    expect(frameAtMediaTime(0.2, 0.5, 15, 100)).toBe(0)
  })
})

describe('playerReducer', () => {
  it('loads, seeks and resets a dataset deterministically', () => {
    let state = playerReducer(initialPlayerState, { type: 'load', frameCount: 16, fps: 12 })
    state = playerReducer(state, { type: 'seek', frame: 8.9 })
    expect(state.frame).toBe(8)
    expect(state.fps).toBe(12)
    state = playerReducer(state, { type: 'step', delta: 99 })
    expect(state.frame).toBe(15)
    expect(state.playing).toBe(false)
  })
})
