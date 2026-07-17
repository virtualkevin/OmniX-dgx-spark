import { describe, expect, it } from 'vitest'
import { mapApiError } from './api'

describe('mapApiError', () => {
  it('extracts safe JSON detail', () => {
    expect(mapApiError(422, '{"detail":"trajectory has the wrong shape"}')).toEqual({
      title: 'The tensors do not match the OmniX schema',
      detail: 'trajectory has the wrong shape',
    })
  })

  it('extracts the converter nested error envelope', () => {
    expect(mapApiError(400, '{"error":{"code":"invalid_pt_archive","message":"Not a torch archive."}}')).toEqual({
      title: 'This file is not a supported OmniX output',
      detail: 'Not a torch archive.',
    })
  })

  it('does not show proxy HTML', () => {
    expect(mapApiError(503, '<html>offline</html>').detail).toContain('local conversion service')
  })
})
