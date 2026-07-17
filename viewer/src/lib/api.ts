export interface ApiFailure {
  title: string
  detail: string
}

const statusTitles: Record<number, string> = {
  400: 'This file is not a supported OmniX output',
  408: 'Conversion timed out',
  413: 'The selected file is too large',
  415: 'Unsupported file type',
  422: 'The tensors do not match the OmniX schema',
  429: 'The converter is busy',
  503: 'The local converter is unavailable',
}

export function mapApiError(status: number, body: string): ApiFailure {
  let detail = body.trim()
  try {
    const parsed = JSON.parse(body) as {
      detail?: unknown
      message?: unknown
      error?: string | { message?: unknown } | null
    }
    const nestedError = parsed.error && typeof parsed.error === 'object'
      ? parsed.error.message
      : parsed.error
    const candidate = [parsed.detail, parsed.message, nestedError]
      .find((value): value is string => typeof value === 'string')
    if (candidate) detail = candidate
  } catch {
    // Plain-text errors from a reverse proxy remain useful.
  }

  if (!detail || /<html/i.test(detail)) {
    detail = status >= 500
      ? 'Start the local conversion service and try again.'
      : 'Check the file and conversion settings, then try again.'
  }

  return {
    title: statusTitles[status] ?? (status >= 500 ? 'Conversion service error' : 'Conversion failed'),
    detail: detail.slice(0, 500),
  }
}
