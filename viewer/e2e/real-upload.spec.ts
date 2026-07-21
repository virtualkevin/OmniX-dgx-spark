import { expect, test, type Page } from '@playwright/test'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const realPtPath = process.env.OMNIX_REAL_PT
const realOmx4dPath = process.env.OMNIX_REAL_OMX4D
const hardwareWebGl = process.env.OMNIX_E2E_HARDWARE === '1'
const testDirectory = path.dirname(fileURLToPath(import.meta.url))
const bundledOmx4dPath = path.resolve(testDirectory, '../public/sample/deer.omx4d')

test.describe.configure({ mode: 'serial' })

interface BrowserMonitor {
  beginSelection: () => void
  errors: string[]
  requests: string[]
}

function monitorBrowserOnlyExecution(page: Page): BrowserMonitor {
  const errors: string[] = []
  const requests: string[] = []
  let selectionStarted = false

  page.on('pageerror', (error) => errors.push('pageerror: ' + error.message))
  page.on('console', (message) => {
    if (message.type() === 'error') errors.push('console: ' + message.text())
  })
  page.on('request', (request) => {
    const url = request.url()
    if (selectionStarted && !url.startsWith('blob:') && !url.startsWith('data:')) {
      requests.push(request.method() + ' ' + url)
    }
  })

  return {
    beginSelection: () => { selectionStarted = true },
    errors,
    requests,
  }
}

function datasetInput(page: Page, extension: '.pt' | '.omx4d') {
  return page.locator('input[type="file"][accept*="' + extension + '"]')
}

async function expectWebGlRenderer(page: Page) {
  const canvas = page.locator('canvas')
  await expect(canvas).toBeVisible()

  // Do not create a default 300 by 150 context before R3F sizes its renderer.
  await expect.poll(() => canvas.evaluate((node) => (
    node.width > 300
    && node.height > 150
    && node.clientWidth > 0
    && node.clientHeight > 0
  )), { timeout: 15_000 }).toBe(true)
  await expect.poll(() => canvas.evaluate((node) => Boolean(
    node.getContext('webgl2') ?? node.getContext('webgl'),
  ))).toBe(true)

  if (hardwareWebGl) {
    const renderer = await canvas.evaluate((node) => {
      const gl = node.getContext('webgl2') ?? node.getContext('webgl')
      if (!gl) return null
      const debugInfo = gl.getExtension('WEBGL_debug_renderer_info')
      return debugInfo
        ? String(gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL))
        : null
    })
    expect(renderer, 'hardware mode requires unmasked renderer information').not.toBeNull()
    expect(renderer ?? '', 'hardware mode must not fall back to a software renderer')
      .not.toMatch(/swiftshader|llvmpipe|software rasterizer/i)
  }
}

async function expectWebGlContextHealthy(page: Page) {
  const contextHealthy = await page.locator('canvas').evaluate((node) => {
    const gl = node.getContext('webgl2') ?? node.getContext('webgl')
    return Boolean(gl && !gl.isContextLost())
  })
  expect(contextHealthy, 'the WebGL context must survive full playback').toBe(true)
}

async function waitForCanvasFrame(page: Page) {
  await page.evaluate(() => new Promise<void>((resolve) => {
    requestAnimationFrame(() => requestAnimationFrame(() => resolve()))
  }))
}

async function expectSeekPlayAndFullLoop(page: Page) {
  const frame = page.locator('.timeline-frame')
  const timeline = page.getByRole('slider', { name: 'Timeline' })
  const canvas = page.locator('canvas')

  await expect(frame).toHaveText('Frame 01 / 32')
  await expect(timeline).toHaveAttribute('max', '31')
  await waitForCanvasFrame(page)
  const frameOneImage = await canvas.screenshot()
  await timeline.fill('16')
  await expect(frame).toHaveText('Frame 17 / 32')
  await waitForCanvasFrame(page)
  const frameSeventeenImage = await canvas.screenshot()
  expect(
    frameSeventeenImage.equals(frameOneImage),
    'seeking must visibly change the rendered point cloud',
  ).toBe(false)
  await timeline.fill('0')
  await expect(frame).toHaveText('Frame 01 / 32')
  await expect(page.getByRole('button', { name: 'Loop (L)' })).toHaveClass(/active/)

  await page.evaluate(() => {
    const output = document.querySelector<HTMLElement>('.timeline-frame')
    if (!output) throw new Error('The timeline frame readout is missing.')
    const seen = new Set<number>()
    let previous = 0
    let wraps = 0
    let startedAt: number | undefined
    const record = () => {
      const current = Number(output.textContent?.match(/Frame (\d+) \/ 32/)?.[1] ?? 0)
      if (!current) return
      if (previous && current !== previous && startedAt === undefined) {
        startedAt = performance.now()
      }
      if (previous && current < previous) {
        wraps += 1
        if (startedAt !== undefined) {
          document.body.dataset.omnixLoopElapsedMs = String(performance.now() - startedAt)
        }
      }
      previous = current
      seen.add(current)
      document.body.dataset.omnixFramesSeen = String(seen.size)
      document.body.dataset.omnixFrameWraps = String(wraps)
    }
    new MutationObserver(record).observe(output, {
      childList: true,
      characterData: true,
      subtree: true,
    })
    record()
  })

  await page.getByRole('button', { name: 'Play' }).click()
  await expect(page.getByRole('button', { name: 'Pause' })).toBeVisible()
  await expect.poll(() => page.locator('body').getAttribute('data-omnix-frames-seen'), {
    message: 'playback should present all 32 frames',
    timeout: 20_000,
    intervals: [100, 200, 500],
  }).toBe('32')
  await expect.poll(() => page.locator('body').getAttribute('data-omnix-frame-wraps'), {
    message: 'playback should loop from frame 32 back to frame 1',
    timeout: 5_000,
  }).not.toBe('0')
  await page.getByRole('button', { name: 'Pause' }).click()
  await expectWebGlContextHealthy(page)

  if (hardwareWebGl) {
    const loopElapsedMs = Number(
      await page.locator('body').getAttribute('data-omnix-loop-elapsed-ms'),
    )
    expect(loopElapsedMs, '32 frames at 8 FPS should not run too quickly')
      .toBeGreaterThanOrEqual(3_000)
    expect(loopElapsedMs, '32 frames at 8 FPS should sustain realtime cadence')
      .toBeLessThanOrEqual(5_500)
  }
}

function expectNoServerDependency(monitor: BrowserMonitor) {
  expect(
    monitor.requests,
    'file selection must not call an upload, API, Python, or conversion server',
  ).toEqual([])
  expect(monitor.errors, 'the browser must not emit page or console errors').toEqual([])
}

test('samples at 500k, renders, seeks, and loops a real predictions.pt in-browser', async ({ page }) => {
  test.skip(!realPtPath, 'Set OMNIX_REAL_PT to run the full browser-only .pt gate')
  test.setTimeout(600_000)

  const monitor = monitorBrowserOnlyExecution(page)
  await page.goto('/')
  await expect(page.getByText('100,000 pts')).toBeVisible()

  const quality = page.locator('.quality-select select')
  await quality.selectOption('500000')
  await expect(quality).toHaveValue('500000')
  const fps = page.getByRole('spinbutton', { name: 'FPS', exact: true })
  await fps.fill('8')
  await expect(fps).toHaveValue('8')

  const input = datasetInput(page, '.pt')
  await expect(input).toHaveCount(1)
  monitor.beginSelection()
  await input.setInputFiles(realPtPath!)
  const cancel = page.getByRole('button', { name: 'Cancel' })
  await expect(cancel).toBeVisible()
  await cancel.click()
  await expect(cancel).toBeHidden()
  await expect(page.getByRole('button', { name: 'Play' })).toBeEnabled()

  await input.setInputFiles(realPtPath!)
  await expect(page.getByText('500,000 pts', { exact: true })).toBeVisible({ timeout: 540_000 })

  await expectWebGlRenderer(page)
  await expectSeekPlayAndFullLoop(page)
  expectNoServerDependency(monitor)
})

test('opens a local OMX4D through the picker without a server conversion', async ({ page }) => {
  const monitor = monitorBrowserOnlyExecution(page)
  await page.goto('/')
  await expect(page.getByText('100,000 pts')).toBeVisible()

  const input = datasetInput(page, '.omx4d')
  monitor.beginSelection()
  await input.setInputFiles(bundledOmx4dPath)
  await expect(page.getByText('deer.omx4d', { exact: true })).toBeVisible()
  await expect(page.locator('.timeline-frame')).toHaveText('Frame 01 / 16')

  await expectWebGlRenderer(page)
  await page.getByRole('button', { name: 'Play' }).click()
  await expect.poll(() => page.locator('.timeline-frame').textContent()).not.toBe('Frame 01 / 16')
  await page.getByRole('button', { name: 'Pause' }).click()
  expectNoServerDependency(monitor)
})

test('opens, renders, seeks, and loops a real 500k OMX4D in-browser', async ({ page }) => {
  test.skip(!realOmx4dPath, 'Set OMNIX_REAL_OMX4D to run the full browser-only OMX4D gate')
  test.setTimeout(180_000)

  const monitor = monitorBrowserOnlyExecution(page)
  await page.goto('/')
  await expect(page.getByText('100,000 pts')).toBeVisible()

  const input = datasetInput(page, '.omx4d')
  await expect(input).toHaveCount(1)
  monitor.beginSelection()
  await input.setInputFiles(realOmx4dPath!)
  await expect(page.getByText('500,000 pts', { exact: true })).toBeVisible({ timeout: 120_000 })
  await page.getByRole('button', { name: 'Toggle diagnostics' }).click()
  await expect(page.getByText('Source RGB was not supplied', { exact: false })).toHaveCount(0)
  await expect(page.getByRole('spinbutton', { name: 'FPS', exact: true })).toHaveValue('8')

  await expectWebGlRenderer(page)
  await expectSeekPlayAndFullLoop(page)
  expectNoServerDependency(monitor)
})

test('does not let a delayed baked sample replace a user-selected OMX4D', async ({ page }) => {
  let releaseSample = () => {}
  const sampleGate = new Promise<void>((resolve) => { releaseSample = resolve })
  let finishRoute = () => {}
  const routeFinished = new Promise<void>((resolve) => { finishRoute = resolve })

  await page.route('**/sample/deer.omx4d', async (route) => {
    await sampleGate
    try {
      await route.continue()
    } catch {
      // Selecting a local file aborts the initial fetch before this route resumes.
    } finally {
      finishRoute()
    }
  })

  await page.goto('/')
  const input = datasetInput(page, '.omx4d')
  await input.setInputFiles(bundledOmx4dPath)
  await expect(page.getByText('deer.omx4d', { exact: true })).toBeVisible()

  releaseSample()
  await routeFinished
  await waitForCanvasFrame(page)
  await expect(page.getByText('deer.omx4d', { exact: true })).toBeVisible()
})
