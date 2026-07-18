import { expect, test } from '@playwright/test'

test('supports deterministic scrubbing, transport settings, and keyboard shortcuts', async ({ page }) => {
  const errors: string[] = []
  page.on('pageerror', (error) => errors.push(error.message))

  await page.goto('/')
  await expect(page.getByText('100,000 pts')).toBeVisible()

  const frame = page.locator('.timeline-frame')
  const timeline = page.getByRole('slider', { name: 'Timeline' })
  await timeline.fill('5')
  await expect(frame).toHaveText('Frame 06 / 16')

  await page.locator('canvas').click({ position: { x: 20, y: 20 } })
  await page.keyboard.press('ArrowRight')
  await expect(frame).toHaveText('Frame 07 / 16')
  await page.keyboard.press('ArrowLeft')
  await expect(frame).toHaveText('Frame 06 / 16')
  await page.keyboard.press('Home')
  await expect(frame).toHaveText('Frame 01 / 16')

  const loop = page.getByRole('button', { name: 'Loop (L)' })
  await expect(loop).toHaveClass(/active/)
  await page.keyboard.press('l')
  await expect(loop).not.toHaveClass(/active/)

  await page.getByLabel('Speed').selectOption('2')
  await expect(page.getByLabel('Speed')).toHaveValue('2')
  await page.locator('canvas').click({ position: { x: 20, y: 20 } })
  await page.keyboard.press('Space')
  await expect(page.getByRole('button', { name: 'Pause' })).toBeVisible()
  await page.keyboard.press('Space')
  await expect(page.getByRole('button', { name: 'Play' })).toBeVisible()

  const trails = page.getByRole('switch', { name: 'Motion trails' })
  await trails.click()
  await expect(trails).toHaveAttribute('aria-checked', 'true')
  expect(errors).toEqual([])
})

test('shows a sanitized error and keeps the current scene after a malformed upload', async ({ page }) => {
  const errors: string[] = []
  page.on('pageerror', (error) => errors.push(error.message))

  await page.goto('/')
  await expect(page.getByText('100,000 pts')).toBeVisible()

  await page.locator('input[type="file"][accept^=".pt"]').setInputFiles({
    name: 'malformed.pt',
    mimeType: 'application/octet-stream',
    buffer: Buffer.from('this is not a torch archive'),
  })

  const alert = page.getByRole('alert')
  await expect(alert).toContainText('The tensors do not match the OmniX schema')
  await expect(alert).toContainText('The file is not a supported ZIP-based torch.save archive.')
  await alert.getByRole('button', { name: 'Keep current scene' }).click()
  await expect(page.getByText('100,000 pts')).toBeVisible()
  expect(errors).toEqual([])
})

test('reports a format-specific error for malformed OMX4D and keeps the scene', async ({ page }) => {
  const errors: string[] = []
  page.on('pageerror', (error) => errors.push(error.message))

  await page.goto('/')
  await expect(page.getByText('100,000 pts')).toBeVisible()

  await page.locator('input[type="file"][accept*=".omx4d"]').setInputFiles({
    name: 'malformed.omx4d',
    mimeType: 'application/octet-stream',
    buffer: Buffer.from('bad OMX4D'),
  })

  const alert = page.getByRole('alert')
  await expect(alert).toContainText('The OMX4D renderer payload is invalid')
  await expect(alert).toContainText('OMX4D payload must contain at least 16 bytes.')
  await alert.getByRole('button', { name: 'Keep current scene' }).click()
  await expect(page.getByText('100,000 pts')).toBeVisible()
  expect(errors).toEqual([])
})
