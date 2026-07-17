import { expect, test } from '@playwright/test'

test('renders the baked point cloud and advances real-time playback', async ({ page }) => {
  const errors: string[] = []
  page.on('pageerror', (error) => errors.push(error.message))
  page.on('console', (message) => {
    if (message.type() === 'error') errors.push(message.text())
  })

  await page.goto('/')
  await expect(page.getByLabel('OmniX 4D Viewer')).toBeVisible()
  await expect(page.getByText('100,000 pts')).toBeVisible()

  const canvas = page.locator('canvas')
  await expect(canvas).toBeVisible()
  await expect.poll(() => canvas.evaluate((node) => Boolean(
    node.getContext('webgl2') ?? node.getContext('webgl'),
  ))).toBe(true)

  const frame = page.locator('.timeline-frame')
  await expect(frame).toHaveText('Frame 01 / 16')
  await page.getByRole('button', { name: 'Play' }).click()
  await expect.poll(() => frame.textContent()).not.toBe('Frame 01 / 16')
  await page.getByRole('button', { name: 'Pause' }).click()

  await page.getByRole('button', { name: 'Dynamic', exact: true }).click()
  await expect(page.getByRole('button', { name: 'Dynamic', exact: true })).toHaveClass(/active/)
  await page.getByRole('combobox').filter({ has: page.locator('option[value="-1"]') }).selectOption('0')
  await page.getByRole('button', { name: 'Reset camera' }).click()
  expect(errors).toEqual([])
})

test('keeps core controls usable at the compact breakpoint', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto('/')
  await expect(page.getByText('100,000 pts')).toBeVisible()
  await expect(page.getByRole('button', { name: 'Open .pt' })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Play' })).toBeVisible()
  await expect(page.getByRole('slider', { name: 'Timeline' })).toBeVisible()
})
