import { expect, test } from '@playwright/test'

const realPtPath = process.env.OMNIX_REAL_PT

test.skip(!realPtPath, 'Set OMNIX_REAL_PT to run the full browser-only archive gate')

test('reads, samples, renders, and plays a real predictions.pt entirely in the browser', async ({ page }) => {
  test.setTimeout(360_000)

  const errors: string[] = []
  const apiRequests: string[] = []
  const requestsAfterSelection: string[] = []
  let selectionStarted = false
  page.on('pageerror', (error) => errors.push(error.message))
  page.on('request', (request) => {
    if (selectionStarted) requestsAfterSelection.push(request.url())
  })
  page.on('console', (message) => {
    if (message.type() === 'error') errors.push(message.text())
  })
  await page.route('**/api/**', async (route) => {
    apiRequests.push(route.request().url())
    await route.abort()
  })

  await page.goto('/')
  await expect(page.getByText('100,000 pts')).toBeVisible()

  const input = page.locator('input[type="file"][accept^=".pt"]')
  selectionStarted = true
  await input.setInputFiles(realPtPath!)
  const cancel = page.getByRole('button', { name: 'Cancel' })
  await expect(cancel).toBeVisible()
  await cancel.click()
  await expect(cancel).toBeHidden()
  await expect(page.getByRole('button', { name: 'Play' })).toBeEnabled()

  await input.setInputFiles(realPtPath!)
  await expect(page.getByText('predictions.pt', { exact: true })).toBeVisible({ timeout: 330_000 })
  await expect(page.getByText('100,000 pts')).toBeVisible()

  const frame = page.locator('.timeline-frame')
  await expect(frame).toHaveText('Frame 01 / 16')
  await page.getByRole('button', { name: 'Play' }).click()
  await expect.poll(() => frame.textContent()).not.toBe('Frame 01 / 16')
  await page.getByRole('button', { name: 'Pause' }).click()

  await expect(page.locator('canvas')).toBeVisible()
  expect(apiRequests).toEqual([])
  expect(requestsAfterSelection).toEqual([])
  expect(errors).toEqual([])
})
