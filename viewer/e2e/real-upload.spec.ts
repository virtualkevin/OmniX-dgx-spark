import { expect, test } from '@playwright/test'

const realPtPath = process.env.OMNIX_REAL_PT

test.skip(!realPtPath, 'Set OMNIX_REAL_PT to run the full local conversion gate')

test('uploads, converts, renders, and plays a real predictions.pt', async ({ page }) => {
  test.setTimeout(360_000)

  const errors: string[] = []
  page.on('pageerror', (error) => errors.push(error.message))
  page.on('console', (message) => {
    if (message.type() === 'error') errors.push(message.text())
  })

  await page.goto('/')
  await expect(page.getByText('100,000 pts')).toBeVisible()

  await page.locator('input[type="file"][accept^=".pt"]').setInputFiles(realPtPath!)
  await expect(page.getByText('predictions.pt', { exact: true })).toBeVisible({ timeout: 330_000 })
  await expect(page.getByText('100,000 pts')).toBeVisible()

  const frame = page.locator('.timeline-frame')
  await expect(frame).toHaveText('Frame 01 / 16')
  await page.getByRole('button', { name: 'Play' }).click()
  await expect.poll(() => frame.textContent()).not.toBe('Frame 01 / 16')
  await page.getByRole('button', { name: 'Pause' }).click()

  await expect(page.locator('canvas')).toBeVisible()
  expect(errors).toEqual([])
})
