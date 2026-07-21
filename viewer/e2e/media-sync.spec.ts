import { expect, test } from '@playwright/test'

function silentWav(durationSeconds = 2, sampleRate = 44_100): Buffer {
  const sampleCount = durationSeconds * sampleRate
  const payloadBytes = sampleCount * 2
  const wav = Buffer.alloc(44 + payloadBytes)
  wav.write('RIFF', 0)
  wav.writeUInt32LE(36 + payloadBytes, 4)
  wav.write('WAVE', 8)
  wav.write('fmt ', 12)
  wav.writeUInt32LE(16, 16)
  wav.writeUInt16LE(1, 20)
  wav.writeUInt16LE(1, 22)
  wav.writeUInt32LE(sampleRate, 24)
  wav.writeUInt32LE(sampleRate * 2, 28)
  wav.writeUInt16LE(2, 32)
  wav.writeUInt16LE(16, 34)
  wav.write('data', 36)
  wav.writeUInt32LE(payloadBytes, 40)
  return wav
}

test('uses uploaded source audio as the synchronized playback clock', async ({ page }) => {
  const errors: string[] = []
  page.on('pageerror', (error) => errors.push(error.message))
  page.on('console', (message) => {
    if (message.type() === 'error') errors.push(message.text())
  })

  await page.goto('/')
  await expect(page.getByText('100,000 pts')).toBeVisible()

  const chooserPromise = page.waitForEvent('filechooser')
  await page.getByRole('button', { name: 'Add source media' }).click()
  const chooser = await chooserPromise
  await chooser.setFiles({
    name: 'source-clock.wav',
    mimeType: 'audio/wav',
    buffer: silentWav(),
  })
  await expect(page.getByText('source-clock.wav')).toBeVisible()
  await expect.poll(() => page.locator('audio').evaluate((node) => node.readyState)).toBeGreaterThanOrEqual(1)

  const frame = page.locator('.timeline-frame')
  await page.getByRole('button', { name: 'Play' }).click()
  await expect.poll(() => frame.textContent()).not.toBe('Frame 01 / 16')
  await page.getByRole('button', { name: 'Pause' }).click()

  await page.getByRole('button', { name: 'Mute' }).click()
  await expect(page.getByRole('button', { name: 'Unmute' })).toBeVisible()
  await page.getByRole('button', { name: 'Remove source media' }).click()
  await expect(page.getByRole('button', { name: 'Add source media' })).toBeVisible()
  expect(errors).toEqual([])
})
