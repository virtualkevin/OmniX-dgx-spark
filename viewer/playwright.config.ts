import { defineConfig, devices } from '@playwright/test'

const hardwareWebGl = process.env.OMNIX_E2E_HARDWARE === '1'
const realFileGate = Boolean(
  process.env.OMNIX_REAL_PT || process.env.OMNIX_REAL_OMX4D,
)
const webglArgs = hardwareWebGl
  ? [
    '--enable-gpu',
    '--use-gl=angle',
    '--use-angle=vulkan',
    '--enable-features=Vulkan,VulkanFromANGLE',
    '--enable-webgl',
    '--ignore-gpu-blocklist',
    '--ozone-platform=x11',
  ]
  : [
    '--use-gl=angle',
    '--use-angle=swiftshader',
    '--enable-webgl',
    '--ignore-gpu-blocklist',
  ]

export default defineConfig({
  testDir: './e2e',
  timeout: 30_000,
  expect: { timeout: 15_000 },
  fullyParallel: !realFileGate,
  workers: realFileGate ? 1 : undefined,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 1 : 0,
  reporter: 'line',
  use: {
    baseURL: 'http://127.0.0.1:4173',
    headless: !hardwareWebGl,
    channel: hardwareWebGl ? 'chromium' : undefined,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    launchOptions: { args: webglArgs },
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
  webServer: {
    command: 'pnpm dev',
    url: 'http://127.0.0.1:4173',
    reuseExistingServer: !process.env.CI,
    timeout: 30_000,
  },
})
