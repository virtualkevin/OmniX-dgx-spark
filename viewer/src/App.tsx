import {
  AlertTriangle,
  AudioLines,
  Box,
  ChevronLeft,
  ChevronRight,
  FileUp,
  Gauge,
  Grid3X3,
  Info,
  LoaderCircle,
  PanelRight,
  Pause,
  Play,
  Repeat2,
  RotateCcw,
  SlidersHorizontal,
  Upload,
  Video,
  Volume2,
  VolumeX,
  X,
} from 'lucide-react'
import {
  useCallback,
  useEffect,
  useMemo,
  useReducer,
  useRef,
  useState,
  type CSSProperties,
  type ChangeEvent,
  type DragEvent,
  type ReactNode,
} from 'react'
import { ViewerScene } from './components/Scene'
import type { ColorMode, RenderSettings, ViewerDataset } from './lib/dataset'
import { DecoderClient } from './lib/decoder-client'
import {
  PLAYBACK_RATES,
  durationSeconds,
  formatTime,
  initialPlayerState,
  playerReducer,
  timeForFrame,
} from './lib/playback'

type LoadPhase = 'sample' | 'ready' | 'processing' | 'error'

interface LoadFailure {
  title: string
  detail: string
}

interface LoadState {
  phase: LoadPhase
  progress?: number
  status?: string
  error?: LoadFailure
}

interface MediaState {
  url: string
  name: string
  kind: 'audio' | 'video'
  duration: number | null
  status: 'loading' | 'ready' | 'error'
  error?: string
}

const defaultSettings: RenderSettings = {
  colorMode: 'rgb',
  dynamicThreshold: 0,
  selectedView: -1,
  pointSize: 2.2,
  trails: false,
  grid: true,
  cameraFrusta: false,
}

const MAX_PT_FILE_BYTES = 1024 ** 3

function supportsWebGl(): boolean {
  try {
    const canvas = document.createElement('canvas')
    return Boolean(canvas.getContext('webgl2') || canvas.getContext('webgl'))
  } catch {
    return false
  }
}

function Toggle({ checked, onChange, label }: { checked: boolean; onChange: () => void; label: string }) {
  return (
    <button
      type="button"
      className={`toggle ${checked ? 'toggle--on' : ''}`}
      role="switch"
      aria-checked={checked}
      onClick={onChange}
    >
      <span className="toggle__track"><span /></span>
      <span>{label}</span>
    </button>
  )
}

function IconButton({
  label,
  children,
  active = false,
  disabled = false,
  onClick,
}: {
  label: string
  children: ReactNode
  active?: boolean
  disabled?: boolean
  onClick: () => void
}) {
  return (
    <button
      type="button"
      className={`icon-button ${active ? 'icon-button--active' : ''}`}
      aria-label={label}
      title={label}
      disabled={disabled}
      onClick={onClick}
    >
      {children}
    </button>
  )
}

export default function App() {
  const [dataset, setDataset] = useState<ViewerDataset | null>(null)
  const [datasetName, setDatasetName] = useState('Deer · baked sample')
  const [loadState, setLoadState] = useState<LoadState>({ phase: 'sample' })
  const [player, dispatch] = useReducer(playerReducer, initialPlayerState)
  const [settings, setSettings] = useState(defaultSettings)
  const [pointBudget, setPointBudget] = useState(100_000)
  const [diagnosticsOpen, setDiagnosticsOpen] = useState(false)
  const [dragActive, setDragActive] = useState(false)
  const [resetToken, setResetToken] = useState(0)
  const [seekToken, setSeekToken] = useState(0)
  const [media, setMedia] = useState<MediaState | null>(null)
  const [muted, setMuted] = useState(false)
  const [volume, setVolume] = useState(0.8)
  const [syncOffset, setSyncOffset] = useState(0)
  const decoderRef = useRef<DecoderClient | null>(null)
  const conversionRef = useRef<AbortController | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const mediaInputRef = useRef<HTMLInputElement>(null)
  const mediaRef = useRef<HTMLMediaElement | null>(null)
  const webGlAvailable = useMemo(supportsWebGl, [])
  const reducedPerformance = useMemo(() => navigator.hardwareConcurrency > 0 && navigator.hardwareConcurrency <= 4, [])

  const loadSample = useCallback(async (client: DecoderClient, signal?: AbortSignal) => {
    setLoadState({ phase: 'sample' })
    try {
      const response = await fetch('/sample/deer.omx4d', { signal })
      if (!response.ok) throw new Error(`The baked sample returned HTTP ${response.status}.`)
      const buffer = await response.arrayBuffer()
      if (signal?.aborted) return
      const nextDataset = await client.decode(buffer)
      if (signal?.aborted) return
      setDataset(nextDataset)
      setDatasetName(nextDataset.manifest.name || 'Deer · baked sample')
      setSettings((current) => ({ ...current, selectedView: -1, dynamicThreshold: 0 }))
      dispatch({ type: 'load', frameCount: nextDataset.manifest.frameCount, fps: nextDataset.manifest.fps || 15 })
      setSeekToken((token) => token + 1)
      setLoadState({ phase: 'ready' })
    } catch (error) {
      if (signal?.aborted) return
      setLoadState({
        phase: 'error',
        error: {
          title: 'The baked sample could not be loaded',
          detail: error instanceof Error ? error.message : 'Check the static viewer files and try again.',
        },
      })
    }
  }, [])

  useEffect(() => {
    const client = new DecoderClient()
    const abort = new AbortController()
    decoderRef.current = client
    void loadSample(client, abort.signal)
    return () => {
      abort.abort()
      conversionRef.current?.abort()
      client.dispose()
      decoderRef.current = null
    }
  }, [loadSample])

  useEffect(() => {
    const element = mediaRef.current
    if (!element) return
    element.playbackRate = player.rate
    element.volume = volume
    element.muted = muted
  }, [media, muted, player.rate, volume])

  useEffect(() => () => {
    if (media?.url) URL.revokeObjectURL(media.url)
  }, [media?.url])

  const seek = useCallback((frame: number) => {
    const nextFrame = Math.max(0, Math.min(player.frameCount - 1, Math.floor(frame)))
    dispatch({ type: 'seek', frame: nextFrame })
    setSeekToken((token) => token + 1)
    const element = mediaRef.current
    if (element) {
      const target = timeForFrame(nextFrame, player.fps) + syncOffset
      element.currentTime = Math.max(0, Math.min(Number.isFinite(element.duration) ? element.duration : target, target))
    }
  }, [player.fps, player.frameCount, syncOffset])

  const togglePlayback = useCallback(async () => {
    if (!dataset) return
    const element = mediaRef.current
    if (player.playing) {
      element?.pause()
      dispatch({ type: 'setPlaying', playing: false })
      return
    }
    if (player.frame >= player.frameCount - 1 && !player.loop) seek(0)
    if (element) {
      try {
        element.playbackRate = player.rate
        await element.play()
        dispatch({ type: 'setPlaying', playing: true })
      } catch {
        setMedia((current) => current ? {
          ...current,
          status: 'error',
          error: 'Playback was blocked or this media codec is not supported.',
        } : null)
      }
    } else {
      dispatch({ type: 'setPlaying', playing: true })
    }
  }, [dataset, player.frame, player.frameCount, player.loop, player.playing, player.rate, seek])

  const uploadPt = useCallback(async (file: File) => {
    const client = decoderRef.current
    if (!client) return
    if (!file.name.toLowerCase().endsWith('.pt')) {
      setLoadState({ phase: 'error', error: {
        title: 'Choose an OmniX .pt file',
        detail: 'The browser accepts the exact plain-tensor predictions.pt schema produced by OmniX.',
      } })
      return
    }
    if (file.size > MAX_PT_FILE_BYTES) {
      setLoadState({ phase: 'error', error: {
        title: 'The selected file is too large',
        detail: 'The browser-only reader accepts files up to 1 GiB.',
      } })
      return
    }

    conversionRef.current?.abort()
    const controller = new AbortController()
    conversionRef.current = controller
    dispatch({ type: 'setPlaying', playing: false })
    mediaRef.current?.pause()
    setLoadState({
      phase: 'processing',
      progress: 0,
      status: 'Opening the archive in your browser',
    })

    try {
      const nextDataset = await client.decodePt(
        file,
        { pointBudget, fps: player.fps, name: file.name },
        {
          signal: controller.signal,
          onProgress: (progress) => {
            if (controller.signal.aborted) return
            setLoadState({
              phase: 'processing',
              progress: Math.max(0, Math.min(1, progress.progress)),
              status: progress.message,
            })
          },
        },
      )
      if (controller.signal.aborted) return
      setDataset(nextDataset)
      setDatasetName(file.name)
      setSettings((current) => ({ ...current, selectedView: -1, dynamicThreshold: 0 }))
      dispatch({ type: 'load', frameCount: nextDataset.manifest.frameCount, fps: nextDataset.manifest.fps || player.fps })
      setSeekToken((token) => token + 1)
      setResetToken((token) => token + 1)
      setLoadState({ phase: 'ready' })
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') {
        setLoadState(dataset ? { phase: 'ready' } : { phase: 'sample' })
        return
      }
      const detail = error instanceof Error && error.message.length <= 400
        ? error.message
        : 'The browser rejected this archive before allocating renderer data.'
      setLoadState({
        phase: 'error',
        error: {
          title: 'The tensors do not match the OmniX schema',
          detail,
        },
      })
    } finally {
      if (conversionRef.current === controller) conversionRef.current = null
    }
  }, [dataset, player.fps, pointBudget])
  const handleDatasetFile = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) void uploadPt(file)
    event.target.value = ''
  }

  const handleMediaFile = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return
    if (media?.url) URL.revokeObjectURL(media.url)
    mediaRef.current?.pause()
    const kind = file.type.startsWith('video/') ? 'video' : 'audio'
    setMedia({ url: URL.createObjectURL(file), name: file.name, kind, duration: null, status: 'loading' })
    dispatch({ type: 'setPlaying', playing: false })
    event.target.value = ''
  }

  const removeMedia = () => {
    mediaRef.current?.pause()
    if (media?.url) URL.revokeObjectURL(media.url)
    mediaRef.current = null
    setMedia(null)
    dispatch({ type: 'setPlaying', playing: false })
    setSeekToken((token) => token + 1)
  }

  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setDragActive(false)
    const file = Array.from(event.dataTransfer.files).find((candidate) => candidate.name.toLowerCase().endsWith('.pt'))
    if (file) void uploadPt(file)
    else setLoadState({ phase: 'error', error: {
      title: 'No .pt file found',
      detail: 'Drop an OmniX predictions.pt file anywhere in the viewer.',
    } })
  }

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null
      if (target?.matches('input, select, textarea, button, [contenteditable="true"]')) return
      if (event.code === 'Space') {
        event.preventDefault()
        void togglePlayback()
      } else if (event.key === 'ArrowLeft') {
        event.preventDefault()
        seek(player.frame - 1)
      } else if (event.key === 'ArrowRight') {
        event.preventDefault()
        seek(player.frame + 1)
      } else if (event.key === 'Home') {
        event.preventDefault()
        seek(0)
      } else if (event.key.toLowerCase() === 'l') {
        dispatch({ type: 'toggleLoop' })
      } else if (event.key.toLowerCase() === 'r') {
        setResetToken((token) => token + 1)
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [player.frame, seek, togglePlayback])

  const currentTime = timeForFrame(player.frame, player.fps)
  const duration = durationSeconds(player.frameCount, player.fps)
  const timelineProgress = player.frameCount > 1 ? (player.frame / (player.frameCount - 1)) * 100 : 0
  const durationMismatch = Boolean(media?.duration && Math.abs((media.duration - syncOffset) - duration) > 1 / player.fps)
  const busy = loadState.phase === 'processing'
  const warnings = dataset?.manifest.warnings ?? []

  return (
    <main
      className={`app ${dragActive ? 'app--dragging' : ''}`}
      onDragEnter={(event) => { event.preventDefault(); setDragActive(true) }}
      onDragOver={(event) => event.preventDefault()}
      onDragLeave={(event) => {
        if (event.currentTarget === event.target) setDragActive(false)
      }}
      onDrop={onDrop}
    >
      <header className="topbar">
        <div className="brand" aria-label="OmniX 4D Viewer">
          <span className="brand__mark"><Box size={18} strokeWidth={1.7} /></span>
          <span className="brand__name">OmniX</span>
          <span className="brand__division" />
          <span className="brand__product">4D viewer</span>
        </div>

        <div className="dataset-pill" title={datasetName}>
          <span className={`status-dot ${loadState.phase === 'ready' ? 'status-dot--ready' : ''}`} />
          <span className="dataset-pill__name">{datasetName}</span>
          {dataset && <span className="dataset-pill__meta">{dataset.manifest.pointCount.toLocaleString()} pts</span>}
        </div>

        <div className="topbar__actions">
          <label className="quality-select">
            <span>Import quality</span>
            <select value={pointBudget} onChange={(event) => setPointBudget(Number(event.target.value))} disabled={busy}>
              <option value={50_000}>50k · light</option>
              <option value={100_000}>100k · balanced</option>
              <option value={200_000}>200k · fine</option>
            </select>
          </label>
          <button type="button" className="button button--primary" onClick={() => fileInputRef.current?.click()} disabled={busy}>
            <Upload size={16} /> Open .pt
          </button>
          <IconButton label="Toggle diagnostics" active={diagnosticsOpen} onClick={() => setDiagnosticsOpen((open) => !open)}>
            <PanelRight size={18} />
          </IconButton>
        </div>
      </header>

      <input ref={fileInputRef} type="file" accept=".pt,application/octet-stream" hidden onChange={handleDatasetFile} />
      <input ref={mediaInputRef} type="file" accept="video/*,audio/*" hidden onChange={handleMediaFile} />

      <section className="workspace" aria-label="3D point cloud viewport">
        {!webGlAvailable ? (
          <div className="state-card state-card--error">
            <AlertTriangle />
            <strong>WebGL is unavailable</strong>
            <span>Enable browser hardware acceleration to render this dataset.</span>
          </div>
        ) : dataset ? (
          <ViewerScene
            dataset={dataset}
            frame={player.frame}
            player={player}
            settings={settings}
            resetToken={resetToken}
            seekToken={seekToken}
            mediaRef={mediaRef}
            syncOffset={syncOffset}
            onFrame={(frame) => dispatch({ type: 'seek', frame })}
            onPlaybackEnd={() => dispatch({ type: 'setPlaying', playing: false })}
          />
        ) : null}

        {loadState.phase === 'sample' && (
          <div className="state-card">
            <LoaderCircle className="spinner" />
            <strong>Loading the deer study</strong>
            <span>Decoding the renderer payload off the main thread…</span>
          </div>
        )}

        {loadState.phase === 'error' && (
          <div className="state-card state-card--error" role="alert">
            <AlertTriangle />
            <strong>{loadState.error?.title}</strong>
            <span>{loadState.error?.detail}</span>
            <div className="state-card__actions">
              {dataset && <button type="button" className="button" onClick={() => setLoadState({ phase: 'ready' })}>Keep current scene</button>}
              {!dataset && <button type="button" className="button" onClick={() => decoderRef.current && void loadSample(decoderRef.current)}>Retry sample</button>}
              <button type="button" className="button button--primary" onClick={() => fileInputRef.current?.click()}>Choose another .pt</button>
            </div>
          </div>
        )}

        {busy && (
          <div className="conversion-card" role="status" aria-live="polite">
            <div className="conversion-card__icon"><LoaderCircle className="spinner" /></div>
            <div className="conversion-card__copy">
              <strong>{loadState.status ?? 'Reading the PyTorch archive locally'}</strong>
              <span>
                {loadState.progress !== undefined
                  ? `${Math.round(loadState.progress * 100)}% complete · no file data leaves this browser`
                  : 'The selected file stays inside this browser.'}
              </span>
            </div>
            {loadState.progress !== undefined && (
              <div className="progress"><span style={{ width: `${loadState.progress * 100}%` }} /></div>
            )}
            <button type="button" className="button button--quiet" onClick={() => conversionRef.current?.abort()}>Cancel</button>
          </div>
        )}
        {dragActive && (
          <div className="drop-overlay">
            <div><FileUp size={30} /><strong>Drop predictions.pt</strong><span>Validate and open in this scene</span></div>
          </div>
        )}

        {dataset && (
          <aside className="control-rail" aria-label="Visualization controls">
            <div className="panel-heading"><SlidersHorizontal size={15} /><span>Appearance</span></div>
            <div className="control-group">
              <span className="control-label">Color</span>
              <div className="segment-grid">
                {(['rgb', 'dynamic', 'source', 'depth'] as ColorMode[]).map((mode) => (
                  <button
                    type="button"
                    key={mode}
                    className={settings.colorMode === mode ? 'active' : ''}
                    onClick={() => setSettings((value) => ({ ...value, colorMode: mode }))}
                  >
                    {mode === 'rgb' ? 'RGB' : mode === 'source' ? 'Views' : `${mode[0].toUpperCase()}${mode.slice(1)}`}
                  </button>
                ))}
              </div>
            </div>
            <label className="control-group">
              <span className="control-label"><span>Dynamic threshold</span><output>{settings.dynamicThreshold.toFixed(2)}</output></span>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={settings.dynamicThreshold}
                onChange={(event) => setSettings((value) => ({ ...value, dynamicThreshold: Number(event.target.value) }))}
              />
            </label>
            <label className="control-group">
              <span className="control-label"><span>Source view</span><output>{settings.selectedView < 0 ? 'All' : String(settings.selectedView + 1).padStart(2, '0')}</output></span>
              <select value={settings.selectedView} onChange={(event) => setSettings((value) => ({ ...value, selectedView: Number(event.target.value) }))}>
                <option value={-1}>All source views</option>
                {Array.from({ length: dataset.manifest.sourceViewCount }, (_, view) => (
                  <option key={view} value={view}>View {String(view + 1).padStart(2, '0')}</option>
                ))}
              </select>
            </label>
            <label className="control-group">
              <span className="control-label"><span>Point size</span><output>{settings.pointSize.toFixed(1)} px</output></span>
              <input
                type="range"
                min="0.7"
                max="6"
                step="0.1"
                value={settings.pointSize}
                onChange={(event) => setSettings((value) => ({ ...value, pointSize: Number(event.target.value) }))}
              />
            </label>
            <div className="control-divider" />
            <Toggle checked={settings.trails} label="Motion trails" onChange={() => setSettings((value) => ({ ...value, trails: !value.trails }))} />
            <Toggle checked={settings.grid} label="Ground grid" onChange={() => setSettings((value) => ({ ...value, grid: !value.grid }))} />
            <Toggle checked={settings.cameraFrusta} label="Camera frusta" onChange={() => setSettings((value) => ({ ...value, cameraFrusta: !value.cameraFrusta }))} />
            <button type="button" className="rail-action" onClick={() => setResetToken((token) => token + 1)}>
              <RotateCcw size={14} /> Reset camera <kbd>R</kbd>
            </button>
          </aside>
        )}

        {media?.kind === 'video' && (
          <div className="source-preview">
            <div className="source-preview__title"><Video size={13} /><span>Source clock</span></div>
            <video
              ref={(element) => { mediaRef.current = element }}
              src={media.url}
              muted={muted}
              playsInline
              preload="metadata"
              onLoadedMetadata={(event) => {
                const duration = event.currentTarget.duration
                setMedia((current) => current ? { ...current, duration, status: 'ready' } : null)
              }}
              onError={() => setMedia((current) => current ? { ...current, status: 'error', error: 'This video could not be decoded.' } : null)}
              onEnded={() => dispatch({ type: 'setPlaying', playing: false })}
            />
          </div>
        )}

        {media?.kind === 'audio' && (
          <audio
            ref={(element) => { mediaRef.current = element }}
            src={media.url}
            preload="metadata"
            onLoadedMetadata={(event) => {
              const duration = event.currentTarget.duration
              setMedia((current) => current ? { ...current, duration, status: 'ready' } : null)
            }}
            onError={() => setMedia((current) => current ? { ...current, status: 'error', error: 'This audio could not be decoded.' } : null)}
            onEnded={() => dispatch({ type: 'setPlaying', playing: false })}
          />
        )}

        {reducedPerformance && dataset && dataset.manifest.pointCount > 50_000 && (
          <div className="performance-note"><Gauge size={14} /><span>Reduced-performance device · choose 50k on your next import</span></div>
        )}
      </section>

      {diagnosticsOpen && dataset && (
        <aside className="diagnostics" aria-label="Dataset diagnostics">
          <div className="diagnostics__header">
            <div><Info size={16} /><strong>Dataset diagnostics</strong></div>
            <IconButton label="Close diagnostics" onClick={() => setDiagnosticsOpen(false)}><X size={17} /></IconButton>
          </div>
          <dl className="diagnostic-list">
            <div><dt>Schema</dt><dd>OMX4D v{dataset.manifest.schemaVersion}</dd></div>
            <div><dt>Frames</dt><dd>{dataset.manifest.frameCount}</dd></div>
            <div><dt>Points</dt><dd>{dataset.manifest.pointCount.toLocaleString()}</dd></div>
            <div><dt>Source views</dt><dd>{dataset.manifest.sourceViewCount}</dd></div>
            <div><dt>Native FPS</dt><dd>{dataset.manifest.fps}</dd></div>
            <div><dt>Duration</dt><dd>{dataset.manifest.durationSeconds.toFixed(3)} s</dd></div>
            <div><dt>Coordinates</dt><dd>{dataset.manifest.coordinateSystem}</dd></div>
            <div><dt>Units</dt><dd>{dataset.manifest.units}</dd></div>
          </dl>
          {warnings.length > 0 && (
            <div className="warning-list">
              <span>Conversion notes</span>
              {warnings.map((warning) => <p key={warning}><AlertTriangle size={13} />{warning}</p>)}
            </div>
          )}
          <p className="privacy-note">The worker reads a restricted, non-executing subset of the PyTorch archive format. The selected .pt file never leaves this browser.</p>
        </aside>
      )}

      <footer className="transport" aria-label="Playback controls">
        <div className="transport__main">
          <IconButton label="Restart (Home)" disabled={!dataset} onClick={() => seek(0)}><RotateCcw size={17} /></IconButton>
          <IconButton label="Previous frame (Left arrow)" disabled={!dataset} onClick={() => seek(player.frame - 1)}><ChevronLeft size={19} /></IconButton>
          <button
            type="button"
            className="play-button"
            aria-label={player.playing ? 'Pause' : 'Play'}
            title={`${player.playing ? 'Pause' : 'Play'} (Space)`}
            disabled={!dataset || busy}
            onClick={() => void togglePlayback()}
          >
            {player.playing ? <Pause size={20} fill="currentColor" /> : <Play size={20} fill="currentColor" />}
          </button>
          <IconButton label="Next frame (Right arrow)" disabled={!dataset} onClick={() => seek(player.frame + 1)}><ChevronRight size={19} /></IconButton>
          <IconButton label="Loop (L)" active={player.loop} disabled={!dataset} onClick={() => dispatch({ type: 'toggleLoop' })}><Repeat2 size={17} /></IconButton>
        </div>

        <div className="timeline-wrap">
          <div className="timeline-readout">
            <span>{formatTime(currentTime)}</span>
            <span className="timeline-frame">Frame {String(player.frame + 1).padStart(2, '0')} / {String(player.frameCount).padStart(2, '0')}</span>
            <span>{formatTime(duration)}</span>
          </div>
          <input
            className="timeline"
            type="range"
            aria-label="Timeline"
            min="0"
            max={Math.max(0, player.frameCount - 1)}
            step="1"
            value={player.frame}
            style={{ '--timeline-progress': `${timelineProgress}%` } as CSSProperties}
            disabled={!dataset}
            onChange={(event) => seek(Number(event.target.value))}
          />
        </div>

        <div className="transport__settings">
          <label className="compact-field" title="Inference frames per second">
            <span>FPS</span>
            <input type="number" min="1" max="120" step="1" value={player.fps} onChange={(event) => dispatch({ type: 'setFps', fps: Number(event.target.value) })} />
          </label>
          <label className="compact-field">
            <span>Speed</span>
            <select value={player.rate} onChange={(event) => dispatch({ type: 'setRate', rate: Number(event.target.value) })}>
              {PLAYBACK_RATES.map((rate) => <option key={rate} value={rate}>{rate}×</option>)}
            </select>
          </label>
        </div>

        <div className="media-control">
          {media ? (
            <>
              <button type="button" className="media-name" title={media.name} onClick={() => mediaInputRef.current?.click()}>
                {media.kind === 'video' ? <Video size={15} /> : <AudioLines size={15} />}
                <span>{media.name}</span>
              </button>
              <IconButton label={muted ? 'Unmute' : 'Mute'} onClick={() => setMuted((value) => !value)}>
                {muted ? <VolumeX size={17} /> : <Volume2 size={17} />}
              </IconButton>
              <input className="volume" aria-label="Volume" type="range" min="0" max="1" step="0.05" value={volume} onChange={(event) => setVolume(Number(event.target.value))} />
              <label className="offset-field" title="Media clock minus 3D clock">
                <span>Offset</span>
                <input type="number" step="0.01" value={syncOffset} onChange={(event) => setSyncOffset(Number(event.target.value) || 0)} />
                <span>s</span>
              </label>
              <IconButton label="Remove source media" onClick={removeMedia}><X size={16} /></IconButton>
            </>
          ) : (
            <button type="button" className="button button--quiet" onClick={() => mediaInputRef.current?.click()}>
              <AudioLines size={15} /> Add source media
            </button>
          )}
        </div>
      </footer>

      {(durationMismatch || media?.status === 'error') && (
        <div className="media-warning" role="status">
          <AlertTriangle size={15} />
          <span>{media?.error ?? 'Source media duration differs from the 3D sequence; adjust the sync offset if needed.'}</span>
        </div>
      )}
    </main>
  )
}
