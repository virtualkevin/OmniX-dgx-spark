import { OrbitControls } from '@react-three/drei'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { Component, useEffect, useLayoutEffect, useMemo, useRef, type ElementRef, type ReactNode, type RefObject } from 'react'
import * as THREE from 'three'
import type { PlayerState } from '../lib/playback'
import { durationSeconds, frameAtMediaTime, frameAtTime } from '../lib/playback'
import type { RenderSettings, ViewerDataset } from '../lib/dataset'

interface ViewerSceneProps {
  dataset: ViewerDataset
  frame: number
  player: PlayerState
  settings: RenderSettings
  resetToken: number
  seekToken: number
  mediaRef: RefObject<HTMLMediaElement | null>
  syncOffset: number
  onFrame: (frame: number) => void
  onPlaybackEnd: () => void
}

const vertexShader = /* glsl */ `
  attribute vec3 color;
  attribute float dynamicScore;
  attribute float sourceView;
  uniform float uPointSize;
  uniform float uPixelRatio;
  uniform float uThreshold;
  uniform float uSelectedView;
  varying vec3 vColor;
  varying float vDynamic;
  varying float vSource;
  varying float vDepth;
  varying float vVisible;

  void main() {
    vec4 viewPosition = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * viewPosition;
    gl_PointSize = clamp(uPointSize * uPixelRatio, 1.0, 12.0);
    vColor = color;
    vDynamic = dynamicScore;
    vSource = sourceView;
    vDepth = max(0.0, -viewPosition.z);
    vVisible = step(uThreshold, dynamicScore) * step(abs(sourceView - uSelectedView), 0.49);
    if (uSelectedView < -0.5) vVisible = step(uThreshold, dynamicScore);
  }
`

const fragmentShader = /* glsl */ `
  precision highp float;
  uniform int uColorMode;
  varying vec3 vColor;
  varying float vDynamic;
  varying float vSource;
  varying float vDepth;
  varying float vVisible;

  vec3 sourcePalette(float value) {
    return 0.30 + 0.70 * fract(sin(vec3(value * 12.9898, value * 78.233, value * 39.425) + vec3(0.2, 1.7, 3.1)) * 43758.5453);
  }

  vec3 turbo(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 c0 = vec3(0.10, 0.06, 0.40);
    vec3 c1 = vec3(0.08, 0.70, 0.95);
    vec3 c2 = vec3(0.96, 0.82, 0.20);
    vec3 c3 = vec3(0.90, 0.12, 0.12);
    return t < 0.5 ? mix(c0, c1, t * 2.0) : mix(c2, c3, (t - 0.5) * 2.0);
  }

  void main() {
    if (vVisible < 0.5) discard;
    vec2 point = gl_PointCoord - vec2(0.5);
    float radius = length(point);
    if (radius > 0.5) discard;

    vec3 outputColor = vColor;
    if (uColorMode == 1) outputColor = turbo(vDynamic);
    if (uColorMode == 2) outputColor = sourcePalette(vSource + 1.0);
    if (uColorMode == 3) outputColor = turbo(clamp(vDepth / 35.0, 0.0, 1.0));
    float edge = 1.0 - smoothstep(0.34, 0.5, radius);
    gl_FragColor = vec4(outputColor * (0.78 + 0.22 * edge), edge);
  }
`

function colorModeIndex(mode: RenderSettings['colorMode']): number {
  return { rgb: 0, dynamic: 1, source: 2, depth: 3 }[mode]
}

function PointCloud({ dataset, frame, settings }: Pick<ViewerSceneProps, 'dataset' | 'frame' | 'settings'>) {
  const positionAttribute = useMemo(() => {
    const array = new Float32Array(dataset.manifest.pointCount * 3)
    const attribute = new THREE.BufferAttribute(array, 3)
    attribute.setUsage(THREE.DynamicDrawUsage)
    return attribute
  }, [dataset])

  const geometry = useMemo(() => {
    const result = new THREE.BufferGeometry()
    result.setAttribute('position', positionAttribute)
    result.setAttribute('color', new THREE.BufferAttribute(dataset.colors, 3, true))
    result.setAttribute('dynamicScore', new THREE.BufferAttribute(dataset.dynamicScore, 1))
    result.setAttribute('sourceView', new THREE.BufferAttribute(dataset.sourceView, 1))
    result.setDrawRange(0, dataset.manifest.pointCount)
    return result
  }, [dataset, positionAttribute])

  const material = useMemo(
    () =>
      new THREE.ShaderMaterial({
        uniforms: {
          uPointSize: { value: settings.pointSize },
          uPixelRatio: { value: 1 },
          uThreshold: { value: settings.dynamicThreshold },
          uSelectedView: { value: settings.selectedView },
          uColorMode: { value: colorModeIndex(settings.colorMode) },
        },
        vertexShader,
        fragmentShader,
        transparent: true,
        depthWrite: true,
      }),
    [dataset],
  )

  const { gl } = useThree()

  useLayoutEffect(() => {
    const start = frame * dataset.manifest.pointCount * 3
    const target = positionAttribute.array as Float32Array
    target.set(dataset.positions.subarray(start, start + target.length))
    positionAttribute.needsUpdate = true
  }, [dataset, frame, positionAttribute])

  useEffect(() => {
    material.uniforms.uPointSize.value = settings.pointSize
    material.uniforms.uThreshold.value = settings.dynamicThreshold
    material.uniforms.uSelectedView.value = settings.selectedView
    material.uniforms.uColorMode.value = colorModeIndex(settings.colorMode)
    material.uniforms.uPixelRatio.value = Math.min(gl.getPixelRatio(), 1.75)
  }, [gl, material, settings])

  useEffect(
    () => () => {
      geometry.dispose()
      material.dispose()
    },
    [geometry, material],
  )

  return (
    <points geometry={geometry} material={material} frustumCulled={false}>
      <primitive object={geometry} attach="geometry" />
      <primitive object={material} attach="material" />
    </points>
  )
}

function MotionTrails({ dataset, frame, settings }: Pick<ViewerSceneProps, 'dataset' | 'frame' | 'settings'>) {
  const maxTrails = Math.min(20_000, dataset.manifest.pointCount)
  const geometry = useMemo(() => {
    const result = new THREE.BufferGeometry()
    const attribute = new THREE.BufferAttribute(new Float32Array(maxTrails * 2 * 3), 3)
    attribute.setUsage(THREE.DynamicDrawUsage)
    result.setAttribute('position', attribute)
    result.setDrawRange(0, 0)
    return result
  }, [dataset, maxTrails])

  useLayoutEffect(() => {
    if (frame < 1 || !settings.trails) {
      geometry.setDrawRange(0, 0)
      return
    }
    const output = geometry.getAttribute('position') as THREE.BufferAttribute
    const values = output.array as Float32Array
    const stride = Math.max(1, Math.floor(dataset.manifest.pointCount / maxTrails))
    const currentOffset = frame * dataset.manifest.pointCount * 3
    const previousOffset = (frame - 1) * dataset.manifest.pointCount * 3
    let cursor = 0
    for (let point = 0; point < dataset.manifest.pointCount && cursor < maxTrails; point += stride) {
      if (dataset.dynamicScore[point] < Math.max(0.05, settings.dynamicThreshold)) continue
      if (settings.selectedView >= 0 && dataset.sourceView[point] !== settings.selectedView) continue
      const sourceIndex = point * 3
      const targetIndex = cursor * 6
      values[targetIndex] = dataset.positions[previousOffset + sourceIndex]
      values[targetIndex + 1] = dataset.positions[previousOffset + sourceIndex + 1]
      values[targetIndex + 2] = dataset.positions[previousOffset + sourceIndex + 2]
      values[targetIndex + 3] = dataset.positions[currentOffset + sourceIndex]
      values[targetIndex + 4] = dataset.positions[currentOffset + sourceIndex + 1]
      values[targetIndex + 5] = dataset.positions[currentOffset + sourceIndex + 2]
      cursor += 1
    }
    geometry.setDrawRange(0, cursor * 2)
    output.needsUpdate = true
  }, [dataset, frame, geometry, maxTrails, settings])

  useEffect(() => () => geometry.dispose(), [geometry])

  return (
    <lineSegments geometry={geometry} frustumCulled={false}>
      <lineBasicMaterial color="#77d9ff" transparent opacity={0.22} depthWrite={false} />
    </lineSegments>
  )
}

function transformPosePoint(pose: Float32Array, view: number, point: THREE.Vector3): THREE.Vector3 {
  const offset = view * 16
  return new THREE.Vector3(
    pose[offset] * point.x + pose[offset + 1] * point.y + pose[offset + 2] * point.z + pose[offset + 3],
    pose[offset + 4] * point.x + pose[offset + 5] * point.y + pose[offset + 6] * point.z + pose[offset + 7],
    pose[offset + 8] * point.x + pose[offset + 9] * point.y + pose[offset + 10] * point.z + pose[offset + 11],
  )
}

function CameraFrusta({ dataset }: { dataset: ViewerDataset }) {
  const geometry = useMemo(() => {
    const segments: number[] = []
    const depth = -0.16
    const corners = [
      new THREE.Vector3(-0.12, -0.075, depth),
      new THREE.Vector3(0.12, -0.075, depth),
      new THREE.Vector3(0.12, 0.075, depth),
      new THREE.Vector3(-0.12, 0.075, depth),
    ]
    const edgePairs = [[0, 1], [1, 2], [2, 3], [3, 0]]
    for (let view = 0; view < dataset.manifest.sourceViewCount; view += 1) {
      const origin = transformPosePoint(dataset.cameraPose, view, new THREE.Vector3())
      const worldCorners = corners.map((corner) => transformPosePoint(dataset.cameraPose, view, corner))
      for (const corner of worldCorners) segments.push(...origin.toArray(), ...corner.toArray())
      for (const [a, b] of edgePairs) segments.push(...worldCorners[a].toArray(), ...worldCorners[b].toArray())
    }
    const result = new THREE.BufferGeometry()
    result.setAttribute('position', new THREE.Float32BufferAttribute(segments, 3))
    return result
  }, [dataset])

  useEffect(() => () => geometry.dispose(), [geometry])

  return (
    <lineSegments geometry={geometry}>
      <lineBasicMaterial color="#ffbd5a" transparent opacity={0.58} depthWrite={false} />
    </lineSegments>
  )
}

function getBounds(dataset: ViewerDataset) {
  const declared = dataset.manifest.bounds
  let rawMin: [number, number, number]
  let rawMax: [number, number, number]
  if (declared) {
    rawMin = declared.min
    rawMax = declared.max
  } else {
    rawMin = [Infinity, Infinity, Infinity]
    rawMax = [-Infinity, -Infinity, -Infinity]
    const firstFrame = dataset.positions.subarray(0, dataset.manifest.pointCount * 3)
    for (let index = 0; index < firstFrame.length; index += 3) {
      rawMin[0] = Math.min(rawMin[0], firstFrame[index])
      rawMin[1] = Math.min(rawMin[1], firstFrame[index + 1])
      rawMin[2] = Math.min(rawMin[2], firstFrame[index + 2])
      rawMax[0] = Math.max(rawMax[0], firstFrame[index])
      rawMax[1] = Math.max(rawMax[1], firstFrame[index + 1])
      rawMax[2] = Math.max(rawMax[2], firstFrame[index + 2])
    }
  }
  const min = new THREE.Vector3(...rawMin)
  const max = new THREE.Vector3(...rawMax)
  const center = min.clone().add(max).multiplyScalar(0.5)
  const radius = Math.max(0.5, min.distanceTo(max) * 0.5)
  return { min, max, center, radius }
}

function CameraHome({ dataset, resetToken }: Pick<ViewerSceneProps, 'dataset' | 'resetToken'>) {
  const controlsRef = useRef<ElementRef<typeof OrbitControls>>(null)
  const { camera } = useThree()

  useEffect(() => {
    const { radius } = getBounds(dataset)
    const values = dataset.cameraPose
    const pose = new THREE.Matrix4()
    pose.set(
      values[0], values[1], values[2], values[3],
      values[4], values[5], values[6], values[7],
      values[8], values[9], values[10], values[11],
      values[12], values[13], values[14], values[15],
    )
    pose.decompose(camera.position, camera.quaternion, camera.scale)
    const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion)
    camera.position.addScaledVector(forward, -radius * 0.12)
    camera.near = Math.max(0.001, radius / 1_000)
    camera.far = Math.max(1_000, radius * 100)
    if (camera instanceof THREE.PerspectiveCamera) {
      camera.fov = 58
    }
    camera.updateProjectionMatrix()
    const target = camera.position.clone().addScaledVector(forward, Math.max(1, radius * 0.6))
    controlsRef.current?.target.copy(target)
    controlsRef.current?.update()
  }, [camera, dataset, resetToken])

  return <OrbitControls ref={controlsRef} makeDefault enableDamping dampingFactor={0.08} />
}

function PlaybackDriver({
  player,
  seekToken,
  mediaRef,
  syncOffset,
  onFrame,
  onPlaybackEnd,
}: Pick<ViewerSceneProps, 'player' | 'seekToken' | 'mediaRef' | 'syncOffset' | 'onFrame' | 'onPlaybackEnd'>) {
  const playhead = useRef(player.frame / player.fps)
  const lastFrame = useRef(player.frame)
  const callbacks = useRef({ onFrame, onPlaybackEnd })
  callbacks.current = { onFrame, onPlaybackEnd }

  useEffect(() => {
    playhead.current = player.frame / player.fps
    lastFrame.current = player.frame
  }, [player.fps, seekToken])

  useFrame((_, delta) => {
    if (!player.playing) return
    const media = mediaRef.current
    let nextFrame: number

    if (media) {
      nextFrame = frameAtMediaTime(media.currentTime, syncOffset, player.fps, player.frameCount)
      const datasetEnd = durationSeconds(player.frameCount, player.fps) + syncOffset
      if (media.currentTime >= datasetEnd) {
        if (player.loop) {
          media.currentTime = Math.max(0, syncOffset)
          void media.play().catch(() => callbacks.current.onPlaybackEnd())
          nextFrame = 0
        } else {
          media.pause()
          nextFrame = player.frameCount - 1
          callbacks.current.onPlaybackEnd()
        }
      }
    } else {
      playhead.current += Math.min(delta, 0.1) * player.rate
      const duration = durationSeconds(player.frameCount, player.fps)
      if (playhead.current >= duration) {
        if (player.loop) playhead.current %= duration
        else {
          playhead.current = Math.max(0, (player.frameCount - 1) / player.fps)
          callbacks.current.onPlaybackEnd()
        }
      }
      nextFrame = frameAtTime(playhead.current, player.fps, player.frameCount)
    }

    if (nextFrame !== lastFrame.current) {
      lastFrame.current = nextFrame
      callbacks.current.onFrame(nextFrame)
    }
  })

  return null
}

function SceneContent(props: ViewerSceneProps) {
  const { min, radius } = getBounds(props.dataset)
  return (
    <>
      <color attach="background" args={['#080b0f']} />
      <fog attach="fog" args={['#080b0f', radius * 4, radius * 15]} />
      <PointCloud dataset={props.dataset} frame={props.frame} settings={props.settings} />
      {props.settings.trails && <MotionTrails dataset={props.dataset} frame={props.frame} settings={props.settings} />}
      {props.settings.cameraFrusta && <CameraFrusta dataset={props.dataset} />}
      {props.settings.grid && (
        <gridHelper args={[Math.max(10, radius * 8), 20, '#294653', '#17262d']} position-y={min.y} />
      )}
      <CameraHome dataset={props.dataset} resetToken={props.resetToken} />
      <PlaybackDriver
        player={props.player}
        seekToken={props.seekToken}
        mediaRef={props.mediaRef}
        syncOffset={props.syncOffset}
        onFrame={props.onFrame}
        onPlaybackEnd={props.onPlaybackEnd}
      />
    </>
  )
}

interface ErrorBoundaryState { failed: boolean }

export class WebGlErrorBoundary extends Component<{ children: ReactNode; fallback: ReactNode }, ErrorBoundaryState> {
  state: ErrorBoundaryState = { failed: false }

  static getDerivedStateFromError(): ErrorBoundaryState {
    return { failed: true }
  }

  render() {
    return this.state.failed ? this.props.fallback : this.props.children
  }
}

export function ViewerScene(props: ViewerSceneProps) {
  return (
    <WebGlErrorBoundary
      fallback={
        <div className="state-card state-card--error">
          <strong>WebGL could not start</strong>
          <span>Enable hardware acceleration or try a current desktop browser.</span>
        </div>
      }
    >
      <Canvas
        dpr={[1, 1.75]}
        camera={{ fov: 48, near: 0.01, far: 10_000 }}
        gl={{ antialias: true, alpha: false, powerPreference: 'high-performance' }}
      >
        <SceneContent {...props} />
      </Canvas>
    </WebGlErrorBoundary>
  )
}
