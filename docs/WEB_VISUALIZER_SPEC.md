# OmniX real-time web visualizer specification

Status: implemented and release-verified
Target: local-first React Three Fiber application, backed by a restricted PyTorch ingestion service
Package manager: pnpm

## 1. Product objective

Build a desktop-first web application that turns OmniX inference output into an interactive, real-time 4D point-cloud player. A user must be able to orbit the scene while it plays, pause, scrub, change speed, inspect motion, and load another `predictions.pt` produced by this repository.

The first screen must be useful without setup: it loads a compact, baked version of the validated deer result. "Done" also requires the same UI to accept a user-selected `.pt` file; the baked file is not a substitute for upload support.

An optional original video or audio file may be selected locally. When present, its `HTMLMediaElement.currentTime` is the master playback clock, so its existing audio stays synchronized with the 3D result. Without media, a monotonic internal clock drives playback.

## 2. Facts and constraints discovered in the repository

The current inference writer saves a Python dictionary with four CPU float32 tensors:

| Key | Shape for the deer fixture | Meaning |
| --- | --- | --- |
| `trajectory` | `[16, 16, 280, 504, 3]` | World-space position for `[source view, target time, y, x, xyz]` |
| `camera_pose` | `[16, 3, 4]` | OpenCV camera-to-world pose per source view |
| `intrinsics` | `[16, 3, 3]` | OpenCV intrinsic matrix per source view |
| `pts3d_dynamic_score` | `[16, 280, 504]` | Time-invariant dynamic probability per source pixel |

The validated deer `.pt` is 442,556,165 bytes. Its trajectory alone is 433,520,640 bytes. It has no format version, timestamps, FPS, source grouping, RGB, units, or audio. Source colors are saved separately under `gt_images/cam_NN.jpg`; existing renders assume 15 FPS.

PyTorch documents that ordinary `torch.save` files are ZIP64 archives containing pickle metadata and separate tensor storages. `.pt` is only a naming convention, not a universal schema. PyTorch also warns that even `torch.load(..., weights_only=True)` does not eliminate denial-of-service or memory-corruption risk. Therefore the browser will not deserialize arbitrary pickle, and the app will promise compatibility only with the exact plain-tensor OmniX output contract above.

## 3. Architecture

```text
selected predictions.pt                 baked deer fixture
          |                                      |
          v                                      v
restricted local Python API              static .omx4d asset
  - stream to a temporary file                    |
  - weights_only=True                              |
  - strict schema/resource checks                  |
  - deterministic sampling                         |
          |                                        |
          +---------------+------------------------+
                          v
                versioned .omx4d payload
                          |
                          v
                browser Web Worker decode
                          |
                          v
             transferable typed-array buffers
                          |
                          v
        one R3F/Three.js Points draw call + timeline
```

### 3.1 Frontend

- `viewer/` is a Vite + React + TypeScript application managed exclusively with pnpm and a committed `pnpm-lock.yaml`.
- Use React Three Fiber on Three.js WebGL 2. WebGPU is not an MVP dependency because R3F still describes that path as incomplete.
- Render sampled data with one `THREE.Points` and one `BufferGeometry`. Keep the geometry/material mounted, use `DynamicDrawUsage`, copy a new frame into the existing position buffer only when the frame index changes, and set `needsUpdate`.
- Decode the renderer payload in a Web Worker and transfer its `ArrayBuffer`s to the main thread. Never parse a 400+ MiB `.pt` or a large JSON trajectory on the UI thread.
- Use OrbitControls and a perspective camera with sensible reset/home framing computed from sampled bounds.

### 3.2 Ingestion API

- A small Python service receives multipart upload to `POST /api/convert`; it never executes model inference.
- Stream uploads to a bounded temporary file instead of reading them into request memory.
- Load with a current patched PyTorch, `map_location="cpu"`, `weights_only=True`, and never retry with `weights_only=False`.
- Validate exact key allowlist, tensor type, float32 dtype, contiguity after normalization, rank/shape relationships, finite values, positive dimensions, camera matrices, and configured element/byte limits before conversion.
- Reject unknown keys, objects/classes, sparse/quantized tensors, unsupported devices/dtypes, non-finite values, dimension mismatches, zip bombs, oversized requests, and conversion timeouts with actionable 4xx errors.
- The service is local-first and documented to run without credentials or outbound network access. Temporary files are deleted on success, error, and cancellation.

### 3.3 Renderer-native `.omx4d` contract

Use a compact versioned binary envelope shared by baked and uploaded datasets:

1. fixed magic and format version;
2. little-endian JSON-header byte length;
3. UTF-8 manifest;
4. aligned raw typed-array sections.

Manifest fields:

- `schemaVersion`, `name`, `fps`, `frameCount`, `durationSeconds`;
- `sourceViewCount`, `pointCount`, `coordinateSystem`, `units`, `primitive`;
- bounds and deterministic sampling metadata;
- byte offset, length, dtype, and shape for every attribute;
- optional warnings (for example, missing source RGB or assumed FPS).

Required sections:

- `positions`: float32 `[frameCount, pointCount, 3]`;
- `colors`: uint8 `[pointCount, 3]`;
- `dynamicScore`: float32 `[pointCount]`;
- `sourceView`: uint16 `[pointCount]`;
- `cameraPose`: float32 `[sourceViewCount, 4, 4]`;
- `intrinsics`: float32 `[sourceViewCount, 3, 3]`.

Sampling must keep a stable point identity across time, discard points non-finite at any frame, be deterministic for identical input/options, and cap the default at 100,000 points. It should combine spatially even coverage with a reserved dynamic-point budget so small moving subjects remain visible. The upload UI exposes a quality choice (50k / 100k / 200k) before conversion.

If companion RGB images are not supplied, derive a stable, readable palette from source view and dynamic score. A `.pt`-only upload must still render correctly. The baked fixture may use its companion `gt_images` for RGB.

## 4. Playback and interaction

The timeline duration is `frameCount / fps`; default FPS is 15 because that is what the repository's current renderer uses. The user can override FPS because the `.pt` carries no timing metadata.

Controls:

- play/pause, restart, previous/next frame;
- draggable timeline with current frame and time;
- loop toggle, 0.25x / 0.5x / 1x / 1.5x / 2x rate, FPS override;
- point size and point-budget display;
- dynamic-score threshold and color mode (`RGB`, `dynamic`, `source view`, `depth`);
- all views versus one source view;
- short motion trails toggle;
- grid, camera-frustum, and reset-view toggles;
- drag/drop and file-picker paths for replacing the `.pt`;
- optional video/audio picker, mute/volume, sync offset, and a duration-mismatch warning.

Playback rules:

- With source media, derive the 3D frame from `floor((media.currentTime - syncOffset) * fps)`; play, pause, seek, rate, loop, and volume act on the media element. The browser's user-gesture requirement is satisfied by the explicit Play control.
- Without media, advance from a monotonic clock in `useFrame`; do not call React state setters every render tick.
- Scrubbing is deterministic and updates the 3D frame immediately while paused.
- Do not interpolate in v1. Nearest discrete inference frames are truthful and avoid inventing geometry.

## 5. Visual design and states

Use a restrained dark visualization workspace: large canvas, compact translucent control rail, high-contrast timeline, and a diagnostics drawer. The 3D content—not dashboard chrome—is primary.

Required states:

- baked sample loading and ready;
- upload drag-over, progress/converting, cancellation, success;
- schema/security/resource validation error with recovery action;
- WebGL unavailable and reduced-performance warning;
- no audio, media ready, media duration mismatch, and blocked/failed media decode.

All controls need keyboard focus styles, labels/tooltips, and usable hit targets. Core playback shortcuts: Space, Left/Right, Home, `L`, and `R` for reset camera. Respect reduced-motion preferences for UI animation; this does not disable explicitly requested point-cloud playback.

## 6. Performance budgets

- First interactive baked sample: target under 3 seconds on a modern desktop after local assets are available.
- Default payload: at most 100k points and roughly 20 MiB for 16 frames; no 400+ MiB browser allocation.
- Steady playback: target 60 rendering FPS on the DGX Spark browser at 100k points, never below 30 FPS in the acceptance fixture.
- One point-cloud draw call; avoid per-point React elements and per-frame object allocation.
- Only upload a new GPU position buffer when the discrete inference frame changes.
- Cap device pixel ratio and offer 50k fallback quality when frame-time degradation is detected or chosen by the user.

## 7. Security and privacy

- Files stay local to the user's machine/service and are not sent to a third party.
- Treat uploaded `.pt` as hostile. Extension and MIME are hints only; validate container and contents.
- Default maximum upload: 1 GiB. Default supported logical tensor maximums: 64 source views, 600 time frames, 16 million source pixels, and a configurable total-byte ceiling. Enforce limits before materializing derived copies where possible.
- Run conversion with CPU, address-space, file-size, open-file, and wall-clock limits when launched through the provided container entry point.
- Never include `weights_only=False`, generic pickle execution, uploaded filenames in shell commands, or unsanitized error tracebacks in API responses.

## 8. Testing and acceptance criteria

### Automated

- Python unit tests cover valid conversion, deterministic sampling, malformed/wrong-key tensors, dtype/shape mismatch, NaN/Inf, limit rejection, safe errors, and binary manifest/offset integrity.
- TypeScript unit tests cover payload parsing, frame/time clamping, media offset math, reducer state, and user-facing API error mapping.
- Browser tests cover baked startup, play/pause, seek, speed, loop, camera reset, filters, source-view switch, responsive layout, and keyboard controls.
- An upload browser test selects a small valid `.pt`, confirms conversion, and verifies its manifest is shown. Negative upload tests use malformed and schema-invalid fixtures.

### Manual / release gate

- The baked asset is converted from the actual validated deer output and visibly reconstructs a coherent moving deer/street scene.
- A user can select the real 422 MiB `predictions.pt`; conversion completes without locking the browser, and the result plays in real time.
- Baked and uploaded conversions made with the same options have matching manifests and sampled hashes.
- Play/pause/seek/loop/rate always select the expected discrete frame.
- Optional original video or audio remains within one inference frame of the 3D timeline after seeking and during a five-minute synthetic clock test; duration mismatch is disclosed.
- Replacing a dataset/media file releases object URLs, worker buffers, server temp files, and old Three.js GPU resources.
- `pnpm lint`, `pnpm test`, `pnpm build`, Python tests, and the browser smoke suite pass.

## 9. Delivery plan

1. Define and test the strict Python schema validator, deterministic sampler, and `.omx4d` encoder.
2. Generate a compact real deer fixture and expose the upload conversion endpoint.
3. Scaffold the pnpm/Vite/React/R3F application and Web Worker decoder.
4. Implement point rendering, camera framing, playback state, timeline, filters, and trails.
5. Add optional original-video/audio clocking and sync controls.
6. Add loading/error/accessibility states, automated tests, container/dev scripts, and documentation.
7. Run the release gate with both the baked sample and the full real `.pt` upload.

## 10. Deliberate non-goals for v1

- Running OmniX inference in the browser;
- accepting arbitrary `.pt` schemas or falling back to unsafe unpickling;
- photorealistic Gaussian-splat rasterization (the current output is point trajectories, not Gaussian parameters);
- geometry interpolation between inferred time steps;
- extracting or transcoding audio on the server;
- multi-user hosting, authentication, or persistent cloud storage.

## 11. Research references

- [PyTorch serialization semantics](https://docs.pytorch.org/docs/main/notes/serialization.html)
- [PyTorch `torch.load` security guidance](https://docs.pytorch.org/docs/stable/generated/torch.load.html)
- [React Three Fiber installation and version pairing](https://r3f.docs.pmnd.rs/getting-started/installation)
- [React Three Fiber render-loop hooks](https://r3f.docs.pmnd.rs/api/hooks)
- [React Three Fiber performance guidance](https://r3f.docs.pmnd.rs/advanced/pitfalls)
- [Three.js `BufferGeometry`](https://threejs.org/docs/pages/BufferGeometry.html)
- [Three.js `BufferAttribute`](https://threejs.org/docs/pages/BufferAttribute.html)
- [Three.js `Points`](https://threejs.org/docs/pages/Points.html)
- [MDN File API](https://developer.mozilla.org/en-US/docs/Web/API/File_API)
- [MDN transferable objects](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Transferable_objects)
- [MDN media `currentTime`](https://developer.mozilla.org/en-US/docs/Web/API/HTMLMediaElement/currentTime)
- [MDN media autoplay guidance](https://developer.mozilla.org/en-US/docs/Web/Media/Guides/Autoplay)


## 12. Release verification record

Release gate completed against the real repository artifact on 2026-07-16:

- Input: 442,556,165-byte `predictions.pt`, SHA-256 `39b2e72dafaa56f634d4185d44372e63074d28ef36537a1bac11600ddcc76743`.
- Baked renderer fixture: 20,103,008 bytes, SHA-256 `d04f33095a1fc5adb523652746aa3f6c3ed6ad64cfb3035bd8757096d23afdaf`.
- Baked RGB and browser-uploaded conversions matched the deterministic identity hash `e4ef50be1678d24592fc63867cc4b3341882afe77d839273eebcf60622035ff3` at 100,000 points, 16 frames, and 15 FPS.
- Python converter suite: 17 tests passed; frontend suite: 12 tests passed.
- Production Chromium suite: 6 tests passed, including the full real upload, WebGL playback, transport/keyboard controls, responsive layout, synchronized WAV clocking, and sanitized malformed-upload recovery.
- The hardened Compose stack built, became healthy, exposed only Nginx on `127.0.0.1:4173`, and served the API exclusively through its proxy.

The browser gate used Chromium with SwiftShader WebGL so it exercised a real graphics context in the headless container rather than mocking Three.js or the canvas.
