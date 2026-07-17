# OmniX real-time web visualizer specification

Status: browser-only architecture implemented and release-verified
Target: static, local-first React Three Fiber application with in-browser `.pt` ingestion
Package manager: pnpm

## 1. Product objective

Build a desktop-first web application that turns OmniX inference output into an
interactive, real-time 4D point-cloud player. A user must be able to orbit the
scene while it plays, pause, scrub, change speed, inspect motion, and load a
`predictions.pt` produced by this repository.

The first screen loads a compact baked deer result. Completion also requires the
same UI to accept a user-selected raw `.pt` without an application server,
Python, PyTorch, or a file upload. Parsing, validation, deterministic sampling,
and renderer-array construction happen in the visiting browser.

An optional original video or audio file may be selected locally. When present,
its `HTMLMediaElement.currentTime` is the master playback clock, so its existing
audio stays synchronized with the 3D result. Without media, a monotonic internal
clock drives playback.

## 2. Repository facts and compatibility boundary

The inference writer saves a dictionary with four CPU float32 tensors:

| Key | Deer shape | Meaning |
| --- | --- | --- |
| `trajectory` | `[16, 16, 280, 504, 3]` | World position for `[source view, target time, y, x, xyz]` |
| `camera_pose` | `[16, 3, 4]` | OpenCV camera-to-world pose per source view |
| `intrinsics` | `[16, 3, 3]` | OpenCV intrinsic matrix per source view |
| `pts3d_dynamic_score` | `[16, 280, 504]` | Time-invariant dynamic probability per source pixel |

The validated deer `.pt` is 442,556,165 bytes. Its trajectory storage is
433,520,640 bytes. It has no timestamps, FPS, RGB, world-unit, or audio
metadata. Source colors are stored separately under `gt_images/cam_NN.jpg`, and
existing repository renders assume 15 FPS.

The artifact is a `torch.save` ZIP archive containing a 443-byte protocol-2
metadata pickle and four uncompressed little-endian float32 storage entries.
`.pt` is only a filename convention; model checkpoints and arbitrary PyTorch
objects are outside the product contract.

The browser supports this repository's exact plain-tensor grammar. It does not
provide a general unpickler and never imports or invokes a pickle global or
reduction. Tensor reconstruction tokens are inert metadata that must match the
allowlist before any large allocation occurs.

## 3. Architecture

```text
selected predictions.pt                         baked deer.omx4d
          |                                             |
          v                                             v
browser Web Worker                              OMX4D envelope parser
  - bounded ZIP directory parser                         |
  - restricted pickle metadata grammar                   |
  - tensor/schema/resource validation                    |
  - chunked File.slice() storage reads                    |
  - deterministic stable-identity sampling               |
          |                                             |
          +----------------------+----------------------+
                                 v
                       ViewerDataset typed arrays
                                 |
                      transferable ArrayBuffers
                                 |
                                 v
              one R3F/Three.js Points draw call + timeline
```

The production deployment is static HTML, JavaScript, CSS, worker code, and the
baked sample. Nginx is an optional static file host, not an application backend.

### 3.1 Frontend and rendering

- `viewer/` is Vite, React, and TypeScript, managed exclusively with pnpm and a
  committed `pnpm-lock.yaml`.
- Use React Three Fiber on Three.js WebGL. WebGPU is not a requirement.
- Render sampled data with one `THREE.Points` and one `BufferGeometry`. Keep the
  geometry/material mounted, use `DynamicDrawUsage`, copy a frame into the
  existing position buffer only when the discrete frame changes, and set
  `needsUpdate`.
- Keep archive parsing and all O(point-count) work in a module Web Worker.
  Transfer final typed-array buffers to the main thread instead of cloning them.
- Use OrbitControls and a perspective camera with reset/home framing calculated
  from sampled bounds.

### 3.2 Browser-side `.pt` ingestion

- Pass the selected `File` to the worker by structured clone. Do not call
  `arrayBuffer()` on the complete 400+ MiB archive and do not send it over HTTP.
- Read the bounded ZIP central directory and local headers first. Reject
  malformed, multi-disk, encrypted, compressed, duplicate, unsafe-path,
  overlapping, unsupported-version, and oversized entries.
- Require one archive root, `data.pkl`, `version`, `byteorder`, and the tensor
  storage entries referenced by the metadata. Only little-endian storage is in
  scope.
- Parse only the protocol/opcodes needed by the exact tensor dictionary. Permit
  the inert symbols `torch._utils._rebuild_tensor_v2`, `torch.FloatStorage`, and
  an empty `collections.OrderedDict` in their expected positions; reject every
  other global, reduction shape, object, key, device, dtype, layout, or trailing
  opcode.
- Require exactly the four schema keys, positive compatible dimensions,
  contiguous strides, zero storage offsets, exact float32 storage lengths,
  finite tensor values, scores in `[0, 1]`, positive focal lengths, homogeneous
  intrinsic rows, and approximately right-handed orthonormal camera rotations.
- Enforce file, archive-entry, source-view, frame, source-pixel, tensor-byte,
  sampled-output, and point-budget ceilings before derived allocations.
- Use `File.slice()` to read bounded storage ranges. Frame-zero points and
  dynamic scores drive selection; trajectory chunks are then validated and
  gathered directly into sampled output without materializing the full tensor.
- Report validation/sampling progress, checkpoint cancellation between chunks,
  preserve the current scene after recoverable errors, and release worker
  buffers when a dataset is replaced.

### 3.3 Stable point selection

Select identities once and gather those same identities at every target frame.
The default is 100,000 points, with 50k and 200k choices. Reserve up to 25% of
the selection for the highest dynamic scores at or above 0.5; distribute the
remainder across normalized frame-zero 3D voxels with deterministic
evenly-spaced fill. Sort final source-pixel identities before gathering.

Convert OpenCV world coordinates to Three.js' right-handed y-up basis without
changing handedness. If companion RGB is absent, derive a stable palette from
source view and dynamic score. A raw `.pt` must render without image files.

### 3.4 Baked `.omx4d` asset

The compact baked sample remains a versioned binary envelope:

1. fixed magic and format version;
2. little-endian JSON-header length;
3. UTF-8 manifest;
4. aligned raw typed-array sections.

Its required arrays match the worker's `ViewerDataset` output:

- `positions`: float32 `[frameCount, pointCount, 3]`;
- `colors`: uint8 `[pointCount, 3]`;
- `dynamicScore`: float32 `[pointCount]`;
- `sourceView`: uint16 `[pointCount]`;
- `cameraPose`: float32 `[sourceViewCount, 4, 4]`;
- `intrinsics`: float32 `[sourceViewCount, 3, 3]`.

Raw `.pt` imports may construct these arrays directly in memory; serializing an
intermediate `.omx4d` is unnecessary.

## 4. Playback and interaction

Timeline duration is `frameCount / fps`. Default FPS is 15 because the `.pt`
carries no timing metadata, and the user can override it.

Controls include:

- play/pause, restart, previous/next frame, timeline scrubbing, and looping;
- 0.25x / 0.5x / 1x / 1.5x / 2x rate and FPS override;
- point size, dynamic threshold, RGB/dynamic/source/depth color modes;
- all source views versus one source view;
- motion trails, grid, camera frusta, and reset view;
- file picker and drag/drop for replacing the `.pt`;
- optional audio/video, mute/volume, sync offset, and duration warning.

With source media, derive the frame from
`floor((media.currentTime - syncOffset) * fps)`. Without media, advance from a
monotonic clock in `useFrame`. Scrubbing must update immediately while paused.
Do not interpolate geometry in v1.

## 5. Visual design and states

Use a restrained dark visualization workspace in which the 3D content is
primary. Required states are:

- baked sample loading and ready;
- local file reading, metadata validation, sampling, cancellation, and success;
- archive/schema/security/resource error with recovery action;
- drag-over, WebGL unavailable, and reduced-performance warnings;
- no media, media ready, duration mismatch, and blocked/failed media decode.

All controls require keyboard focus styles, labels/tooltips, and usable hit
targets. Core shortcuts are Space, Left/Right, Home, `L`, and `R`. Respect
reduced-motion preferences for UI animation without disabling requested 3D
playback.

## 6. Performance budgets

- First interactive baked sample: under 3 seconds on a modern desktop after
  local assets are available.
- Default 16-frame renderer dataset: at most 100k points and roughly 20 MiB of
  positions.
- Do not duplicate the full input archive in JavaScript memory. Temporary reads
  are bounded tensor chunks; frame-zero working data and sampled output are the
  principal allocations.
- Large-import time is O(total tensor bytes), even when the selected render
  budget is smaller, because all values must be validated.
- Parsing and sampling must not block UI input or animation on the main thread.
- Steady playback target: 60 render FPS on the DGX Spark at 100k points and no
  less than 30 FPS in the acceptance fixture.
- Use one point-cloud draw call, avoid per-point React objects, upload GPU
  positions only on inference-frame changes, cap DPR, and offer the 50k fallback.

## 7. Security and privacy

- Selected `.pt`, audio, and video files remain on the user's device. The
  application must not issue an API request, upload, telemetry event, or other
  outbound request containing selected-file data.
- Treat `.pt` as hostile bytes. File extensions and MIME types are hints only;
  validate the container, declarative metadata grammar, storage boundaries,
  schema, values, and resource ceilings.
- A restricted parser is safer than executing pickle but does not eliminate
  resource exhaustion. Keep work in a cancellable worker, check limits before
  allocations, and surface bounded, non-attacker-controlled error messages.
- Never implement generic pickle object construction, import pickle globals,
  call reductions, place a selected filename in a shell command, or expose raw
  parser traces in the UI.
- Static hosting removes authentication, temporary-file, native PyTorch, and
  conversion-service attack surfaces. Deployments exposed beyond loopback must
  still review origin policy and the supplied CSP/security headers.

## 8. Testing and acceptance criteria

### Automated

- TypeScript unit tests cover ZIP boundary/entry validation, the exact pickle
  grammar, malicious global/reduction rejection, tensor shape/stride/storage
  checks, finite/range/camera validation, deterministic sampling, coordinate
  conversion, `.omx4d` parsing, playback math, and safe user errors.
- Browser tests cover baked startup, playback, seek, speed, loop, camera reset,
  filters, source views, responsive layout, keyboard controls, media sync,
  malformed local files, cancellation, and recovery.
- Focused synthetic fixtures exercise the ZIP, pickle, schema, sampler, and
  streaming converter paths. The real-file gate exercises the complete worker
  path without an API.
- The full browser gate selects `OMNIX_REAL_PT`, blocks `/api/**`, asserts no
  selected-file network request, and verifies the resulting manifest and WebGL
  playback.

### Manual / release gate

- The baked asset visibly reconstructs a coherent moving deer/street scene.
- Chromium selects the real 442,556,165-byte archive; parsing completes without
  locking the UI and yields 16 frames at the requested point budget.
- Play/pause/seek/loop/rate always select the expected discrete frame.
- Original media remains within one inference frame after seeking and during a
  synthetic clock test; duration mismatch is disclosed.
- Replacing or cancelling a dataset releases old worker arrays and Three.js GPU
  resources while preserving the prior valid scene after an error.
- `pnpm lint`, `pnpm test`, `pnpm build`, the browser suite, the real-file gate,
  and `docker compose ... config --quiet` pass without Python installed or an
  application API running.

## 9. Browser-only delivery plan

1. Implement and unit-test bounded ZIP and restricted pickle metadata readers.
2. Port schema validation, stable sampling, basis conversion, and palette logic
   into the existing Web Worker.
3. Add typed client requests, progress, cancellation, and direct
   `ViewerDataset` transfer while retaining baked `.omx4d` decoding.
4. Replace upload/API UI language and error mapping with local-read states.
5. Remove the Python viewer service, Vite/Nginx API proxies, and multi-service
   Compose topology.
6. Update automated browser fixtures, security documentation, and deployment
   instructions.
7. Run the release gate against the actual 422 MiB artifact in Chromium before
   declaring the revision complete.

## 10. Deliberate non-goals

- Running OmniX inference in the browser;
- accepting arbitrary `.pt` schemas, model checkpoints, devices, dtypes, or
  pickle objects;
- photorealistic Gaussian-splat rendering;
- geometry interpolation between inferred frames;
- extracting or transcoding media;
- server-side conversion, upload persistence, authentication, or multi-user data
  storage.

## 11. References

- [PyTorch serialization semantics](https://docs.pytorch.org/docs/main/notes/serialization.html)
- [PyTorch `torch.load` security guidance](https://docs.pytorch.org/docs/stable/generated/torch.load.html)
- [React Three Fiber installation](https://r3f.docs.pmnd.rs/getting-started/installation)
- [React Three Fiber performance guidance](https://r3f.docs.pmnd.rs/advanced/pitfalls)
- [Three.js `BufferGeometry`](https://threejs.org/docs/pages/BufferGeometry.html)
- [Three.js `BufferAttribute`](https://threejs.org/docs/pages/BufferAttribute.html)
- [MDN File API](https://developer.mozilla.org/en-US/docs/Web/API/File_API)
- [MDN `Blob.slice()`](https://developer.mozilla.org/en-US/docs/Web/API/Blob/slice)
- [MDN Web Workers](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API)
- [MDN transferable objects](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Transferable_objects)
- [MDN media `currentTime`](https://developer.mozilla.org/en-US/docs/Web/API/HTMLMediaElement/currentTime)

## 12. Verification record

The original server-backed release gate completed on 2026-07-16. It established
the reference artifact and renderer behavior but is not evidence that the new
browser-only ingestion path works:

- Input: 442,556,165-byte `predictions.pt`, SHA-256
  `39b2e72dafaa56f634d4185d44372e63074d28ef36537a1bac11600ddcc76743`.
- Baked renderer fixture: 20,103,008 bytes, SHA-256
  `d04f33095a1fc5adb523652746aa3f6c3ed6ad64cfb3035bd8757096d23afdaf`.
- Reference 100k identity hash:
  `e4ef50be1678d24592fc63867cc4b3341882afe77d839273eebcf60622035ff3`.
- The baked sample, transport, responsive layout, synchronized WAV clock, and
  WebGL rendering passed in production Chromium.

The browser-only release gate completed on 2026-07-17:

- The production Compose definition resolved to one static `web` service with
  no Python runtime, application API, or converter container.
- Production Chromium selected the same 442,556,165-byte archive and decoded it
  to 100,000 stable points across 16 frames entirely in its Web Worker.
- All six browser tests passed in 8.7 seconds, including real-file
  cancellation/retry, WebGL rendering and playback, malformed-input recovery,
  synchronized source media, and zero post-selection network requests.
- The real-file test blocked `/api/**`, observed zero API requests, and recorded
  no page or console errors.
- The standard Playwright gate used Chromium with ANGLE SwiftShader. This gate
  proves browser functionality; it does not claim native-GPU performance.
- `pnpm lint`, `pnpm test`, `pnpm build`, and
  `docker compose --file viewer/compose.yaml config --quiet` passed.
