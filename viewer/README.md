# OmniX 4D viewer

This directory contains a static React Three Fiber application that opens raw
OmniX `predictions.pt` output and baked `.omx4d` renderer payloads directly in
the browser. PT tensor storage is read and sampled in a Web Worker; OMX4D is
validated and decoded in that worker without resampling. Selected files are
never uploaded, and no Python runtime or application server is involved. A
compact `.omx4d` deer sample remains bundled so the first screen is useful
before a file is selected.

The only network requests made by the application are ordinary same-origin
requests for its JavaScript, CSS, worker bundle, and baked sample. Selecting a
`.pt`, `.omx4d`, audio file, or video file creates browser-local data and does
not issue a request containing that file.

## Supported `.omx4d` contract

The viewer accepts OMX4D format/schema version 1 point payloads. The worker
validates the manifest, frame timing, bounds, six typed attribute descriptors,
section alignment and overlap, finite float data, dynamic-score range, and
source-view indices before transferring zero-copy typed arrays to the renderer.
The baked file keeps its own point count, FPS, RGB values, and warnings; the PT
import-quality selector does not resample an OMX4D file.

## Supported `.pt` contract

`.pt` is a filename convention, not a universal interchange format. The viewer
accepts the exact plain-tensor archive written by this repository: a little-
endian, ZIP-based `torch.save` file whose root is a dictionary with four dense,
contiguous CPU float32 tensors.

| Key | Shape | Meaning |
| --- | --- | --- |
| `trajectory` | `[source view, frame, height, width, 3]` | World-space xyz trajectory |
| `camera_pose` | `[source view, 3, 4]` | OpenCV camera-to-world pose |
| `intrinsics` | `[source view, 3, 3]` | Camera intrinsic matrix |
| `pts3d_dynamic_score` | `[source view, height, width]` | Per-source-pixel motion probability |

The worker does not contain a general Python pickle implementation. It reads a
small, explicitly whitelisted metadata grammar, treats tensor reconstruction
symbols as inert descriptors, validates storage sizes and contiguous strides,
and rejects every other object, global, device, dtype, compression method, or
schema. Model checkpoints and arbitrary third-party `.pt` files are not
supported.

## Run locally for development

Prerequisites are Node.js 20.19 or newer, pnpm 10.22.0, a modern desktop browser
with WebGL and module-worker support, and enough browser memory for the selected
quality. Run:

```bash
cd viewer
pnpm install --frozen-lockfile
pnpm dev
```

Open [http://127.0.0.1:4173](http://127.0.0.1:4173), then use
**Open .pt / .omx4d** or drop either supported format onto the viewer. The worker
reports PT read/validation/sampling progress and transfers only renderer-ready
arrays back to the UI thread. OMX4D validation is also off-main-thread. The
optional media picker uses a local audio or video file as the playback clock and
keeps its existing audio synchronized with the 3D data.

The production build is also a set of static files:

```bash
cd viewer
pnpm build
pnpm exec vite preview --host 127.0.0.1 --port 4173
```

Opening `dist/index.html` directly with a `file:` URL is not supported because
browsers restrict module workers and asset requests on opaque origins. Any
ordinary static HTTP host is sufficient; there is no conversion backend.

## Run the static Docker image

The Compose profile builds the frontend and serves it from an unprivileged
Nginx container. It does not build the OmniX/PyTorch image and contains no API
service:

```bash
docker compose --file viewer/compose.yaml up --build
```

Open [http://127.0.0.1:4173](http://127.0.0.1:4173). The port binds to loopback,
the root filesystem is read-only, all Linux capabilities are dropped, and
`no-new-privileges` is enabled. Stop it with:

```bash
docker compose --file viewer/compose.yaml down
```

The Dockerfile also exposes a Vite development target:

```bash
docker build --file viewer/Dockerfile --target development --tag omnix-viewer-dev .
docker run --rm --publish 127.0.0.1:4173:4173 omnix-viewer-dev
```

Both container routes support local `.pt` and `.omx4d` selection because
conversion/decoding happens inside the visiting browser rather than inside the
container.

## Verify changes

Run type checks, unit tests, and a production build:

```bash
cd viewer
pnpm lint
pnpm test
pnpm build
```

Install Chromium once and run the real-browser suite:

```bash
cd viewer
pnpm exec playwright install chromium
pnpm test:e2e
```

The unit suite includes synthetic ZIP, pickle, schema, sampling, and streaming
converter fixtures; the browser suite covers local OMX4D selection and
malformed-input recovery. The release gates additionally open an actual OmniX
archive and a baked 500k OMX4D through the complete worker and renderer paths.
Paths must be absolute; without them, the corresponding expensive tests are
skipped. For a full-resolution performance gate, serve the production build in
one terminal:

```bash
pnpm build
pnpm exec vite preview --host 127.0.0.1 --port 4173
```

Then run the files through hardware Chromium from a second terminal:

```bash
OMNIX_REAL_PT=/absolute/path/to/predictions.pt \
OMNIX_REAL_OMX4D=/absolute/path/to/chunk.omx4d \
OMNIX_E2E_HARDWARE=1 \
pnpm test:e2e
```

Hardware mode uses full headed Chromium, requires a working X11 display, and
fails if WebGL falls back to SwiftShader. Full-file gates automatically use one
worker. Without hardware mode, the suite uses portable SwiftShader for ordinary
CI coverage.

The release gates select each file through Chromium, verify that no
API/upload/server request occurs, wait for a 500k renderer-ready dataset,
compare two rendered frames, confirm the WebGL context survives a full loop,
visit all 32 frames, and enforce realtime 8 FPS cadence. Validate the static
container definition separately with:

```bash
docker compose --file viewer/compose.yaml config --quiet
```

## Performance and limits

- PT imports accept files and validated tensor storage up to 2 GiB. The current
  32-view, 32-frame inference artifacts are about 1.75 GB.
- PT quality options are 50k, 100k, 200k, and 500k. A 32-frame 500k result uses
  about 187 MiB for its dataset, plus the current-frame and GPU buffers.
- The worker uses `File.slice()` and bounded tensor chunks. It does not first
  copy a 1.75 GB PT archive into one JavaScript `ArrayBuffer`.
- OMX4D is already renderer-ready and is streamed into a worker-owned
  `ArrayBuffer`; cancellation is checked between chunks and its baked point
  count is preserved. A 500k file is about 187.4 MiB.
- Every trajectory value still has to be read and validated locally. Large
  files therefore take time proportional to their tensor storage even when the
  rendered point budget is small.
- Parsing and sampling stay off the UI thread. The final sampled positions,
  scores, colors, camera matrices, and source identities are transferred rather
  than cloned.
- Playback uses one Three.js points draw call and updates the GPU position
  attribute only when the discrete inference frame changes.
- Browser memory and typed-array ceilings vary. Use 50k on constrained devices;
  closing the tab releases all selected-file and GPU state.

## Security and privacy

- Treat every selected `.pt` or `.omx4d` as untrusted data. Extension and MIME
  type are hints only; archive/payload structure, metadata, shapes, storage and
  section lengths, finite values, calibration, and configured resource ceilings
  are checked before rendering.
- Pickle globals and reductions are parsed only as tokens in the allowlisted
  tensor descriptor grammar. They are never imported, invoked, or evaluated.
- Browser-only parsing removes the network upload and native PyTorch loading
  surfaces, but it cannot prevent a deliberately expensive local file from
  consuming CPU or tab memory. The worker enforces limits and supports
  cancellation; browser process isolation remains an additional boundary.
- Selected `.pt`, `.omx4d`, audio, and video files remain on the user's device.
  There are no credentials, temporary server files, analytics, persistence, or
  outbound conversion requests.
- Host the static build with the supplied security headers or equivalent. The
  provided container binds only to `127.0.0.1`; review authentication and origin
  policy before exposing any deployment publicly.
