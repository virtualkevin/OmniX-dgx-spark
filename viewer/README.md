# OmniX 4D viewer

This directory contains a static React Three Fiber application that opens the
OmniX `predictions.pt` output directly in the browser. Raw tensor storage is
read and sampled in a Web Worker; the selected file is never uploaded and no
Python runtime or application server is involved. A compact `.omx4d` deer
sample remains bundled so the first screen is useful before a file is selected.

The only network requests made by the application are ordinary same-origin
requests for its JavaScript, CSS, worker bundle, and baked sample. Selecting a
`.pt`, audio file, or video file creates browser-local data and does not issue a
request containing that file.

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

Open [http://127.0.0.1:4173](http://127.0.0.1:4173), then use **Open .pt** or
drop a supported `predictions.pt` onto the viewer. The worker reports local
read/validation/sampling progress and transfers only renderer-ready arrays back
to the UI thread. The optional media picker uses a local audio or video file as
the playback clock and keeps its existing audio synchronized with the 3D data.

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

Both container routes support local `.pt` selection because conversion happens
inside the visiting browser rather than inside the container.

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
converter fixtures; the browser suite covers malformed-input recovery. The
release gate additionally opens the actual OmniX archive through the complete
worker path. The path must be absolute; without it, the expensive test is
skipped:

```bash
OMNIX_REAL_PT=/absolute/path/to/predictions.pt pnpm test:e2e
```

That test selects the file through Chromium, verifies that no `/api` or upload
request occurs, waits for the renderer-ready dataset, and exercises WebGL
playback. Validate the static container definition separately with:

```bash
docker compose --file viewer/compose.yaml config --quiet
```

## Performance and limits

- The default 100k selection produces about 20 MiB of positions for the
  16-frame deer sequence; 50k and 200k quality options trade detail for memory.
- The worker uses `File.slice()` and bounded tensor chunks. It does not first
  copy a 400+ MiB archive into one JavaScript `ArrayBuffer`.
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

- Treat every selected `.pt` as untrusted data. Extension and MIME type are
  hints only; archive structure, entry names, metadata grammar, tensor shapes,
  storage lengths, finite values, camera matrices, and configured resource
  ceilings are checked before rendering.
- Pickle globals and reductions are parsed only as tokens in the allowlisted
  tensor descriptor grammar. They are never imported, invoked, or evaluated.
- Browser-only parsing removes the network upload and native PyTorch loading
  surfaces, but it cannot prevent a deliberately expensive local file from
  consuming CPU or tab memory. The worker enforces limits and supports
  cancellation; browser process isolation remains an additional boundary.
- Selected `.pt`, audio, and video files remain on the user's device. There are
  no credentials, temporary server files, analytics, persistence, or outbound
  conversion requests.
- Host the static build with the supplied security headers or equivalent. The
  provided container binds only to `127.0.0.1`; review authentication and origin
  policy before exposing any deployment publicly.
