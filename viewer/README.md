# OmniX 4D viewer

This directory contains the local React Three Fiber viewer and the restricted
Python converter that turns one supported OmniX `predictions.pt` into the
browser-native `.omx4d` payload. The baked sample works without the API; opening
another `.pt` requires the API to be running.

The browser requests only relative URLs:

- `GET /sample/deer.omx4d` loads the baked sample.
- `GET /api/health` reports converter limits.
- `POST /api/convert` accepts multipart fields `file`, `point_budget`, and `fps`.

## Run locally for development

Prerequisites are Node.js 20.19 or newer, pnpm 10.22.0, Python 3.11 or newer,
and enough local RAM for the selected `.pt`. Run commands from the repository
root unless a command changes directory.

Create the API environment and start the API in terminal 1:

```bash
python3 -m venv .venv-viewer
source .venv-viewer/bin/activate
python -m pip install --upgrade pip
python -m pip install -r viewer/server/requirements.txt
uvicorn viewer.server.app:app --host 127.0.0.1 --port 8000 --workers 1
```

Install and start the Vite application in terminal 2:

```bash
cd viewer
pnpm install --frozen-lockfile
pnpm dev
```

Open [http://127.0.0.1:4173](http://127.0.0.1:4173). Vite proxies `/api` to
`http://127.0.0.1:8000`. Use **Open .pt** or drag a supported `predictions.pt`
onto the viewer. The optional media picker keeps a local audio or video file as
the playback clock; it does not upload that media to the API.

Check the local API directly:

```bash
curl --fail http://127.0.0.1:8000/api/health
```

Convert without the browser:

```bash
python -m viewer.server /absolute/path/to/predictions.pt /tmp/result.omx4d \
  --point-budget 100000 --fps 15 \
  --image-dir /absolute/path/to/gt_images
```

The HTTP equivalent is:

```bash
curl --fail-with-body http://127.0.0.1:8000/api/convert \
  --form file=@/absolute/path/to/predictions.pt \
  --form point_budget=100000 \
  --form fps=15 \
  --output /tmp/result.omx4d
```

## Run the isolated Docker stack

The API image intentionally extends the repository's `omnix-dgx-spark:latest`
image so its patched PyTorch runtime is reused. Build that base once, then start
the API and production web server:

```bash
docker build --file Dockerfile.spark --tag omnix-dgx-spark:latest .
docker compose --file viewer/compose.yaml up --build
```

Open [http://127.0.0.1:4173](http://127.0.0.1:4173). Only the web proxy is
published, and it is bound to loopback; the API is reachable through
`http://127.0.0.1:4173/api/health` but has no host port of its own.

Stop the stack and remove its ephemeral containers:

```bash
docker compose --file viewer/compose.yaml down
```

The Compose profile applies a 12 GiB memory ceiling, four-CPU quota, PID and
open-file limits, a 3 GiB per-file limit, and a 4 GiB temporary filesystem to
the converter. It also runs both services with read-only roots, all Linux
capabilities dropped, and `no-new-privileges`. The app's 300-second conversion
deadline is a separate wall-clock limit. Tune these deliberately in
`viewer/compose.yaml` if a trusted dataset requires different ceilings; keep the
proxy's `client_max_body_size` and timeouts in `viewer/nginx.conf` aligned.

The web Dockerfile also exposes a development target if a containerized Vite
process is useful:

```bash
docker build --file viewer/Dockerfile --target development --tag omnix-viewer-dev .
docker run --rm --publish 127.0.0.1:4173:4173 omnix-viewer-dev
```

That standalone development container serves the baked sample, but `.pt`
upload needs a reachable API. The Compose stack is the supported all-in-one
container route.

## Verify changes

Run frontend type checks, unit tests, and a production build:

```bash
cd viewer
pnpm lint
pnpm test
pnpm build
```

Run the Python converter tests from the repository root with the API virtual
environment active:

```bash
python -m pip install -r viewer/server/requirements-dev.txt
python -m pytest viewer/server/tests
```

Install the browser once and run the Chromium smoke suite:

```bash
cd viewer
pnpm exec playwright install chromium
pnpm test:e2e
```

The full release gate can drive an actual OmniX archive through the browser and
local API. The path must be absolute; without it, that expensive test is skipped:

```bash
OMNIX_REAL_PT=/absolute/path/to/predictions.pt pnpm test:e2e
```

For a Docker configuration-only check that does not start containers:

```bash
docker compose --file viewer/compose.yaml config --quiet
```

## Converter configuration

All numeric settings must be positive. Values are bytes unless noted.

| Variable | Default | Purpose |
| --- | ---: | --- |
| `OMX_MAX_UPLOAD_BYTES` | 1,073,741,824 | Maximum streamed request file |
| `OMX_MAX_ARCHIVE_BYTES` | 8,589,934,592 | Maximum expanded ZIP contents |
| `OMX_MAX_TENSOR_BYTES` | 8,589,934,592 | Maximum validated tensor storage |
| `OMX_MAX_OUTPUT_BYTES` | 2,147,483,648 | Maximum `.omx4d` output |
| `OMX_MAX_SOURCE_VIEWS` | 64 | Maximum source views |
| `OMX_MAX_FRAMES` | 600 | Maximum target frames |
| `OMX_MAX_SOURCE_PIXELS` | 16,000,000 | Maximum pixels over source views |
| `OMX_MAX_POINT_BUDGET` | 200,000 | Maximum requested sampled points |
| `OMX_MAX_ZIP_ENTRIES` | 512 | Maximum archive entries |
| `OMX_MAX_COMPRESSION_RATIO` | 200 | Maximum ratio for large ZIP entries |
| `OMX_FINITE_CHUNK_ELEMENTS` | 8,000,000 | Finite-check chunk size |
| `OMX_CONVERSION_TIMEOUT` | 300 | Conversion seconds after upload |
| `OMX_UPLOAD_CHUNK_BYTES` | 1,048,576 | Request streaming chunk size |
| `OMX_TEMP_DIRECTORY` | system temp | Temporary source/output directory |

## Security and privacy

- The API accepts only the exact plain-tensor OmniX schema. It uses
  `torch.load(..., weights_only=True, map_location="cpu")` and never falls back
  to unrestricted pickle loading. This reduces code-execution exposure but does
  not make arbitrary `.pt` files safe from resource-exhaustion attempts.
- Uploads are streamed to temporary storage, checked against archive, tensor,
  shape, dtype, finite-value, and output limits, and deleted after the response,
  failure, cancellation, or timeout. The service does not persist datasets.
- The converter has no authentication and is designed for one trusted local
  operator. Do not bind it or the proxy to a LAN/public interface without an
  authenticated, rate-limited isolation layer.
- The API stays exclusively on a Docker network with no external route. Nginx
  also joins a separate edge network, but its only published port binds to
  `127.0.0.1`. Container limits are defense in depth, not a guarantee that a
  hostile PyTorch archive cannot exploit a native-library defect. Keep the Spark
  base and PyTorch packages patched.
- Selected audio/video remains a browser-local object URL. A `.pt` is sent only
  to the local converter endpoint configured by the page origin.
