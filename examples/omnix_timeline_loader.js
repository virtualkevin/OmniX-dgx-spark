/** Dependency-free browser loader for `omnix.timeline.v1` web shards. */

const TIMELINE_SCHEMA = "omnix.timeline.v1";
const SAFE_ID = /^[a-z0-9][a-z0-9_-]*$/;
const SHA256 = /^[0-9a-f]{64}$/;
const TYPE_INFO = {
  "float32-le": { ArrayType: Float32Array, bytes: 4 },
  "uint32-le": { ArrayType: Uint32Array, bytes: 4 },
  uint8: { ArrayType: Uint8Array, bytes: 1 },
};

function assert(condition, message) {
  if (!condition) throw new Error(`Invalid OmniX timeline: ${message}`);
}

function product(shape) {
  return shape.reduce((total, value) => {
    assert(Number.isSafeInteger(value) && value > 0, `invalid shape ${shape}`);
    const next = total * value;
    assert(Number.isSafeInteger(next), `shape is too large ${shape}`);
    return next;
  }, 1);
}

function isSafeInteger(value, minimum = 0) {
  return Number.isSafeInteger(value) && value >= minimum;
}

function approximatelyEqual(left, right, tolerance) {
  return Math.abs(left - right) <= tolerance;
}

function timestampTolerance(fps) {
  return Math.max(1e-6, (1 / fps) * 1e-5);
}

function validateId(value, label) {
  assert(typeof value === "string" && SAFE_ID.test(value), `${label} is not a lowercase slug`);
}

function validateSha256(value, label) {
  assert(typeof value === "string" && SHA256.test(value), `${label} is not lowercase SHA-256`);
}

function validateDescriptor(descriptor, dtype, shape, axes, expectedPath, paths) {
  assert(descriptor && descriptor.dtype === dtype, `expected ${dtype} descriptor`);
  assert(
    JSON.stringify(descriptor.shape) === JSON.stringify(shape),
    `expected shape [${shape}], got [${descriptor.shape}]`,
  );
  assert(
    JSON.stringify(descriptor.axes) === JSON.stringify(axes),
    `${expectedPath} axis contract mismatch`,
  );
  const type = TYPE_INFO[dtype];
  assert(descriptor.bytes === product(shape) * type.bytes, `${descriptor.path} byte count mismatch`);
  assert(descriptor.path === expectedPath, `expected descriptor path ${expectedPath}`);
  validateSha256(descriptor.sha256, `${descriptor.path} checksum`);
  assert(!paths.has(descriptor.path), `duplicate descriptor path ${descriptor.path}`);
  paths.add(descriptor.path);
}

function bytesToHex(bytes) {
  return [...bytes].map((value) => value.toString(16).padStart(2, "0")).join("");
}

async function loadTypedArray(manifestUrl, descriptor, verifyChecksum) {
  const type = TYPE_INFO[descriptor.dtype];
  assert(type, `unsupported dtype ${descriptor.dtype}`);
  const response = await fetch(new URL(descriptor.path, manifestUrl));
  if (!response.ok) throw new Error(`Failed ${response.status}: ${descriptor.path}`);
  const buffer = await response.arrayBuffer();
  assert(buffer.byteLength === descriptor.bytes, `${descriptor.path} response length mismatch`);
  if (verifyChecksum) {
    validateSha256(descriptor.sha256, `${descriptor.path} checksum`);
    assert(globalThis.crypto?.subtle, "Web Crypto is required for checksum verification");
    const digest = new Uint8Array(await globalThis.crypto.subtle.digest("SHA-256", buffer));
    assert(bytesToHex(digest) === descriptor.sha256, `${descriptor.path} checksum mismatch`);
  }
  return new type.ArrayType(buffer);
}

function validateChunkMetadata(manifest, video, chunk, descriptorPaths) {
  const frameCount = manifest.sampling.frames_per_chunk;
  const fps = manifest.sampling.fps;
  const modelWidth = manifest.sampling.model_width;
  const modelHeight = manifest.sampling.model_height;
  const tolerance = timestampTolerance(fps);
  validateId(chunk.id, `chunk ID in ${video.id}`);
  assert(isSafeInteger(chunk.index), `${chunk.id} has invalid index`);
  assert(Number.isFinite(chunk.start_time_s) && chunk.start_time_s >= 0, `${chunk.id} has invalid start`);
  assert(
    Number.isFinite(chunk.end_time_s) && chunk.end_time_s > chunk.start_time_s,
    `${chunk.id} has invalid end`,
  );
  assert(chunk.web && typeof chunk.web === "object", `${chunk.id} has no web metadata`);
  const pointCount = chunk.web.point_count;
  assert(isSafeInteger(chunk.valid_frames, 1), `${chunk.id} has no valid frames`);
  assert(isSafeInteger(chunk.pad_frames), `${chunk.id} has invalid padding`);
  assert(chunk.valid_frames + chunk.pad_frames === frameCount, `${chunk.id} padding mismatch`);
  assert(Array.isArray(chunk.sample_times_s), `${chunk.id} has no timestamps`);
  assert(chunk.sample_times_s.length === frameCount, `${chunk.id} timestamp count mismatch`);
  assert(
    chunk.sample_times_s.every((value) => Number.isFinite(value)),
    `${chunk.id} contains a non-finite timestamp`,
  );
  const expectedEnd = chunk.start_time_s + chunk.valid_frames / fps;
  assert(
    approximatelyEqual(chunk.end_time_s, expectedEnd, tolerance),
    `${chunk.id} end does not match its valid sample count`,
  );
  for (let index = 0; index < chunk.valid_frames; index += 1) {
    const expected = chunk.start_time_s + index / fps;
    assert(
      approximatelyEqual(chunk.sample_times_s[index], expected, tolerance),
      `${chunk.id} timestamp ${index} does not match sampling FPS`,
    );
  }
  const lastValidTime = chunk.sample_times_s[chunk.valid_frames - 1];
  for (let index = chunk.valid_frames; index < frameCount; index += 1) {
    assert(
      approximatelyEqual(chunk.sample_times_s[index], lastValidTime, tolerance),
      `${chunk.id} padded timestamp ${index} does not repeat the final valid frame`,
    );
  }
  assert(isSafeInteger(pointCount, 1), `${chunk.id} has invalid point count`);
  assert(
    pointCount <= Math.min(manifest.point_selection.maximum_points, modelWidth * modelHeight),
    `${chunk.id} point count exceeds its declared maximum`,
  );
  assert(
    isSafeInteger(chunk.web.reference_index) && chunk.web.reference_index < chunk.valid_frames,
    `${chunk.id} reference frame is padded`,
  );
  const prefix = `chunks/${manifest.packaging.profile_id}/${video.id}/${chunk.id}/`;
  validateDescriptor(
    chunk.web.positions,
    "float32-le",
    [frameCount, pointCount, 3],
    ["time", "point", "xyz"],
    `${prefix}positions.f32.bin`,
    descriptorPaths,
  );
  validateDescriptor(
    chunk.web.pixel_indices,
    "uint32-le",
    [pointCount],
    ["point"],
    `${prefix}pixels.u32.bin`,
    descriptorPaths,
  );
  validateDescriptor(
    chunk.web.colors,
    "uint8",
    [pointCount, 3],
    ["point", "rgb"],
    `${prefix}colors.u8.bin`,
    descriptorPaths,
  );
  validateDescriptor(
    chunk.web.dynamic_score,
    "uint8",
    [pointCount],
    ["point"],
    `${prefix}dynamic.u8.bin`,
    descriptorPaths,
  );
  assert(
    approximatelyEqual(chunk.web.dynamic_score.scale, 1 / 255, 1e-12),
    `${chunk.id} dynamic-score scale mismatch`,
  );
  validateDescriptor(
    chunk.web.camera_pose,
    "float32-le",
    [frameCount, 3, 4],
    ["time", "row", "column"],
    `${prefix}camera_pose.f32.bin`,
    descriptorPaths,
  );
  validateDescriptor(
    chunk.web.intrinsics,
    "float32-le",
    [frameCount, 3, 3],
    ["time", "row", "column"],
    `${prefix}intrinsics.f32.bin`,
    descriptorPaths,
  );
  assert(chunk.raw_pt && typeof chunk.raw_pt === "object", `${chunk.id} has no PT provenance`);
  assert(isSafeInteger(chunk.raw_pt.bytes, 1), `${chunk.id} has invalid PT byte count`);
  validateSha256(chunk.raw_pt.sha256, `${chunk.id} raw PT checksum`);
  assert(
    chunk.raw_pt.role === "provenance-only-not-required-by-browser",
    `${chunk.id} has invalid raw PT role`,
  );
  assert(
    chunk.alignment?.status === "chunk-local" && chunk.alignment.segment === chunk.index,
    `${chunk.id} alignment metadata mismatch`,
  );
}

function validateVideo(manifest, video, descriptorPaths) {
  validateId(video.id, "video ID");
  validateSha256(video.source_sha256, `${video.id} source checksum`);
  assert(Number.isFinite(video.duration_s) && video.duration_s > 0, `${video.id} has invalid duration`);
  assert(isSafeInteger(video.source_chunk_count, 1), `${video.id} has invalid source chunk count`);
  assert(isSafeInteger(video.packed_chunk_count, 1), `${video.id} has invalid packed chunk count`);
  assert(Array.isArray(video.chunks) && video.chunks.length > 0, `${video.id} has no chunks`);
  assert(video.packed_chunk_count === video.chunks.length, `${video.id} packed count mismatch`);
  assert(video.packed_chunk_count <= video.source_chunk_count, `${video.id} packed too many chunks`);
  assert(typeof video.timeline_complete === "boolean", `${video.id} has no completeness flag`);
  assert(
    video.timeline_complete === (video.packed_chunk_count === video.source_chunk_count),
    `${video.id} completeness flag disagrees with chunk counts`,
  );

  const tolerance = timestampTolerance(manifest.sampling.fps);
  const framePeriod = 1 / manifest.sampling.fps;
  const chunkIds = new Set();
  const chunkIndices = new Set();
  let previous;
  for (const chunk of video.chunks) {
    validateChunkMetadata(manifest, video, chunk, descriptorPaths);
    assert(!chunkIds.has(chunk.id), `duplicate chunk ID ${video.id}/${chunk.id}`);
    assert(!chunkIndices.has(chunk.index), `duplicate chunk index ${video.id}/${chunk.index}`);
    chunkIds.add(chunk.id);
    chunkIndices.add(chunk.index);
    assert(
      chunk.start_time_s <= video.duration_s + tolerance,
      `${video.id}/${chunk.id} starts beyond the source duration`,
    );
    assert(
      chunk.end_time_s <= video.duration_s + framePeriod + tolerance,
      `${video.id}/${chunk.id} ends too far beyond the source duration`,
    );
    if (previous) {
      assert(chunk.index > previous.index, `${video.id} chunks are not ordered by index`);
      assert(chunk.start_time_s >= previous.end_time_s, `${video.id} chunks overlap`);
      if (video.timeline_complete) {
        assert(chunk.index === previous.index + 1, `${video.id} complete timeline skips an index`);
        assert(
          approximatelyEqual(chunk.start_time_s, previous.end_time_s, tolerance),
          `${video.id} complete timeline has a gap`,
        );
      }
    }
    previous = chunk;
  }

  if (video.timeline_complete) {
    const first = video.chunks[0];
    const last = video.chunks.at(-1);
    assert(first.index === 0, `${video.id} complete timeline does not begin at index zero`);
    assert(approximatelyEqual(first.start_time_s, 0, tolerance), `${video.id} does not begin at zero`);
    const sourceTail = video.duration_s - last.end_time_s;
    assert(
      sourceTail >= -framePeriod - tolerance && sourceTail <= framePeriod + tolerance,
      `${video.id} final chunk is not within one sample period of source duration`,
    );
  }
}

function validateManifest(manifest) {
  assert(manifest && typeof manifest === "object", "manifest is not an object");
  assert(manifest.schema === TIMELINE_SCHEMA, `unsupported schema ${manifest.schema}`);
  const sampling = manifest.sampling;
  assert(sampling && typeof sampling === "object", "sampling metadata is missing");
  assert(Number.isFinite(sampling.fps) && sampling.fps > 0, "invalid sampling FPS");
  assert(isSafeInteger(sampling.frames_per_chunk, 1), "invalid frames per chunk");
  assert(isSafeInteger(sampling.model_width, 1), "invalid model width");
  assert(isSafeInteger(sampling.model_height, 1), "invalid model height");
  assert(sampling.non_overlapping === true, "timeline must use non-overlapping chunks");
  assert(sampling.tail_policy === "repeat-last-frame-padding", "unsupported tail policy");

  const packaging = manifest.packaging;
  assert(packaging && typeof packaging === "object", "packaging metadata is missing");
  assert(typeof packaging.complete === "boolean", "invalid package completeness flag");
  validateId(packaging.profile_id, "packaging profile ID");
  for (const field of [
    "source_chunk_count",
    "selected_chunk_count",
    "packed_chunk_count",
    "skipped_chunk_count",
  ]) {
    assert(isSafeInteger(packaging[field], field === "skipped_chunk_count" ? 0 : 1), `invalid ${field}`);
  }
  assert(packaging.selected_chunk_count <= packaging.source_chunk_count, "selected too many chunks");
  assert(packaging.packed_chunk_count <= packaging.selected_chunk_count, "packed too many chunks");
  assert(
    packaging.packed_chunk_count + packaging.skipped_chunk_count === packaging.selected_chunk_count,
    "selected/packed/skipped counts disagree",
  );
  assert(packaging.filters && typeof packaging.filters === "object", "packaging filters are missing");
  const filterVideo = packaging.filters.video;
  const filterLimit = packaging.filters.limit;
  assert(filterVideo === null || typeof filterVideo === "string", "invalid video filter");
  if (filterVideo !== null) validateId(filterVideo, "video filter");
  assert(filterLimit === null || isSafeInteger(filterLimit, 1), "invalid limit filter");
  assert(Array.isArray(packaging.skipped), "skipped chunk metadata is missing");
  assert(packaging.skipped.length === packaging.skipped_chunk_count, "skipped count mismatch");
  for (const skipped of packaging.skipped) {
    validateId(skipped.video_id, "skipped video ID");
    validateId(skipped.chunk_id, "skipped chunk ID");
    assert(typeof skipped.reason === "string" && skipped.reason.length > 0, "invalid skip reason");
  }
  const shouldBeComplete =
    filterVideo === null &&
    filterLimit === null &&
    packaging.skipped_chunk_count === 0 &&
    packaging.packed_chunk_count === packaging.source_chunk_count;
  assert(packaging.complete === shouldBeComplete, "package completeness metadata is inconsistent");

  const pointSelection = manifest.point_selection;
  assert(pointSelection && typeof pointSelection === "object", "point selection metadata is missing");
  assert(isSafeInteger(pointSelection.maximum_points, 1), "invalid maximum point count");
  assert(
    pointSelection.maximum_points <= sampling.model_width * sampling.model_height,
    "maximum point count exceeds model pixels",
  );
  const expectedProfile = `${TIMELINE_SCHEMA.replaceAll(".", "-")}-points-${String(
    pointSelection.maximum_points,
  ).padStart(6, "0")}`;
  assert(packaging.profile_id === expectedProfile, "profile ID does not match point count/schema");

  const binary = manifest.binary_contract;
  assert(binary?.endianness === "little", "unsupported binary endianness");
  assert(binary.array_order === "C-row-major", "unsupported binary array order");
  assert(binary.pixel_indexing === "pixel=y*model_width+x", "unsupported pixel indexing");
  assert(binary.paths === "relative to this manifest", "unsupported binary path contract");
  assert(binary.shard_profile === packaging.profile_id, "binary shard profile mismatch");

  const coordinates = manifest.coordinate_contract;
  assert(coordinates?.camera_pose === "camera-to-world in the same chunk-local gauge", "invalid pose contract");
  assert(coordinates.camera_basis === "OpenCV: +x right, +y down, +z forward", "invalid camera basis");
  assert(coordinates.camera_matrix_layout === "row-major 3x4 camera-to-world", "invalid camera layout");
  assert(coordinates.cross_chunk_alignment === "none", "unexpected cross-chunk alignment claim");
  assert(coordinates.threejs_conversion?.points === "p_three=C*p_opencv", "invalid point conversion");
  assert(
    coordinates.threejs_conversion?.camera === "c2w_three=C*c2w_opencv*C",
    "invalid camera conversion",
  );

  assert(
    JSON.stringify(manifest.preprocessing?.model_resolution) ===
      JSON.stringify([sampling.model_width, sampling.model_height]),
    "preprocessing resolution mismatch",
  );
  assert(Array.isArray(manifest.videos) && manifest.videos.length > 0, "manifest has no videos");
  const descriptorPaths = new Set();
  const videoIds = new Set();
  let packedCount = 0;
  let representedSourceCount = 0;
  for (const video of manifest.videos) {
    validateVideo(manifest, video, descriptorPaths);
    assert(!videoIds.has(video.id), `duplicate video ID ${video.id}`);
    videoIds.add(video.id);
    packedCount += video.packed_chunk_count;
    representedSourceCount += video.source_chunk_count;
  }
  assert(packedCount === packaging.packed_chunk_count, "video chunk totals disagree with packaging");
  assert(manifest.packed_chunk_count === packedCount, "top-level packed chunk count mismatch");
  assert(representedSourceCount <= packaging.source_chunk_count, "represented too many source chunks");
  if (packaging.complete) {
    assert(representedSourceCount === packaging.source_chunk_count, "complete package omits a source video");
    assert(manifest.videos.every((video) => video.timeline_complete), "complete package has a partial video");
  }
}

function resolveManifestUrl(manifestUrl) {
  const baseUrl = globalThis.document?.baseURI ?? globalThis.location?.href;
  try {
    return baseUrl ? new URL(manifestUrl, baseUrl) : new URL(manifestUrl);
  } catch (error) {
    throw new TypeError(
      `Invalid OmniX manifest URL: use an absolute URL when document/location is unavailable (${error})`,
    );
  }
}

export async function openOmniXTimeline(
  manifestUrl,
  { verifyChecksums = true, maxCachedChunks = 4 } = {},
) {
  assert(typeof verifyChecksums === "boolean", "verifyChecksums must be boolean");
  assert(isSafeInteger(maxCachedChunks, 1), "maxCachedChunks must be a positive integer");
  const resolvedManifestUrl = resolveManifestUrl(manifestUrl);
  const response = await fetch(resolvedManifestUrl);
  if (!response.ok) throw new Error(`Failed ${response.status}: ${resolvedManifestUrl}`);
  const manifest = await response.json();
  validateManifest(manifest);

  const littleEndian = new Uint8Array(new Uint16Array([0x00ff]).buffer)[0] === 0xff;
  assert(littleEndian, "this loader requires a little-endian browser");
  const modelWidth = manifest.sampling.model_width;
  const modelHeight = manifest.sampling.model_height;

  const videos = new Map(manifest.videos.map((video) => [video.id, video]));
  const cache = new Map();

  function metadata(videoId, chunkId) {
    const video = videos.get(videoId);
    if (!video) throw new RangeError(`Unknown OmniX video: ${videoId}`);
    const chunk = video.chunks.find((candidate) => candidate.id === chunkId);
    if (!chunk) throw new RangeError(`Unknown OmniX chunk: ${videoId}/${chunkId}`);
    return { video, chunk };
  }

  function locate(videoId, timeSeconds) {
    const video = videos.get(videoId);
    if (!video) throw new RangeError(`Unknown OmniX video: ${videoId}`);
    if (!Number.isFinite(timeSeconds)) throw new TypeError("timeSeconds must be finite");
    if (timeSeconds < 0 || timeSeconds > video.duration_s) {
      throw new RangeError(`Time ${timeSeconds} is outside ${videoId}`);
    }
    let chunk = video.chunks.find(
      (candidate) => timeSeconds >= candidate.start_time_s && timeSeconds < candidate.end_time_s,
    );
    if (!chunk && video.timeline_complete) {
      const last = video.chunks.at(-1);
      if (timeSeconds >= last.end_time_s && timeSeconds <= video.duration_s) chunk = last;
    }
    if (!chunk) throw new RangeError(`Time ${timeSeconds} is not covered by a shard in ${videoId}`);
    const times = chunk.sample_times_s.slice(0, chunk.valid_frames);
    let frameIndex = 0;
    while (frameIndex + 1 < times.length && times[frameIndex + 1] <= timeSeconds) frameIndex += 1;
    return { videoId, chunkId: chunk.id, frameIndex, sampleTime: times[frameIndex] };
  }

  async function loadChunk(videoId, chunkId) {
    const key = `${videoId}/${chunkId}`;
    if (cache.has(key)) {
      const value = cache.get(key);
      cache.delete(key);
      cache.set(key, value);
      return value;
    }
    const { chunk } = metadata(videoId, chunkId);
    const web = chunk.web;
    const promise = Promise.all([
      loadTypedArray(resolvedManifestUrl, web.positions, verifyChecksums),
      loadTypedArray(resolvedManifestUrl, web.pixel_indices, verifyChecksums),
      loadTypedArray(resolvedManifestUrl, web.colors, verifyChecksums),
      loadTypedArray(resolvedManifestUrl, web.dynamic_score, verifyChecksums),
      loadTypedArray(resolvedManifestUrl, web.camera_pose, verifyChecksums),
      loadTypedArray(resolvedManifestUrl, web.intrinsics, verifyChecksums),
    ]).then(([positions, pixelIndices, colors, dynamicScoreBytes, cameraPose, intrinsics]) => {
      const invalidPixel = pixelIndices.find((pixel) => pixel >= modelWidth * modelHeight);
      assert(invalidPixel === undefined, `${chunk.id} contains an out-of-range pixel index`);
      const pointCount = web.point_count;
      const result = {
        videoId,
        chunkId,
        frameCount: manifest.sampling.frames_per_chunk,
        validFrameCount: chunk.valid_frames,
        pointCount,
        referenceIndex: web.reference_index,
        sampleTimes: chunk.sample_times_s,
        positions,
        pixelIndices,
        colors,
        dynamicScoreBytes,
        dynamicScoreScale: web.dynamic_score.scale,
        cameraPose,
        intrinsics,
        framePositions(frame) {
          if (!isSafeInteger(frame) || frame >= this.validFrameCount) {
            throw new RangeError(`Frame ${frame} is outside valid frames`);
          }
          const begin = frame * pointCount * 3;
          return positions.subarray(begin, begin + pointCount * 3);
        },
        positionOffset(frame, point) {
          if (!isSafeInteger(frame) || frame >= this.validFrameCount) {
            throw new RangeError(`Frame ${frame} is outside valid frames`);
          }
          if (!isSafeInteger(point) || point >= pointCount) {
            throw new RangeError(`Point ${point} is outside this chunk`);
          }
          return (frame * pointCount + point) * 3;
        },
        pixelXY(point) {
          if (!isSafeInteger(point) || point >= pointCount) {
            throw new RangeError(`Point ${point} is outside this chunk`);
          }
          const pixel = pixelIndices[point];
          return { x: pixel % modelWidth, y: Math.floor(pixel / modelWidth) };
        },
      };
      return result;
    }).catch((error) => {
      if (cache.get(key) === promise) cache.delete(key);
      throw error;
    });
    cache.set(key, promise);
    while (cache.size > maxCachedChunks) cache.delete(cache.keys().next().value);
    return promise;
  }

  return {
    manifest,
    locate,
    loadChunk,
    clearCache: () => cache.clear(),
  };
}
