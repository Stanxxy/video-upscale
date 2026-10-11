# Admitted-run media preparation (opt-in)

`service/analysis_media.py` implements the local media boundary needed by the
position DAG's future provider adapter. It does not upload to Gemini, dispatch a
model, change admission/quotas, persist a provider-file receipt or activate a runner.
No new storage schema or dependency is introduced.

## Host contract

Construct `S3RunMedia` with an explicit S3 client, artifact bucket allowlist,
existing scratch parent, source-byte limit and asynchronous `guard(run)`.
The guard must load current authority for the exact registered run (including its
immutable artifact references), reject cancellation/revocation/lease loss, and
must not treat a browser-supplied record or a bucket allowlist as authorization.
Configure bounded S3 connect/read timeouts and safe retries on the host client.
Use a private worker/container with filesystem, memory and CPU limits; FFmpeg's
individual-allocation limit and subprocess timeout are not an OS sandbox.

`async with resolver.open(registered_run) as media`:

1. Validates the installed run contract and all source/crop/config buckets.
2. Retrieves exact configuration bytes; validates their SHA and core/prompt bundle.
3. Streams the full original video to an exclusively created private file, checks
   its SHA and byte count, and probes its video stream, duration and time origin.
4. Retrieves and decodes both private PNG/JPEG crops in the admitted athlete order.
5. Rechecks current authority before yielding media to the host.

S3 keys are opaque: literal `%20`, spaces, `#`, `?` and path-like content are not
URL-decoded or used as local filenames. Generated scratch directories are mode0700
and finished files mode0400. Unexpected content, missing objects, I/O timeouts and
denied authority fail visibly; no URL/image fallback, re-encoding or placeholder.

Limits are resource controls, not new user duration/quota rules: the host supplies
the source byte cap; defaults bound config to 2MiB, each crop to 10MiB/4096² pixels,
probe/decode calls to 30s, and concurrent open media contexts to two.

## Identity and timestamp meaning

`media.crops` contains exact original bytes plus detected MIME, dimensions, digest,
player ID and display name. A PNG named `.jpg` remains PNG. Corrupt images and
multiple concatenated frames reject. This proves correspondence to the admitted
crop artifact, not that a human selected the correct athlete or that the image was
captured from the requested source: crop capture/provenance and provider identity
accuracy still require their own validation.

`media.source_path` is the full source, never the selected scope silently trimmed
to zero. Duration uses the registration inspector's floor-to-millisecond convention
and must match exactly. This increment supports one non-cover-art video stream
and an explicitly zero container/video start time. Nonzero/unknown starts and
ambiguous video streams reject until a clock/stream conversion is qualified.
The demuxer allowlist matches the existing account inspector's container families;
it is not a claim that every format/codec is Gemini-compatible or fully decodable.

The future native provider adapter must send the exact prepared observation offsets
and FPS, bind ordered crops and translate/verify its response clock against the
core's absolute-millisecond contract. Provider offsets/model behavior have not been
qualified by local probes. No model-accuracy improvement is claimed here.

For upload, `with media.open_source() as source` rehashes and rewinds the same
read-only, no-symlink descriptor that the uploader consumes. Do this blocking work
off the event loop. Do not reopen the path after verification, modify the file,
retain media past the outer context, or assume the descriptor replaces the required
pre-dispatch authorization/lease check.

## Cancellation and cleanup

Blocking S3/probe/decode operations run off the event loop. Cancellation stops
subsequent work but keeps the scratch directory and concurrency slot until the
current worker really exits, including repeated task cancellation. All context
exits remove only generated local files. Storage is read-only; no original S3 data
is changed. Hard process death cannot run cleanup: host startup/reconciliation must
remove abandoned scratch safely by ownership, not by broad path deletion.

References for the actual inspection controls:
[FFprobe machine-readable stream/container inspection](https://ffmpeg.org/ffprobe.html),
[FFmpeg protocol allowlists](https://ffmpeg.org/ffmpeg-protocols.html).

## Local verification

From the engine worktree, with installed core0.3/shared1.5 in the backend environment:

```sh
ANNOTATION_TEST_ENDPOINT=http://127.0.0.1:4566 ../bjj-vision-backend-wt-s39/.venv/bin/python -m pytest --confcutdir=tests/media_preparation tests/media_preparation -q --tb=short
```

Tests generate tiny real videos/crops with FFmpeg and use an explicitly loopback S3
service with disposable UUID-owned buckets. No provider calls, external video
downloads, production resources or user data are involved.
