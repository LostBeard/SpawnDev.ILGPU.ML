# Decouple model delivery from WebTorrent

**Decision (TJ, 2026-09-14):** WebTorrent comes out of `SpawnDev.ILGPU.ML` entirely and returns as a
separate adapter package. ML stops tracking WebTorrent's version, and model delivery becomes "any `Stream`",
fed by an OPFS-cached HTTP download with progress.

> "I am tired of its deficiencies slowing SpawnDev.ILGPU.ML progress... Instead of forcing WebTorrent to
> handle http (which I do like) we take WebTorrent out of SpawnDev.ILGPU.ML for model delivery and let it
> support any Stream."
>
> "the http can still use our hub for cache and cors"

The hub stays the origin. Nothing hits huggingface.co directly - the hub is what supplies CORS, caches, and
keeps us out of HuggingFace's rate limiter.

---

## What was actually coupled - MEASURED, not estimated

Every `.cs` in the library mentioning WebTorrent, classified by whether it has a real code dependency or
only prose:

| File | Real code refs | Disposition |
|---|---|---|
| `InferenceSession.cs` | 27 | split - see below |
| `Hub/HubModelStream.cs` | 5 | moves to the adapter whole |
| `Pipelines/ImageGenerationPipeline.cs` | 0 | comment only |
| `Pipelines/DepthEstimationPipeline.cs` | 0 | comment only |
| `Onnx/OnnxParser.cs`, `OnnxLoader.cs`, `ModelInspectorHelper.cs` | 0 | comment only |
| `Hub/ModelHub.cs` | 0 | comment only |
| `GGUF/GGUFParser.cs` | 0 | comment only |

**Two files.** The surface is far smaller than the 9-file grep suggests, because seven of them only
*mention* WebTorrent in documentation.

`InferenceSession.cs`'s 27 split into exactly two groups:

1. **`CreateFromHuggingFaceAsync`'s `webTorrent` + `http` parameters and the lazy-hash branch** - becomes an
   extension method in the adapter.
2. **The GGUF weight-load profiler** (`AsyncFSChunkStore.ResetReadTiming` / `TraceReadTiming`,
   `Torrent.ResetEnsurePieceTiming` / `TraceEnsurePiece`, and the read-timing dump). This instruments the
   *torrent store's* ranged OPFS reads. The generic half of that block -
   `BrowserBufferPolicy.StreamReadMs/StreamWriteMs/StreamBytes/StreamChunks`, which is `SpawnDev.ILGPU`, not
   WebTorrent - **stays**, and keeps answering the question that actually matters ("stream READ vs GPU
   WRITE"). The torrent-specific counters are inapplicable once the stream is an `OPFSStream`, so moving
   them costs nothing on the default path.

## Why HTTP lost to torrents in the first place

Not because a single-origin download needs a swarm. Because `ModelCache.DownloadWithProgressAsync` was
written to go through the managed heap:

```csharp
using var chunk = result.Value!;
var bytes = chunk.ReadBytes();     // every chunk copied JS -> managed heap
chunks.Add(bytes);                 // whole model accumulated in a List<byte[]>
var output = new byte[received];   // whole model AGAIN, second managed copy
using var uint8 = new Uint8Array(data);   // and back to JS to write OPFS
```

A 1.7 GB checkpoint lands on the single-threaded WASM managed heap **twice** before it is cached. That is a
plain Rule 4 violation, and `GetOrFetchStreamAsync`'s own remarks already named the fix and deferred it:
*"fetching JS-side straight into OPFS ... that is a separate change and is deliberately not smuggled in
here."* This is that change.

## The replacement - `Hub/OpfsModelCache.cs`

`fetch` -> `Response.body` reader -> `Uint8Array` chunk -> `OPFSStream.WriteUint8ArrayAsync` -> OPFS. The
chunk is never `ReadBytes()`d; only its `length` (a number) is read into .NET. The result is an `OPFSStream`,
which is an `IJSReadStream`, which is exactly what `CreateFromOnnxStreamAsync` /
`CreateFromGGUFStreamAsync` already detect for the zero-copy JS->GPU weight upload.

### Write size - MEASURED, and it changed the design

TJ asked whether delivery should use ranged requests at ~4 MB. It does **not**: one `fetch` streams
`Response.body` and progress comes from each chunk's `length`. A ranged loop over a 6.9 GB model is ~1,700
round trips to the hub for progress granularity the streaming body already gives.

But the concern underneath it was right, on the **write** side. MEASURED against the hub, 2,107,653 B file:

| | fetch chunks | avg chunk | OPFS writes | managed alloc |
|---|---|---|---|---|
| chunk-at-a-time | 87-112 | 18,818-24,225 B | **87-112** | 15.8-28.8% |
| 4 MB coalescing | 87-112 | 18,818-24,225 B | **1** | 9.6-20.1% |

`fetch` hands over **~18-24 KB** chunks - squarely in the range that measured ~100 MB/s against ~1000 MB/s
at 4 MB on the OPFS throughput tests. So `WriteBufferSize` (default 4 MiB) coalesces them into one
`Uint8Array` via `TypedArray.set` and writes that, with a `SubArray` VIEW for the tail. All JS-side; the
managed allocation went DOWN, because fewer OPFS writes also means fewer interop crossings. Extrapolated to
a 6.9 GB GGUF: ~1,650 writes of 4 MB instead of ~350,000 of 20 KB.

⚠️ The sidecar records `written` (bytes actually handed to OPFS), never `received` - buffered bytes are not
durable, and a resume trusts the sidecar. On a mid-fill failure the buffer is deliberately NOT flushed.

`OPFSStream` needs **no WebWorkers dependency** - verified, not assumed: its `using
SpawnDev.SpawnJS.WebWorkers.OPFS` resolves to a namespace declared inside SpawnJS's own
`Toolbox/OPFSSyncStream.cs`, and SpawnJS's only `PackageReference` is `SpawnDev.BackgroundServices`. It uses
a `FileSystemSyncAccessHandle` automatically when it happens to be in a dedicated worker, and an async
handle everywhere else.

### Durability rules

These are the difference between a cache and a liability:

- **A non-OK response is never written.** `fetch` resolves (does not throw) on 404/500 with an *error body*.
  Caching that is how a 15-byte 404 page once became a permanently-cached "model" that then failed
  identically on every later run until the cache was cleared by hand.
- **An entry is servable only when its sidecar says `complete` AND the file's real size matches the recorded
  total.** A truncated download is resumed, never served.
- **Resume sends `If-Range`** with the stored ETag, so a changed server-side file yields 200 (full body) and
  a clean restart instead of two different files spliced together.

### Sidecar, not rename

Promotion via an atomic `.part` -> final rename would need `FileSystemFileHandle.move()`, which SpawnJS
does not wrap - deliberately, since MDN does not list it as a method. A `{key}.meta` sidecar does the same
job **and** is needed anyway: resume requires the expected total and last-confirmed count to be persisted.
So there is no new SpawnJS surface and no chain publish gating this work.

The sidecar is `key=value` lines, not JSON: it is internal, ~100 bytes, and a hand-rolled format has no
trimming or source-generator contract to keep in step.

## Prerequisite that was NOT optional

`OPFSStream` landed in **SpawnJS 2.1.15**. ML pinned **2.1.10**. The bump was a hard prerequisite, not
housekeeping:

- `SpawnDev.ILGPU.ML.csproj`: SpawnJS 2.1.10 -> **2.1.17**, SpawnDev.ILGPU 5.2.10 -> **5.2.11**
- `SpawnDev.ILGPU.ML.Demo.csproj`: SpawnJS 2.1.10 -> **2.1.17** (a direct pin lower than what the project
  reference requires is NU1605, Warning-As-Error - the library builds clean and hides it; the Demo does not)

`SpawnDev.AsyncFileSystem` stays at 2.1.1: WebTorrent 4.2.7's nuspec requires exactly 2.1.1, so there is no
conflict to resolve. (Noted separately: WebTorrent **4.2.7 on the feed was packed against RTC 2.2.3 /
SpawnJS 2.1.7 / Cryptography 2.0.1** - the pre-bump chain. The RTC 2.2.4 + Cryptography 2.0.2 bump committed
as `ffdec14` needs a **4.2.8** publish to reach any consumer.)

## Tests - `MLTestBase.OpfsModelCacheTests.cs`

Browser-lane; desktop lanes skip via `UnsupportedTestException`. Source is the hub, live-verified:
`206 Partial Content`, `content-range: bytes 2000000-2107652/2107653`, and
`access-control-expose-headers: Content-Range,Accept-Ranges,Content-Length` so a browser can read it.
(The hub sends **no ETag** for `/hf`, so `If-Range` is currently a no-op there - see Open below.)

| Test | What it would catch |
|---|---|
| `DownloadsCachesAndServes` | length cross-checked against an independent `HttpClient` content-length; second open served from OPFS, byte-compared; result is an `IJSReadStream` |
| `DownloadDoesNotEnterManagedHeap` | 🔴 the treason guard - `GC.GetTotalAllocatedBytes` delta must stay under 50% of the file. A reinstated per-chunk `ReadBytes()` lands at 100-200% |
| `ResumesAPartialDownload` | truncates the entry + rewrites the sidecar to "incomplete", then asserts a real 206 resume produces a byte-identical file |
| `FailedDownloadIsNotCached` | a 404 must throw and must leave no complete entry |

## Open

- **The hub sends no ETag on `/hf`.** Without it a resume cannot detect that the server's copy changed
  mid-download. Content is addressed by repo/revision so churn is unlikely, but an ETag (or
  `Last-Modified`) would make `If-Range` real. Hub-side change.
- **Adapter package** `SpawnDev.ILGPU.ML.WebTorrent` - not yet created. Carries `HubModelStream`, the
  `CreateFromHuggingFaceAsync` torrent branch as an extension, and the torrent-store profiler counters.
