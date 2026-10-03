using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Tensors;

namespace SpawnDev.ILGPU.ML.Pipelines;

/// <summary>
/// Video Depth Anything (Apache-2.0, ByteDance) run as a STREAM: one frame per call, with the model's temporal
/// memory - the input hidden states of its motion modules' attention blocks for the previous frames - kept on the
/// accelerator between calls. This is VDA's own streaming mode (<c>video_depth_stream.py</c>), the one that gives
/// temporally stable depth on a live video at single-frame cost.
/// </summary>
/// <remarks>
/// <para>
/// <b>The model.</b> A streaming step graph: <c>pixel_values [1,1,3,H,W]</c> and <c>cache_i [P_i, F, C_i]</c> in,
/// <c>depth [1,H,W]</c> (relative DISPARITY: high = near) and <c>new_cache_i [P_i, 1, C_i]</c> out, for every
/// <c>cache_i</c> the model declares with a matching <c>new_cache_i</c>: 8 for VDA's own streaming (each attention
/// block's input hidden state - 2 blocks in each of 4 motion modules), or 16 for the K/V-cache export
/// (tools/vda-export --kv: each block's W_k x and W_v x, so a frame projects only itself instead of all 32). <c>F</c> is the number of cached frames: 31 in steady state, and 0 for a clip's FIRST frame - with
/// no cached frames the step is exactly VDA's no-cache first-frame forward, so one graph serves both.
/// </para>
/// <para>
/// <b>The window</b> is VDA's: frame 0 stays as an anchor, plus the frame 40 back, plus the 29 most recent - and while
/// the clip is shorter than that, copies of frame 0 fill in. Position <c>k</c> of the 31 cached frames, after
/// <c>n</c> frames have followed frame 0, is frame <c>0</c> (k = 0), <c>max(0, n-40)</c> (k = 1) or
/// <c>max(0, n-30+k)</c> (k &gt;= 2). Because that is a closed form of <c>n</c>, the window is assembled ON THE GPU by
/// one gather per cache from a ring of frame slots - no per-frame host data, nothing read back.
/// </para>
/// <para>
/// A change of input size invalidates the cache (its <c>P_i</c> follow the patch grid): the stream restarts at the
/// new size, as at a new clip. Call <see cref="Reset"/> on a cut, a seek or a new video.
/// </para>
/// </remarks>
public sealed class VideoDepthAnythingStream : IDisposable
{
    /// <summary>Cached frames per step in steady state (VDA's INFER_LEN 32, minus the current frame).</summary>
    public const int CachedFrames = 31;
    /// <summary>Frames kept besides frame 0: the anchor 40 back through the newest, 41 in all.</summary>
    const int RingFrames = 41;

    readonly InferenceSession _session;
    readonly Accelerator _accelerator;
    readonly string _pixelName;
    readonly string[] _cacheNames;
    readonly string[] _newCacheNames;
    readonly int[] _channels;

    // Per cache: the frame ring [1 + RingFrames, P, C] (slot 0 = frame 0) and the assembled window [P, 31, C].
    MemoryBuffer1D<float, Stride1D.Dense>[]? _rings;
    MemoryBuffer1D<float, Stride1D.Dense>[]? _windows;
    int[]? _p;
    int _h, _w;
    // Frames run since the stream (re)started; 0 = the next call is a clip's first frame.
    long _frames;
    MemoryBuffer1D<float, Stride1D.Dense>? _empty;   // backs the 0-element first-frame caches

    Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int, int>? _gatherKernel;

    /// <summary>True when <paramref name="session"/> is a streaming step graph this class can drive.</summary>
    public static bool IsStreamingModel(InferenceSession session) => CacheNames(session).Length > 0;

    static string[] CacheNames(InferenceSession session) =>
        session.InputNames.Where(n => session.OutputNames.Contains("new_" + n)).ToArray();

    /// <param name="session">A streaming VDA step graph (see the class remarks). The stream does not own it.</param>
    public VideoDepthAnythingStream(InferenceSession session, Accelerator accelerator)
    {
        _session = session;
        _accelerator = accelerator;
        _cacheNames = CacheNames(session);
        if (_cacheNames.Length == 0)
            throw new ArgumentException("Not a streaming model: no input X has a matching output new_X.", nameof(session));
        var pixel = session.InputNames.Except(_cacheNames).ToArray();
        if (pixel.Length != 1)
            throw new ArgumentException($"Expected exactly one non-cache input, found [{string.Join(", ", pixel)}].", nameof(session));
        _pixelName = pixel[0];
        _newCacheNames = _cacheNames.Select(n => "new_" + n).ToArray();
        _channels = _cacheNames.Select(n =>
            session.InputShapes.TryGetValue(n, out var s) && s.Length == 3 && s[2] > 0 ? s[2]
            : throw new ArgumentException($"Cache input '{n}' must be rank 3 [P, F, C] with a static C.", nameof(session))).ToArray();
    }

    /// <summary>Frames run since the stream (re)started.</summary>
    public long FrameIndex => _frames;

    /// <summary>Start over: the next frame is a clip's first (cut, seek, new video).</summary>
    public void Reset() => _frames = 0;

    /// <summary>
    /// One frame: <paramref name="pixelValues"/> is the preprocessed <c>[1,1,3,H,W]</c> input (H and W multiples of 14).
    /// Returns the session's outputs - pool-rented: hand them back with <see cref="InferenceSession.ReturnOutputs"/>
    /// once consumed. Everything is enqueued; nothing is read back.
    /// </summary>
    public async Task<Dictionary<string, Tensor>> RunAsync(Tensor pixelValues)
    {
        if (pixelValues.Rank != 5) throw new ArgumentException($"pixel_values must be [1,1,3,H,W], got [{string.Join(",", pixelValues.Shape)}].");
        int h = pixelValues.Shape[3], w = pixelValues.Shape[4];
        if (h != _h || w != _w)
        {
            // New size: the cached hidden states are on another patch grid. Restart at this size.
            _frames = 0;
            _h = h; _w = w;
        }

        var inputs = new Dictionary<string, Tensor> { [_pixelName] = pixelValues };
        if (_frames == 0)
        {
            _empty ??= _accelerator.Allocate1D<float>(1);
            var p = FirstFramePatchCounts(h, w);
            for (int i = 0; i < _cacheNames.Length; i++)
                inputs[_cacheNames[i]] = new Tensor(_empty.View, new[] { p[i], 0, _channels[i] });
        }
        else
        {
            long n = _frames - 1;   // frames that have followed frame 0 before this one
            for (int i = 0; i < _cacheNames.Length; i++)
            {
                int P = _p![i], C = _channels[i];
                Gather(_rings![i].View, _windows![i].View, P, C, n);
                inputs[_cacheNames[i]] = new Tensor(_windows[i].View, new[] { P, CachedFrames, C });
            }
        }

        var outputs = await _session.RunAsync(inputs).ConfigureAwait(false);
        try
        {
            if (_frames == 0) await EnsureBuffersAsync(outputs).ConfigureAwait(false);
            // This frame's slot: frame 0 has its own; frame t >= 1 replaces t - 41, the anchor this frame just used.
            int slot = _frames == 0 ? 0 : 1 + (int)(_frames % RingFrames);
            for (int i = 0; i < _cacheNames.Length; i++)
            {
                int len = _p![i] * _channels[i];
                var nc = outputs[_newCacheNames[i]];
                if (nc.ElementCount != len)
                    throw new InvalidOperationException($"'{_newCacheNames[i]}' is [{string.Join(",", nc.Shape)}], expected [{_p[i]},1,{_channels[i]}].");
                _rings![i].View.SubView((long)slot * len, len).CopyFrom(nc.Data.SubView(0, len));
            }
        }
        catch
        {
            _session.ReturnOutputs(outputs);
            throw;
        }
        _frames++;
        return outputs;
    }

    /// <summary>
    /// The cache patch counts of a FIRST frame, whose caches are empty but must still name P (the step concatenates
    /// them with the frame's own [P, 1, C]). VDA-Small's DPT head: motion modules 0/1 sit on the 1/14 grid and the
    /// stride-2 1/28 grid, module 2 on 1/14, module 3 on the 2x-upsampled 1/7 grid - two attention blocks each.
    /// Checked against the model's own outputs right after the first frame.
    /// </summary>
    int[] FirstFramePatchCounts(int h, int w)
    {
        int ph = h / 14, pw = w / 14;
        int g14 = ph * pw, g28 = ((ph + 1) / 2) * ((pw + 1) / 2), g7 = (2 * ph) * (2 * pw);
        var perModule = new[] { g14, g28, g14, g7 };
        // Caches come module by module: 8 = one hidden state per attention block (2 per module), 16 = the K/V-cache
        // export's (W_k x, W_v x) per attention block (4 per module). Either way, module = index / (count / 4).
        int n = _cacheNames.Length;
        if (n % 4 != 0 || n == 0)
            throw new NotSupportedException($"First-frame cache geometry expects VDA's 4 motion modules; this model has {n} caches.");
        return Enumerable.Range(0, n).Select(i => perModule[i / (n / 4)]).ToArray();
    }

    async Task EnsureBuffersAsync(Dictionary<string, Tensor> firstOutputs)
    {
        var p = new int[_cacheNames.Length];
        for (int i = 0; i < p.Length; i++)
        {
            var s = firstOutputs[_newCacheNames[i]].Shape;
            if (s.Length != 3 || s[1] != 1 || s[2] != _channels[i])
                throw new InvalidOperationException($"'{_newCacheNames[i]}' is [{string.Join(",", s)}], expected [P,1,{_channels[i]}].");
            p[i] = s[0];
        }
        var expected = FirstFramePatchCounts(_h, _w);
        if (!p.SequenceEqual(expected))
            throw new InvalidOperationException($"Cache patch counts [{string.Join(",", p)}] differ from the first-frame geometry " +
                $"[{string.Join(",", expected)}] at {_w}x{_h}: not the VDA-Small head layout this stream assumes.");
        if (_p != null && _p.SequenceEqual(p)) return;
        if (_rings != null)
        {
            // A queued gather or copy may still read the old buffers (Wasm frees on dispose): drain first.
            await _accelerator.SynchronizeAsync().ConfigureAwait(false);
            DisposeBuffers();
        }
        _p = p;
        _rings = new MemoryBuffer1D<float, Stride1D.Dense>[p.Length];
        _windows = new MemoryBuffer1D<float, Stride1D.Dense>[p.Length];
        for (int i = 0; i < p.Length; i++)
        {
            _rings[i] = _accelerator.Allocate1D<float>((long)(1 + RingFrames) * p[i] * _channels[i]);
            _windows[i] = _accelerator.Allocate1D<float>((long)CachedFrames * p[i] * _channels[i]);
        }
    }

    void Gather(ArrayView1D<float, Stride1D.Dense> ring, ArrayView1D<float, Stride1D.Dense> window, int p, int c, long n)
    {
        _gatherKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, int, int, int>(GatherWindowKernel);
        // n only matters up to the window's reach (frame indices are compared, then taken mod 41): clamp it into int.
        int nn = (int)Math.Min(n, int.MaxValue / 2);
        _gatherKernel(p * CachedFrames * c, ring, window, p, c, nn);
    }

    /// <summary>window[p, k, c] = ring[slot(frame(k)), p, c], VDA's window as a closed form of n (see the class remarks).</summary>
    static void GatherWindowKernel(Index1D idx, ArrayView1D<float, Stride1D.Dense> ring, ArrayView1D<float, Stride1D.Dense> window,
        int p, int c, int n)
    {
        int ci = idx % c;
        int t = idx / c;
        int k = t % CachedFrames;
        int pi = t / CachedFrames;
        int frame = k == 0 ? 0 : k == 1 ? n - 40 : n - 30 + k;
        int slot = frame <= 0 ? 0 : 1 + frame % RingFrames;
        window[idx] = ring[(slot * p + pi) * c + ci];
    }

    void DisposeBuffers()
    {
        if (_rings != null) foreach (var b in _rings) b.Dispose();
        if (_windows != null) foreach (var b in _windows) b.Dispose();
        _rings = _windows = null;
        _p = null;
    }

    public void Dispose()
    {
        DisposeBuffers();
        _empty?.Dispose();
        _empty = null;
    }
}
