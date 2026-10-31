using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Trellis2;

namespace SpawnDev.ILGPU.ML;

using Buf = MemoryBuffer1D<float, Stride1D.Dense>;

/// <summary>
/// The temporaries of the video VAEs (LTX encoder / decoder / latent upscaler, the Wan 2.2 VAE). Small buffers are freed in batches after a fence (on WebGPU a buffer must outlive the dispatches that
/// read it). Large activations (1M floats and up) are recycled by exact size instead: a dropped one is handed to the next request of its size at once, which is safe without a fence
/// because every backend runs the dispatches of one accelerator in order; buffers of a size no longer asked for are freed (after a fence) when a new large size is allocated.
/// This keeps the peak at the activations actually live (three at the decoder's full resolution, not five) without paying an allocation and a fence per convolution.
/// Nothing may be written into a recycled buffer from the host or by a host-side fill (on WebGPU those land before dispatches still queued): use <see cref="GetFresh"/> for such a buffer.
/// </summary>
internal sealed class ActivationPool : IDisposable
{
    private readonly Accelerator _acc;
    private readonly List<Buf> _live = new();
    private readonly List<Buf> _drop = new();
    private readonly Dictionary<long, Stack<Buf>> _free = new();
    private const long Large = 1 << 20;

    private readonly HashSet<Buf> _fresh = new();

    public ActivationPool(Accelerator accelerator) => _acc = accelerator;

    /// <summary>A buffer that is never recycled (freed after a fence like a small one): for one the host fills or clears.</summary>
    public Buf GetFresh(long n)
    {
        var b = _acc.Allocate1D<float>(n);
        _live.Add(b); _fresh.Add(b);
        return b;
    }

    public Buf Get(long n)
    {
        if (n >= Large)
        {
            if (_free.TryGetValue(n, out var st) && st.Count > 0) { var r = st.Pop(); _live.Add(r); return r; }
            if (_free.Values.Any(s => s.Count > 0))
            {
                _acc.Fence();
                foreach (var s in _free.Values) while (s.Count > 0) s.Pop().Dispose();
                _free.Clear();
            }
        }
        var b = _acc.Allocate1D<float>(n);
        _live.Add(b);
        return b;
    }

    public void Drop(Buf b)
    {
        if (b.Length >= Large && !_fresh.Contains(b))
        {
            _live.Remove(b);
            if (!_free.TryGetValue(b.Length, out var st)) _free[b.Length] = st = new Stack<Buf>();
            st.Push(b);
        }
        else _drop.Add(b);
    }

    public void Flush()
    {
        if (_drop.Count == 0) return;
        _acc.Fence();
        foreach (var b in _drop) { _live.Remove(b); _fresh.Remove(b); b.Dispose(); }
        _drop.Clear();
    }

    public void Dispose()
    {
        foreach (var b in _live.ToArray()) b.Dispose();
        _live.Clear(); _drop.Clear(); _fresh.Clear();
        foreach (var s in _free.Values) while (s.Count > 0) s.Pop().Dispose();
        _free.Clear();
    }
}
