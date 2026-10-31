using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Trellis2;

namespace SpawnDev.ILGPU.ML.Ltx;

using Buf = MemoryBuffer1D<float, Stride1D.Dense>;

/// <summary>
/// The temporaries of the LTX VAE encoder / decoder and the latent upscaler. Small buffers are freed in batches after a fence (on WebGPU a buffer must outlive the dispatches that
/// read it). Large activations (1M floats and up) are recycled by exact size instead: a dropped one is handed to the next request of its size at once, which is safe without a fence
/// because every backend runs the dispatches of one accelerator in order; buffers of a size no longer asked for are freed (after a fence) when a new large size is allocated.
/// This keeps the peak at the activations actually live (three at the decoder's full resolution, not five) without paying an allocation and a fence per convolution.
/// Nothing is ever written into a recycled buffer from the host.
/// </summary>
internal sealed class LtxTemps : IDisposable
{
    private readonly Accelerator _acc;
    private readonly List<Buf> _live = new();
    private readonly List<Buf> _drop = new();
    private readonly Dictionary<long, Stack<Buf>> _free = new();
    private const long Large = 1 << 20;

    public LtxTemps(Accelerator accelerator) => _acc = accelerator;

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
        if (b.Length >= Large)
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
        foreach (var b in _drop) { _live.Remove(b); b.Dispose(); }
        _drop.Clear();
    }

    public void Dispose()
    {
        foreach (var b in _live.ToArray()) b.Dispose();
        _live.Clear(); _drop.Clear();
        foreach (var s in _free.Values) while (s.Count > 0) s.Pop().Dispose();
        _free.Clear();
    }
}
