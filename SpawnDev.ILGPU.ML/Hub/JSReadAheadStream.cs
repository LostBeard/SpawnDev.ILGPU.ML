using System.Runtime.InteropServices;
using SpawnDev.SpawnJS.JSObjects;
using SpawnDev.SpawnJS.Toolbox;

namespace SpawnDev.ILGPU.ML.Hub;

/// <summary>
/// Read-ahead over a seekable JS-backed stream (<see cref="IJSReadStream"/>, e.g. the OPFS stream <see cref="ModelHub.OpenStreamFromUrlAsync"/> returns): each JS
/// read fetches a window of <see cref="WindowBytes"/> or more, and reads inside it are served from it - as JS subarrays for <see cref="ReadUint8ArrayAsync"/>
/// (the bytes stay in JS, so a weight still goes straight to the GPU) and as copies for ordinary <see cref="Stream"/> reads (the parser's small structure reads).
/// </summary>
/// <remarks>
/// MEASURED (RTX 4070, Chrome, MobileNetV2, ColdStart_MobileNetV2_CreateFromOpfsVsMemory): session creation from the OPFS stream 276-327 ms against 68-91 ms from a
/// MemoryStream of the same bytes - upload 200-218 vs 36-51, parse 38-43 vs 10-17 - because every weight, and every parser buffer refill, was its own async OPFS read
/// (a few hundred). Bones found it (2026-11-26: ~240 ms of upload in the demo page).
/// </remarks>
internal sealed class JSReadAheadStream : Stream, IJSReadStream
{
    private readonly Stream _inner;
    private readonly IJSReadStream _innerJs;
    private readonly bool _leaveOpen;
    private Uint8Array? _window;
    private long _windowStart;
    private int _windowLength;
    private long _position;

    /// <summary>The smallest JS read; a larger request reads exactly its size.</summary>
    public int WindowBytes { get; }

    /// <summary>JS reads made so far (diagnostics).</summary>
    public int JsReads { get; private set; }

    public JSReadAheadStream(Stream inner, int windowBytes = 8 * 1024 * 1024, bool leaveOpen = false)
    {
        _inner = inner ?? throw new ArgumentNullException(nameof(inner));
        _innerJs = inner as IJSReadStream ?? throw new ArgumentException("the stream must be an IJSReadStream", nameof(inner));
        if (!inner.CanSeek) throw new ArgumentException("the stream must be seekable", nameof(inner));
        WindowBytes = Math.Max(64 * 1024, windowBytes);
        _leaveOpen = leaveOpen;
        _position = inner.Position;
    }

    /// <summary>Wraps <paramref name="stream"/> when it is a seekable JS-backed stream; otherwise returns it unchanged.</summary>
    public static Stream Wrap(Stream stream) => stream is IJSReadStream && stream.CanSeek && stream is not JSReadAheadStream ? new JSReadAheadStream(stream) : stream;

    public override bool CanRead => true;
    public override bool CanSeek => true;
    public override bool CanWrite => false;
    public override long Length => _inner.Length;
    public override long Position { get => _position; set => Seek(value, SeekOrigin.Begin); }
    public bool CanReadSync => _innerJs.CanReadSync;

    public override long Seek(long offset, SeekOrigin origin)
    {
        long p = origin switch { SeekOrigin.Begin => offset, SeekOrigin.Current => _position + offset, _ => Length + offset };
        if (p < 0) throw new IOException("seek before the start of the stream");
        _position = p;
        return p;
    }

    private bool Covers(long position, int count) =>
        _window != null && position >= _windowStart && position + count <= _windowStart + _windowLength;

    private int Want(int count) => (int)Math.Min(Math.Max(count, WindowBytes), Math.Max(0, Length - _position));

    private void SetWindow(Uint8Array window, long start)
    {
        _window?.Dispose();
        _window = window;
        _windowStart = start;
        _windowLength = (int)window.Length;
        JsReads++;
    }

    private async Task<bool> EnsureAsync(int count, CancellationToken ct)
    {
        count = (int)Math.Min(count, Math.Max(0, Length - _position));
        if (count == 0) return false;
        if (Covers(_position, count)) return true;
        _inner.Position = _position;
        SetWindow(await _innerJs.ReadUint8ArrayAsync(Want(count), ct).ConfigureAwait(false), _position);
        return _windowLength > 0;
    }

    private bool Ensure(int count)
    {
        count = (int)Math.Min(count, Math.Max(0, Length - _position));
        if (count == 0) return false;
        if (Covers(_position, count)) return true;
        _inner.Position = _position;
        SetWindow(_innerJs.ReadUint8Array(Want(count)), _position);
        return _windowLength > 0;
    }

    // a view of [position, position + n) of the window (the caller owns it), or of its end when the read reached past what the source returned
    private Uint8Array Slice(int count)
    {
        int off = (int)(_position - _windowStart);
        int n = Math.Min(count, _windowLength - off);
        var view = _window!.SubArray(off, off + n);
        _position += n;
        return view;
    }

    public async Task<Uint8Array> ReadUint8ArrayAsync(int count, CancellationToken cancellationToken = default)
    {
        if (!await EnsureAsync(count, cancellationToken).ConfigureAwait(false)) return new Uint8Array(0);
        return Slice(count);
    }

    public Uint8Array ReadUint8Array(int count)
    {
        if (!Ensure(count)) return new Uint8Array(0);
        return Slice(count);
    }

    private int CopyOut(byte[] buffer, int offset, int count)
    {
        int off = (int)(_position - _windowStart);
        int n = Math.Min(count, _windowLength - off);
        if (n <= 0) return 0;
        _window!.ToArray(off, buffer, offset, n);
        _position += n;
        return n;
    }

    public override async Task<int> ReadAsync(byte[] buffer, int offset, int count, CancellationToken cancellationToken)
    {
        if (!await EnsureAsync(count, cancellationToken).ConfigureAwait(false)) return 0;
        return CopyOut(buffer, offset, count);
    }

    public override async ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default)
    {
        if (MemoryMarshal.TryGetArray<byte>(buffer, out var seg))
            return await ReadAsync(seg.Array!, seg.Offset, seg.Count, cancellationToken).ConfigureAwait(false);
        var tmp = new byte[buffer.Length];
        int n = await ReadAsync(tmp, 0, tmp.Length, cancellationToken).ConfigureAwait(false);
        tmp.AsSpan(0, n).CopyTo(buffer.Span);
        return n;
    }

    public override int Read(byte[] buffer, int offset, int count)
    {
        if (!Ensure(count)) return 0;
        return CopyOut(buffer, offset, count);
    }

    public override void Flush() { }
    public override void SetLength(long value) => throw new NotSupportedException();
    public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _window?.Dispose();
            _window = null;
            if (!_leaveOpen) _inner.Dispose();
        }
        base.Dispose(disposing);
    }
}
