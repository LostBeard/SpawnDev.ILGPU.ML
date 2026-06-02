using System.IO;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace SpawnDev.ILGPU.ML.Onnx;

/// <summary>
/// Streaming protobuf wire-format reader over a <see cref="Stream"/>. Reads the small structural
/// fields (tags, varints, names, shapes) while letting the caller SKIP past bulk fields (a tensor's
/// raw_data weight blob) without ever materializing them. On a seekable stream a skip is a real
/// <see cref="Stream.Seek"/>; on a forward-only stream it is a bounded read-and-discard. Either way
/// the whole file is never held in memory — a multi-GB model is inspected from a few hundred KB of
/// structure.
///
/// Lengths are tracked as <see cref="long"/> so models larger than 2 GB parse correctly.
/// </summary>
internal sealed class StreamProtoReader
{
    private readonly Stream _s;
    private readonly CancellationToken _ct;
    private readonly byte[] _buf;
    private int _bufPos;
    private int _bufLen;

    /// <summary>Absolute number of bytes consumed from the stream so far.</summary>
    public long Position { get; private set; }

    public StreamProtoReader(Stream s, CancellationToken ct, int bufferSize = 64 * 1024)
    {
        _s = s;
        _ct = ct;
        _buf = new byte[bufferSize];
    }

    /// <summary>Ensures the buffer has at least one byte available; refills from the stream if empty.
    /// Returns false at end of stream.</summary>
    private async ValueTask<bool> EnsureAsync()
    {
        if (_bufPos < _bufLen) return true;
        _bufPos = 0;
        _bufLen = await _s.ReadAsync(_buf.AsMemory(0, _buf.Length), _ct).ConfigureAwait(false);
        return _bufLen > 0;
    }

    /// <summary>True if more data remains (refilling the buffer if needed).</summary>
    public ValueTask<bool> HasMoreAsync() => EnsureAsync();

    private async ValueTask<int> ReadByteAsync()
    {
        if (!await EnsureAsync().ConfigureAwait(false)) return -1;
        Position++;
        return _buf[_bufPos++];
    }

    /// <summary>Read a base-128 varint (unsigned, up to 64 bits).</summary>
    public async ValueTask<ulong> ReadVarintAsync()
    {
        ulong result = 0;
        int shift = 0;
        while (true)
        {
            int b = await ReadByteAsync().ConfigureAwait(false);
            if (b < 0) throw new EndOfStreamException("Unexpected end of stream while reading varint.");
            result |= (ulong)(b & 0x7F) << shift;
            if ((b & 0x80) == 0) return result;
            shift += 7;
            if (shift > 63) throw new InvalidOperationException("Varint too long.");
        }
    }

    /// <summary>Read a field tag, decomposed into field number + wire type.</summary>
    public async ValueTask<(int Field, int Wire)> ReadTagAsync()
    {
        ulong tag = await ReadVarintAsync().ConfigureAwait(false);
        return ((int)(tag >> 3), (int)(tag & 0x07));
    }

    /// <summary>Read exactly <paramref name="len"/> bytes into a new array. For SMALL structural
    /// fields only (names, packed dims, value-info/opset submessages) — never for weight blobs.</summary>
    public async ValueTask<byte[]> ReadBytesAsync(int len)
    {
        var outBuf = new byte[len];
        int got = 0;
        while (got < len)
        {
            if (!await EnsureAsync().ConfigureAwait(false))
                throw new EndOfStreamException($"Expected {len} bytes, stream ended {len - got} short.");
            int take = Math.Min(len - got, _bufLen - _bufPos);
            Array.Copy(_buf, _bufPos, outBuf, got, take);
            _bufPos += take;
            got += take;
            Position += take;
        }
        return outBuf;
    }

    /// <summary>Read a length-delimited field as a UTF-8 string.</summary>
    public async ValueTask<string> ReadStringAsync()
    {
        int len = checked((int)await ReadVarintAsync().ConfigureAwait(false));
        var b = await ReadBytesAsync(len).ConfigureAwait(false);
        return Encoding.UTF8.GetString(b);
    }

    /// <summary>
    /// Advance <paramref name="len"/> bytes WITHOUT materializing them. Consumes any buffered bytes
    /// first, then <see cref="Stream.Seek"/>s when the stream is seekable, otherwise reads and discards
    /// in buffer-sized chunks. This is how a tensor's raw_data weight blob is bypassed.
    /// </summary>
    public async ValueTask SkipAsync(long len)
    {
        if (len < 0) throw new ArgumentOutOfRangeException(nameof(len));
        if (len == 0) return;

        // Consume from the in-memory buffer first.
        int inBuf = _bufLen - _bufPos;
        if (inBuf > 0)
        {
            long take = Math.Min(len, inBuf);
            _bufPos += (int)take;
            Position += take;
            len -= take;
        }
        if (len == 0) return;

        // Buffer is now empty. Seek if we can; otherwise read-and-discard.
        if (_s.CanSeek)
        {
            _s.Seek(len, SeekOrigin.Current);
            Position += len;
            return;
        }
        while (len > 0)
        {
            int take = (int)Math.Min(len, _buf.Length);
            int n = await _s.ReadAsync(_buf.AsMemory(0, take), _ct).ConfigureAwait(false);
            if (n == 0) throw new EndOfStreamException("Stream ended while skipping a field.");
            Position += n;
            len -= n;
        }
    }

    /// <summary>Skip an entire field based on its wire type (varint / I64 / length-delimited / I32).</summary>
    public async ValueTask SkipFieldAsync(int wire)
    {
        switch (wire)
        {
            case 0: await ReadVarintAsync().ConfigureAwait(false); break;            // VARINT
            case 1: await SkipAsync(8).ConfigureAwait(false); break;                 // I64
            case 2:                                                                   // LEN
                long len = (long)await ReadVarintAsync().ConfigureAwait(false);
                await SkipAsync(len).ConfigureAwait(false);
                break;
            case 5: await SkipAsync(4).ConfigureAwait(false); break;                 // I32
            default: throw new InvalidOperationException($"Unknown protobuf wire type: {wire}");
        }
    }
}
