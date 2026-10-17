using SpawnDev.ILGPU.ML.Hub;

namespace Trellis2Harness;

/// <summary>
/// An IModelSource over the local checkpoint cache, so the hub code path (Trellis2Weights.FromModelSource: streamed, non-seekable safetensors
/// reads) is exercised on desktop without downloading again. The stream is wrapped as NON-seekable, like an HTTP / OPFS stream can be.
/// </summary>
internal sealed class LocalFolderSource : IModelSource
{
    private static readonly Dictionary<string, string> Repos = new()
    {
        ["microsoft/TRELLIS.2-4B"] = "TRELLIS.2-4B",
        ["microsoft/TRELLIS-image-large"] = "TRELLIS-image-large",
        ["timm/vit_large_patch16_dinov3_qkvb.lvd1689m"] = "dinov3-timm-qkvb",
    };
    public Task<Stream> OpenAsync(string repoId, string filePath, CancellationToken cancellationToken = default)
        => Task.FromResult<Stream>(new ForwardOnly(File.OpenRead(Path.Combine(Gpu.WeightDir, Repos[repoId], filePath.Replace('/', Path.DirectorySeparatorChar)))));
    public async Task<byte[]> FetchBytesAsync(string repoId, string filePath, CancellationToken cancellationToken = default)
    {
        await using var s = await OpenAsync(repoId, filePath, cancellationToken);
        using var ms = new MemoryStream(); await s.CopyToAsync(ms, cancellationToken); return ms.ToArray();
    }

    private sealed class ForwardOnly(Stream inner) : Stream
    {
        public override bool CanRead => true;
        public override bool CanSeek => false;
        public override bool CanWrite => false;
        public override long Length => throw new NotSupportedException();
        public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }
        public override void Flush() { }
        public override int Read(byte[] buffer, int offset, int count) => inner.Read(buffer, offset, count);
        public override ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken ct = default) => inner.ReadAsync(buffer, ct);
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
        protected override void Dispose(bool disposing) { if (disposing) inner.Dispose(); base.Dispose(disposing); }
    }
}
