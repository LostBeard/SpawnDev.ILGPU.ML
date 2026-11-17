using System.IO.Compression;

namespace SpawnDev.ILGPU.ML.AceStep;

/// <summary>
/// ACE-Step 1.5's silence latent: the source latents (and timbre reference) of text-to-music, from ACE-Step's own MIT checkpoint
/// (<c>ACE-Step/Ace-Step1.5</c>, <c>acestep-v15-turbo/silence_latent.pt</c>: a torch tensor [1, 64, 15000], 600 s at 25 frames per second). Read without torch: a .pt file is
/// a zip whose <c>*/data/0</c> entry is the tensor's raw little-endian float32 storage.
/// <para>ComfyUI's template carries its own table instead (4 lead-in frames, then one row repeated, on a 4-decimal grid; GPL, never copied here): the MIT tensor matches its
/// first 4 frames to 4 decimals and is 2.2e-3 from it over 250 frames. AceStep_Graph_MitSilence_VsComfyUi measures what that does to a whole run.</para>
/// </summary>
public static class AceStepSilenceLatent
{
    public const int Channels = 64;

    /// <summary>The tensor's values, channel-major [64, frames], from a silence_latent.pt stream.</summary>
    public static (float[] Values, int Frames) ReadPt(Stream pt)
    {
        using var zip = new ZipArchive(pt, ZipArchiveMode.Read, leaveOpen: true);
        var entry = zip.Entries.FirstOrDefault(e => e.FullName.EndsWith("/data/0", StringComparison.Ordinal))
            ?? throw new InvalidDataException("not a torch .pt file with one storage (no */data/0 entry)");
        using var s = entry.Open();
        using var ms = new MemoryStream();
        s.CopyTo(ms);
        var bytes = ms.ToArray();
        if (bytes.Length % (4 * Channels) != 0) throw new InvalidDataException($"{bytes.Length} bytes is not a [64, frames] float32 tensor");
        var v = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, v, 0, bytes.Length);
        if (!BitConverter.IsLittleEndian) for (int i = 0; i < v.Length; i++) v[i] = BitConverter.Int32BitsToSingle(System.Buffers.Binary.BinaryPrimitives.ReverseEndianness(BitConverter.SingleToInt32Bits(v[i])));
        return (v, v.Length / Channels);
    }

    /// <summary>The first <paramref name="frames"/> frames, time-major [frames, 64] (<see cref="AceStepGenerator.GenerateLatentAsync"/>'s silence input).</summary>
    public static float[] TimeMajor(float[] values, int totalFrames, int frames)
    {
        if (frames > totalFrames) throw new ArgumentOutOfRangeException(nameof(frames), $"the silence latent holds {totalFrames} frames");
        var o = new float[frames * Channels];
        for (int c = 0; c < Channels; c++) for (int t = 0; t < frames; t++) o[t * Channels + c] = values[(long)c * totalFrames + t];
        return o;
    }
}
