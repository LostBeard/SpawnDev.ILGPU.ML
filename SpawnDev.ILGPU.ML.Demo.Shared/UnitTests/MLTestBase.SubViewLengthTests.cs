using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// ML kernels whose RESULT depends on a view argument's in-kernel length, run on host sub-views that do not start on a 256-byte boundary.
/// SpawnDev.ILGPU 5.4.27 and earlier read such a length too large on WebGPU (by (start % 256) / 4 for 32-bit elements; Data, 2026-11-11, fixed in 5.4.28):
/// indexing was right, only <c>view.Length</c> / <c>IntLength</c> inside the kernel was wrong. A scan of ML's kernels found 22 in-kernel length reads: 18 are
/// bounds guards that cannot change a result for valid shapes, Conv1D / ConvTranspose1D's <c>bias.Length &gt; 0</c> never sees a zero-length bias (a missing
/// bias is a zero buffer), and these two use the length as DATA: TurboQuant's sign flip (<c>idx % signs.IntLength</c>) and the face letterbox
/// (<c>plane = blob.Length / 3</c>). Their callers pass whole buffers today; these tests pin the behaviour for a sliced view. Red on WebGPU with ILGPU 5.4.27.
/// </summary>
public abstract partial class MLTestBase
{
    [TestMethod]
    public async Task TurboQuant_SignFlip_UnalignedSignsSubView() => await RunTest(async accelerator =>
    {
        const int n = 37, start = 3, count = 200;          // signs start 12 bytes into their buffer: not 256-byte aligned
        var parent = new int[start + n + 16];
        for (int i = 0; i < parent.Length; i++) parent[i] = 1;   // outside the view: "flip", so a too-long length shows as wrong signs
        var rng = new Random(7);
        for (int i = 0; i < n; i++) parent[start + i] = rng.Next(2);
        var x = new float[count];
        for (int i = 0; i < count; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
        using var signsBuf = accelerator.Allocate1D(parent);
        using var inBuf = accelerator.Allocate1D(x);
        using var outBuf = accelerator.Allocate1D<float>(count);
        using var tq = new TurboQuantKernels(accelerator);
        tq.SignFlip(inBuf.View, outBuf.View, signsBuf.View.SubView(start, n), count);
        await accelerator.SynchronizeAsync();
        var got = await outBuf.CopyToHostAsync<float>(0, count);
        int bad = 0, firstBad = -1;
        for (int i = 0; i < count; i++)
        {
            float want = parent[start + i % n] != 0 ? -x[i] : x[i];
            if (BitConverter.SingleToInt32Bits(got[i]) != BitConverter.SingleToInt32Bits(want)) { bad++; if (firstBad < 0) firstBad = i; }
        }
        if (bad > 0) throw new Exception($"SignFlip over a {n}-element signs sub-view at element {start}: {bad} of {count} outputs wrong (first at {firstBad}): the kernel's signs.IntLength is not {n}");
        return $"{count} outputs, signs period {n} at element {start}: exact";
    });

    [TestMethod]
    public async Task FaceStages_Letterbox_UnalignedBlobSubView() => await RunTest(async accelerator =>
    {
        const int width = 37, height = 29, inW = 32, inH = 32, start = 5;   // the blob starts 20 bytes into its buffer
        int blobLen = 3 * inW * inH;
        var rng = new Random(11);
        var px = new int[width * height];
        for (int i = 0; i < px.Length; i++) px[i] = rng.Next() | unchecked((int)0xFF000000);
        using var pxBuf = accelerator.Allocate1D(px);
        using var whole = accelerator.Allocate1D<float>(blobLen);
        using var parent = accelerator.Allocate1D<float>(start + blobLen + 64);
        var stages = new FaceStages(accelerator);
        var (s1, x1, y1) = stages.Letterbox(pxBuf.View, width, height, PixelOrder.Bgra, whole.View, inW, inH);
        _ = stages.Letterbox(pxBuf.View, width, height, PixelOrder.Bgra, parent.View.SubView(start, blobLen), inW, inH);
        await accelerator.SynchronizeAsync();
        var a = await whole.CopyToHostAsync<float>(0, blobLen);
        var b = await parent.CopyToHostAsync<float>(start, blobLen);
        int bad = 0, firstBad = -1;
        for (int i = 0; i < blobLen; i++)
            if (BitConverter.SingleToInt32Bits(a[i]) != BitConverter.SingleToInt32Bits(b[i])) { bad++; if (firstBad < 0) firstBad = i; }
        if (bad > 0) throw new Exception($"Letterbox into a blob sub-view at element {start}: {bad} of {blobLen} values differ from the whole-buffer letterbox (first at {firstBad}, plane {inW * inH}): the kernel's blob.Length / 3 is not the plane size");
        return $"{width}x{height} -> {inW}x{inH} planar, sub-view at element {start}: bit-identical to the whole buffer (scale {s1:F3}, pad {x1},{y1})";
    });
}
