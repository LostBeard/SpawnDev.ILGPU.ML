using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.WebGPU;

namespace SpawnDev.ILGPU.ML;

/// <summary>
/// Multi-Head Self-Attention helper kernels:
///   SplitHeads: [T, 3*C] → Q[H,T,D], K[H,T,D], V[H,T,D]
///   MergeHeads: [H,T,D] → [T, C]
///
/// These are auto-grouped (no shared memory) to avoid the WGSL redeclaration bug.
/// The actual Q×K^T and scores×V multiplications use the batched MatMulKernel.
///
/// Layout: QKV from MatMul is [T, 3*C] where 3*C = [q0..q383, k0..k383, v0..v383] per token.
/// Within each 384-dim block: [head0_d0..d63, head1_d0..d63, ..., head5_d0..d63].
/// So qkv[t, h*D + d] = Q for token t, head h, dim d (offset 0).
/// And qkv[t, C + h*D + d] = K, qkv[t, 2*C + h*D + d] = V.
///
/// Future home: SpawnDev.ILGPU.ML
/// </summary>
public class AttentionKernels
{
    private const int C = TransformerBlock.C;     // 384
    private const int H = TransformerBlock.H;     // 6
    private const int D = TransformerBlock.D;     // 64

    private readonly WebGPUAccelerator _accelerator;

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int>? _splitHeadsKernel;

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, int>? _mergeHeadsKernel;

    public AttentionKernels(WebGPUAccelerator accelerator) => _accelerator = accelerator;

    /// <summary>
    /// Split QKV [T, 3*C] into Q[H*T*D], K[H*T*D], V[H*T*D] each as contiguous [H, T, D].
    /// One thread per element of one output buffer (total = H*T*D per buffer).
    /// </summary>
    private static void SplitHeadsImpl(
        Index1D idx,
        ArrayView1D<float, Stride1D.Dense> qkv,  // [T * 3 * C]
        ArrayView1D<float, Stride1D.Dense> Q,     // [H * T * D]
        ArrayView1D<float, Stride1D.Dense> K,     // [H * T * D]
        ArrayView1D<float, Stride1D.Dense> V,     // [H * T * D]
        int T)
    {
        // idx ranges over H * T * D
        int d = idx % D;
        int t = (idx / D) % T;
        int h = idx / (D * T);

        // Source: qkv[t * 3*C + component*C + h*D + d]
        int qkvBase = t * 3 * C;
        Q[idx] = qkv[qkvBase + h * D + d];           // Q offset = 0
        K[idx] = qkv[qkvBase + C + h * D + d];       // K offset = C
        V[idx] = qkv[qkvBase + 2 * C + h * D + d];   // V offset = 2*C
    }

    /// <summary>
    /// Merge attention output [H, T, D] back to [T, C].
    /// Inverse of SplitHeads: output[t * C + h * D + d] = input[h * T * D + t * D + d].
    /// One thread per element (T * C total).
    /// </summary>
    private static void MergeHeadsImpl(
        Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,   // [H * T * D]
        ArrayView1D<float, Stride1D.Dense> output,   // [T * C]
        int T)
    {
        // idx ranges over T * C
        int c = idx % C;       // channel (0..383)
        int t = idx / C;       // token (0..T-1)
        int h = c / D;         // head (0..5)
        int d = c % D;         // dim within head (0..63)

        output[idx] = input[h * T * D + t * D + d];
    }

    /// <summary>Split QKV into separate Q, K, V tensors with heads as batch dim.</summary>
    public void SplitHeads(
        ArrayView1D<float, Stride1D.Dense> qkv,
        ArrayView1D<float, Stride1D.Dense> Q,
        ArrayView1D<float, Stride1D.Dense> K,
        ArrayView1D<float, Stride1D.Dense> V,
        int T)
    {
        EnsureLoaded();
        _splitHeadsKernel!(H * T * D, qkv, Q, K, V, T);
    }

    /// <summary>Merge multi-head output back to [T, C] layout.</summary>
    public void MergeHeads(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        int T)
    {
        EnsureLoaded();
        _mergeHeadsKernel!(T * C, input, output, T);
    }

    private void EnsureLoaded()
    {
        var accelerator = _accelerator;
        _splitHeadsKernel ??= accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            int>(SplitHeadsImpl);
        _mergeHeadsKernel ??= accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            int>(MergeHeadsImpl);
    }
}
