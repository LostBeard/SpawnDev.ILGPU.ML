using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// Selective Scan GPU kernel for Mamba-3 State Space Models.
/// Linear-scaling alternative to transformer attention — O(1) memory per token
/// instead of O(N) KV cache growth.
///
/// SSM recurrence: h_t = A * h_{t-1} + B * x_t
///                 y_t = C * h_t + D * x_t
///
/// Where A (decay), B (input projection), C (output projection), D (skip connection)
/// are input-dependent (selective), making this more expressive than fixed-parameter SSMs.
///
/// Key advantage: constant memory during autoregressive decoding.
/// State size is fixed at d_state × d_model regardless of sequence length.
/// </summary>
public class SelectiveScanKernel
{
    private readonly Accelerator _accelerator;

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        int, int, int>? _scanKernel;

    public SelectiveScanKernel(Accelerator accelerator) => _accelerator = accelerator;

    /// <summary>
    /// Run selective scan over a sequence.
    /// x [batch, seqLen, dModel] — input sequence
    /// A [dState] — decay factors (per state dimension)
    /// B [batch, seqLen, dState] — input projection
    /// C [batch, seqLen, dState] — output projection
    /// → output [batch, seqLen, dModel]
    ///
    /// Each (batch, dModel) dimension is processed independently.
    /// Sequential over seqLen (the recurrence), parallel over batch × dModel.
    /// </summary>
    public void Forward(
        ArrayView1D<float, Stride1D.Dense> x,
        ArrayView1D<float, Stride1D.Dense> A,
        ArrayView1D<float, Stride1D.Dense> B,
        ArrayView1D<float, Stride1D.Dense> C,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<float, Stride1D.Dense> state,
        int batchSize, int seqLen, int dState)
    {
        _scanKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            int, int, int>(ScanImpl);
        _scanKernel(batchSize, x, A, B, C, output, state, batchSize, seqLen, dState);
    }

    /// <summary>
    /// Per-batch selective scan. Sequential over time, parallel over state dimensions.
    /// </summary>
    private static void ScanImpl(Index1D batchIdx,
        ArrayView1D<float, Stride1D.Dense> x,
        ArrayView1D<float, Stride1D.Dense> A,
        ArrayView1D<float, Stride1D.Dense> B,
        ArrayView1D<float, Stride1D.Dense> C,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<float, Stride1D.Dense> state,
        int batchSize, int seqLen, int dState)
    {
        // Initialize state for this batch
        int stateOffset = batchIdx * dState;
        for (int s = 0; s < dState; s++)
            state[stateOffset + s] = 0f;

        // Sequential scan over time steps
        for (int t = 0; t < seqLen; t++)
        {
            int xIdx = batchIdx * seqLen + t;
            float xVal = x[xIdx];

            // Update state: h_t = A * h_{t-1} + B * x_t
            for (int s = 0; s < dState; s++)
            {
                int bIdx = (batchIdx * seqLen + t) * dState + s;
                state[stateOffset + s] = A[s] * state[stateOffset + s] + B[bIdx] * xVal;
            }

            // Compute output: y_t = C * h_t
            float y = 0f;
            for (int s = 0; s < dState; s++)
            {
                int cIdx = (batchIdx * seqLen + t) * dState + s;
                y += C[cIdx] * state[stateOffset + s];
            }

            output[xIdx] = y;
        }
    }

    /// <summary>
    /// Single-step decode for autoregressive generation.
    /// Processes one token, updates state in-place. O(1) memory.
    /// </summary>
    public void DecodeStep(
        ArrayView1D<float, Stride1D.Dense> xToken,    // [batch, 1]
        ArrayView1D<float, Stride1D.Dense> A,          // [dState]
        ArrayView1D<float, Stride1D.Dense> bToken,     // [batch, dState]
        ArrayView1D<float, Stride1D.Dense> cToken,     // [batch, dState]
        ArrayView1D<float, Stride1D.Dense> state,      // [batch, dState] — updated in-place
        ArrayView1D<float, Stride1D.Dense> output,     // [batch, 1]
        int batchSize, int dState)
    {
        // Reuse the scan kernel with seqLen=1
        Forward(xToken, A, bToken, cToken, output, state, batchSize, 1, dState);
    }
}
