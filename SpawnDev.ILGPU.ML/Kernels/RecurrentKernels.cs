using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// The LSTM recurrence, on the accelerator.
/// </summary>
/// <remarks>
/// <para>
/// WHY: <c>LSTMOperator</c> computes on the HOST. It reads X, <c>initial_h</c> and <c>initial_c</c> back,
/// runs the recurrence in C#, and uploads Y/Y_h/Y_c - which is what made LSTM correct on six backends in
/// 5.2.3, and is fine on CUDA/OpenCL where a readback is a synchronous memcpy. It is not fine in a browser.
/// MEASURED on Silero VAD (WebGPU, `Vad_Benchmark_FrameRate`): 16 host readbacks per frame, of which
/// ELEVEN exist only to serve this operator - X, the four h/c slices, and both nodes' three outputs.
/// </para>
/// <para>
/// ⚠️ The cost is not the bytes. On WebGPU a readback is a <c>mapAsync</c>: it flushes the command encoder
/// and WAITS, so every one of them is a pipeline barrier that stops the graph's ~125 dispatches ever
/// batching into a single submission. Removing them is worth more than the transfer time they are charged.
/// </para>
/// <para>
/// SHAPE OF THE KERNEL, and why it is not the obvious one: a persistent single-workgroup kernel looping
/// over timesteps in shared memory is the textbook GPU RNN, and it CANNOT run on WebGL, where compute is
/// emulated through transform feedback with no shared memory and no barriers. So this is gather-only, one
/// thread per (batch, hidden) unit, with the time loop OUTSIDE the kernel as one dispatch per step. Those
/// dispatches are queued, never fenced, which is exactly what WebGPU is good at - the same reasoning that
/// shaped the inverted scatter kernels. Silero's sequence length is 1, so it is a single dispatch.
/// </para>
/// <para>
/// State lives in TWO SEPARATE buffers, alternated by step parity. The first attempt used one allocation
/// indexed by step - slice t in, slice t+1 out - to avoid a ping-pong. WebGPU rejected it: hIn and hOut
/// were then the same buffer, which is storage-buffer aliasing however disjoint the ranges look
/// ("binding 5 and binding 7 reference the same GPU buffer"). The ping-pong is safe here because these are
/// per-call SCRATCH, not state that persists across calls - the alternation is decided by C# on every
/// invocation, and the sequence length is fixed, so a recorded plan replays the same bindings.
/// </para>
/// <para>
/// ⚠️ The kernel writes exactly TWO buffers, and Y is filled afterwards by a native GPU-to-GPU copy of the
/// new hidden state. It originally wrote Y as a third output, which worked on CUDA/OpenCL/WebGPU and
/// silently produced ZEROS on WebGL - Y_h was correct, so the recurrence had run fine and only the third
/// buffer's write never landed. Under transform-feedback emulation the number of output buffers is not
/// free. Copying Y out costs nothing (CopyBufferToBuffer) and removes the constraint.
/// </para>
/// </remarks>
public sealed class RecurrentKernels
{
    private readonly Accelerator _accelerator;

    /// <summary>
    /// Every scalar the step needs, in one blittable struct.
    /// </summary>
    /// <remarks>
    /// Not a style choice: <c>Action&lt;&gt;</c> stops at 16 type arguments and the ten buffers plus nine
    /// offsets came to 21. Packing keeps the offsets named rather than bit-packed into an int.
    /// </remarks>
    public struct LstmStepParams
    {
        public int Batch, Hidden, InputSize;
        public int XOff, WOff, ROff, BOff, POff;
        public int HasBias, HasPeephole;
    }

    private Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,   // x
        ArrayView1D<float, Stride1D.Dense>,   // w
        ArrayView1D<float, Stride1D.Dense>,   // r
        ArrayView1D<float, Stride1D.Dense>,   // b        (raw ONNX B: Wb then Rb)
        ArrayView1D<float, Stride1D.Dense>,   // p        (peepholes)
        ArrayView1D<float, Stride1D.Dense>,   // hIn
        ArrayView1D<float, Stride1D.Dense>,   // cIn
        ArrayView1D<float, Stride1D.Dense>,   // hOut
        ArrayView1D<float, Stride1D.Dense>,   // cOut
        LstmStepParams>? _lstmStep;

    public RecurrentKernels(Accelerator accelerator) => _accelerator = accelerator;

    /// <summary>
    /// One LSTM timestep for every (batch, hidden) unit.
    /// </summary>
    /// <param name="hasBias">0 when B is absent - an empty view cannot be indexed.</param>
    /// <param name="hasPeephole">0 when P is absent.</param>
    public void LstmStep(
        ArrayView1D<float, Stride1D.Dense> x,
        ArrayView1D<float, Stride1D.Dense> w,
        ArrayView1D<float, Stride1D.Dense> r,
        ArrayView1D<float, Stride1D.Dense> b,
        ArrayView1D<float, Stride1D.Dense> p,
        ArrayView1D<float, Stride1D.Dense> hIn,
        ArrayView1D<float, Stride1D.Dense> cIn,
        ArrayView1D<float, Stride1D.Dense> hOut,
        ArrayView1D<float, Stride1D.Dense> cOut,
        LstmStepParams prm)
    {
        _lstmStep ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            LstmStepParams>(LstmStepImpl);

        _lstmStep(prm.Batch * prm.Hidden, x, w, r, b, p, hIn, cIn, hOut, cOut, prm);
    }

    /// <summary>
    /// Gates for ONE hidden unit. Mirrors <c>LSTMOperator.ExecuteCore</c> exactly, including ONNX's
    /// <c>[i, o, f, c]</c> gate order and peepholes applied to the PREVIOUS cell for i/f and the NEW cell
    /// for o - the two places a reimplementation silently drifts.
    /// </summary>
    private static void LstmStepImpl(Index1D index,
        ArrayView1D<float, Stride1D.Dense> x,
        ArrayView1D<float, Stride1D.Dense> w,
        ArrayView1D<float, Stride1D.Dense> r,
        ArrayView1D<float, Stride1D.Dense> b,
        ArrayView1D<float, Stride1D.Dense> p,
        ArrayView1D<float, Stride1D.Dense> hIn,
        ArrayView1D<float, Stride1D.Dense> cIn,
        ArrayView1D<float, Stride1D.Dense> hOut,
        ArrayView1D<float, Stride1D.Dense> cOut,
        LstmStepParams prm)
    {
        int H = prm.Hidden;
        int inputSize = prm.InputSize;
        int xOff = prm.XOff, wOff = prm.WOff, rOff = prm.ROff;
        int bOff = prm.BOff, pOff = prm.POff;

        int i = index;
        if (i >= prm.Batch * H) return;

        int bIdx = i / H;
        int hi = i - bIdx * H;

        bool hasBias = prm.HasBias != 0;
        bool hasPeep = prm.HasPeephole != 0;

        // ONNX B is [num_dir, 8*H]: Wb for the four gates then Rb for the four. The host path folds them
        // into one array up front; doing it here keeps B on the GPU and costs three adds.
        float gi = 0f, go = 0f, gf = 0f, gc = 0f;
        if (hasBias)
        {
            gi = b[bOff + hi] + b[bOff + 4 * H + hi];
            go = b[bOff + H + hi] + b[bOff + 5 * H + hi];
            gf = b[bOff + 2 * H + hi] + b[bOff + 6 * H + hi];
            gc = b[bOff + 3 * H + hi] + b[bOff + 7 * H + hi];
        }

        // Xt * W^T. W is [num_dir, 4*H, inputSize] with gate-major rows, matching the host's g*inputSize.
        int wi = wOff + hi * inputSize;
        int wo = wOff + (H + hi) * inputSize;
        int wf = wOff + (2 * H + hi) * inputSize;
        int wc = wOff + (3 * H + hi) * inputSize;
        int xBase = xOff + bIdx * inputSize;
        for (int k = 0; k < inputSize; k++)
        {
            float xv = x[xBase + k];
            gi += xv * w[wi + k];
            go += xv * w[wo + k];
            gf += xv * w[wf + k];
            gc += xv * w[wc + k];
        }

        // Ht-1 * R^T.
        int ri = rOff + hi * H;
        int ro = rOff + (H + hi) * H;
        int rf = rOff + (2 * H + hi) * H;
        int rc = rOff + (3 * H + hi) * H;
        int hBase = bIdx * H;
        for (int k = 0; k < H; k++)
        {
            float hv = hIn[hBase + k];
            gi += hv * r[ri + k];
            go += hv * r[ro + k];
            gf += hv * r[rf + k];
            gc += hv * r[rc + k];
        }

        float cPrev = cIn[hBase + hi];
        if (hasPeep)
        {
            gi += p[pOff + hi] * cPrev;
            gf += p[pOff + 2 * H + hi] * cPrev;
        }

        float it = 1f / (1f + XMath.Exp(-gi));
        float ft = 1f / (1f + XMath.Exp(-gf));
        float newC = ft * cPrev + it * XMath.Tanh(gc);

        // The output gate's peephole reads the NEW cell, not the previous one.
        if (hasPeep) go += p[pOff + H + hi] * newC;
        float ot = 1f / (1f + XMath.Exp(-go));
        float newH = ot * XMath.Tanh(newC);

        cOut[hBase + hi] = newC;
        hOut[hBase + hi] = newH;
    }
}
