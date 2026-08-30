using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using System.Numerics;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// Fused linear layer: Output = Activation(MatMul(Input, Weights) + Bias)
/// Combines matrix multiplication, bias addition, and activation into a single kernel dispatch.
/// Eliminates 2 out of 3 global memory write cycles compared to separate ops.
///
/// This is the single highest-impact optimization for transformer inference.
/// A 12-layer model saves ~24 memory round-trips by fusing linear layers.
///
/// Performance: for the large FFN sizes (e.g. GPT-2's 768→3072 / 3072→768) and any backend that can
/// launch a 256-thread group, <see cref="Forward"/> routes None / GELU to a REGISTER-BLOCKED
/// shared-memory GEMM (64×64 output tile, 4×4 per thread) with the bias + activation fused into the
/// register write-back — the same kernel shape as <c>RegisterBlockedMatMul</c>, so the MatMul runs at
/// full throughput, not the per-element rate. Small matrices, the WebGL / CPU group-cap backends, and the
/// activations without a register-blocked variant (ReLU/SiLU/Sigmoid/Tanh) use the per-element kernels
/// below. The register-blocked GELU is the erf approximation (A&amp;S 5-term) — bit-faithful to
/// <c>ElementWiseKernels.GELUImpl</c> and the per-element path here, so it preserves the ORT-matched logits.
/// </summary>
public class FusedLinearKernel
{
    // Register-blocked tile geometry — must match RegisterBlockedMatMul (BLOCK=16 → 256-thread group,
    // REG=4 per thread, 64×64 output tile). Kept in sync so the fused GEMM has identical launch shape.
    private const int RbBlock = 16;
    private const int RbReg = 4;
    private const int RbTile = RbBlock * RbReg; // 64

    private readonly Accelerator _accelerator;

    // Register-blocked fused kernel (None + GELU): activation code 0 = linear (bias only), 2 = erf-GELU.
    private Action<KernelConfig, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        int, int, int, int, int>? _fusedRegBlockedKernel;

    // One kernel per activation type to avoid branching inside the hot loop
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        int, int, int>? _fusedLinearReluKernel;

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        int, int, int>? _fusedLinearGeluKernel;

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        int, int, int>? _fusedLinearSiluKernel;

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        int, int, int>? _fusedLinearNoneKernel;

    public FusedLinearKernel(Accelerator accelerator) => _accelerator = accelerator;

    /// <summary>
    /// Fused linear: output = activation(input @ weights + bias)
    /// Input: [M, K], Weights: [K, N], Bias: [N], Output: [M, N]
    /// </summary>
    public void Forward(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> weights,
        ArrayView1D<float, Stride1D.Dense> bias,
        ArrayView1D<float, Stride1D.Dense> output,
        int M, int K, int N,
        FusedActivation activation = FusedActivation.None)
    {
        // ── Fast path: register-blocked shared-memory GEMM with fused bias + activation ──
        // None / ReLU / GELU / SiLU have a register-blocked variant (the decoder-FFN + SD-ResNet/FFN lever).
        // Routes exactly like RegisterBlockedMatMul: large enough to fill a 64×64 tile AND the backend can launch
        // a 256-thread group (rules out WebGL / the CPU 64-thread cap → they fall through to the per-element path).
        if ((activation == FusedActivation.None || activation == FusedActivation.ReLU
                || activation == FusedActivation.GELU || activation == FusedActivation.SiLU)
            && M >= RbTile && N >= RbTile
            && _accelerator.MaxNumThreadsPerGroup >= RbBlock * RbBlock)
        {
            int activationCode = activation switch // matches the per-element path + FusedActivate
            {
                FusedActivation.ReLU => 1,
                FusedActivation.GELU => 2,
                FusedActivation.SiLU => 5,
                _ => 0,
            };
            _fusedRegBlockedKernel ??= _accelerator.LoadStreamKernel<
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                int, int, int, int, int>(FusedRegBlockedLinearActivation);

            int numTilesM = (M + RbTile - 1) / RbTile;
            int numTilesN = (N + RbTile - 1) / RbTile;
            var config = new KernelConfig(new Index1D(numTilesM * numTilesN), new Index1D(RbBlock * RbBlock));
            _fusedRegBlockedKernel(config, input, weights, bias, output, M, K, N, numTilesN, activationCode);
            return;
        }

        switch (activation)
        {
            case FusedActivation.ReLU:
                _fusedLinearReluKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
                    ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                    ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                    int, int, int>(FusedLinearReluImpl);
                _fusedLinearReluKernel(M * N, input, weights, bias, output, M, K, N);
                break;

            case FusedActivation.GELU:
                _fusedLinearGeluKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
                    ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                    ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                    int, int, int>(FusedLinearGeluImpl);
                _fusedLinearGeluKernel(M * N, input, weights, bias, output, M, K, N);
                break;

            case FusedActivation.SiLU:
                _fusedLinearSiluKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
                    ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                    ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                    int, int, int>(FusedLinearSiluImpl);
                _fusedLinearSiluKernel(M * N, input, weights, bias, output, M, K, N);
                break;

            default: // None
                _fusedLinearNoneKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
                    ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                    ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                    int, int, int>(FusedLinearNoneImpl);
                _fusedLinearNoneKernel(M * N, input, weights, bias, output, M, K, N);
                break;
        }
    }

    // ── Kernel implementations ──
    // Each output element: sum(input[row] * weights[col]) + bias[col] + activation
    // One thread per output element. Sequential over K (dot product).

    private static void FusedLinearNoneImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> weights,
        ArrayView1D<float, Stride1D.Dense> bias,
        ArrayView1D<float, Stride1D.Dense> output,
        int M, int K, int N)
    {
        int row = idx / N;
        int col = idx % N;

        float sum = 0f;
        for (int k = 0; k < K; k++)
            sum += input[row * K + k] * weights[k * N + col];

        output[idx] = sum + bias[col];
    }

    private static void FusedLinearReluImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> weights,
        ArrayView1D<float, Stride1D.Dense> bias,
        ArrayView1D<float, Stride1D.Dense> output,
        int M, int K, int N)
    {
        int row = idx / N;
        int col = idx % N;

        float sum = 0f;
        for (int k = 0; k < K; k++)
            sum += input[row * K + k] * weights[k * N + col];

        float val = sum + bias[col];
        output[idx] = val > 0f ? val : 0f;
    }

    private static void FusedLinearGeluImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> weights,
        ArrayView1D<float, Stride1D.Dense> bias,
        ArrayView1D<float, Stride1D.Dense> output,
        int M, int K, int N)
    {
        int row = idx / N;
        int col = idx % N;

        float sum = 0f;
        for (int k = 0; k < K; k++)
            sum += input[row * K + k] * weights[k * N + col];

        float x = sum + bias[col];
        // GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2))) — must match ElementWiseKernels.GELUImpl
        if (x > 10f) { output[idx] = x; return; }
        if (x < -10f) { output[idx] = 0f; return; }
        const float INV_SQRT2 = 0.7071067811865475f;
        float z = x * INV_SQRT2;
        float az = z < 0f ? -z : z;
        const float p = 0.3275911f;
        const float a1 = 0.254829592f;
        const float a2 = -0.284496736f;
        const float a3 = 1.421413741f;
        const float a4 = -1.453152027f;
        const float a5 = 1.061405429f;
        float t = 1f / (1f + p * az);
        float t2 = t * t;
        float t3 = t2 * t;
        float t4 = t3 * t;
        float t5 = t4 * t;
        float erfAbs = 1f - (a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * MathF.Exp(-az * az);
        float erf = z < 0f ? -erfAbs : erfAbs;
        output[idx] = 0.5f * x * (1f + erf);
    }

    private static void FusedLinearSiluImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> weights,
        ArrayView1D<float, Stride1D.Dense> bias,
        ArrayView1D<float, Stride1D.Dense> output,
        int M, int K, int N)
    {
        int row = idx / N;
        int col = idx % N;

        float sum = 0f;
        for (int k = 0; k < K; k++)
            sum += input[row * K + k] * weights[k * N + col];

        float x = sum + bias[col];
        // SiLU = x * sigmoid(x)
        output[idx] = x / (1f + MathF.Exp(-x));
    }

    // ── Native low-precision weight path (bf16 / fp16 / FP8) ──
    // A weight loaded NATIVE low-p (no f32 upcast — gpt-oss attn/output projections, any bf16/fp16 linear)
    // carries its data in the typed low-p view; its float Data view is EMPTY. The float kernels above would
    // read out of bounds, so this generic path reads the weight in its native type and converts to float
    // in-register (PrecisionConvert) with fused bias + activation — no f32 weight temp (Rule 4 no-upcast).
    // Mirrors MatMulKernel.MatMulLowPWeight<T>. Weight layout is [K,N] (FuseLinearLayers excludes transB),
    // identical to the float per-element kernels. One thread per output element; fp32 accumulate.
    private readonly Dictionary<Type, object> _fusedLinearLowPKernels = new();
    // Register-blocked counterpart (large M,N on a 256-thread backend) — one compiled kernel per concrete T.
    private readonly Dictionary<Type, object> _fusedRegBlockedLowPKernels = new();

    /// <summary>Fused linear with a NATIVE low-precision weight (ILGPU.Half / BFloat16 / Float8E4M3 / Float8E5M2):
    /// Output = Activation(Input·W + Bias), W read native and converted to float in-register (no f32 weight temp).
    /// Supports None / ReLU / GELU / SiLU (the activations the per-element float kernels implement); Sigmoid and
    /// Tanh fall to None, matching the float path's switch.</summary>
    public void ForwardLowP<T>(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<T, Stride1D.Dense> weights,
        ArrayView1D<float, Stride1D.Dense> bias,
        ArrayView1D<float, Stride1D.Dense> output,
        int M, int K, int N, FusedActivation activation)
        where T : unmanaged, INumber<T>
    {
        int actCode = activation switch
        {
            FusedActivation.ReLU => 1,
            FusedActivation.GELU => 2,
            FusedActivation.SiLU => 5,
            _ => 0, // None (and Sigmoid/Tanh, which the float per-element switch also routes to None)
        };

        // Large matrices on a 256-thread backend: register-blocked low-p path — the same 64×64-tile / 4×4-register
        // GEMM as the fp32 reg-blocked FusedLinear, with the weight decoded to float ONCE on the shared-mem load
        // (PrecisionConvert) and bias+activation fused into the write-back (16 results/thread vs 1). This is the
        // tiled throughput SD's fused fp16 linears forfeited on the per-element kernel. Small / CPU / WebGL fall
        // through to the per-element kernel (same gate as the fp32 reg-blocked path).
        if (M >= RbTile && N >= RbTile && _accelerator.MaxNumThreadsPerGroup >= RbBlock * RbBlock)
        {
            if (!_fusedRegBlockedLowPKernels.TryGetValue(typeof(T), out var rk))
                _fusedRegBlockedLowPKernels[typeof(T)] = rk = _accelerator.LoadStreamKernel<
                    ArrayView1D<float, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>,
                    ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                    int, int, int, int, int>(FusedRegBlockedLowPActivation<T>);
            int numTilesM = (M + RbTile - 1) / RbTile;
            int numTilesN = (N + RbTile - 1) / RbTile;
            var rbConfig = new KernelConfig(new Index1D(numTilesM * numTilesN), new Index1D(RbBlock * RbBlock));
            ((Action<KernelConfig, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                int, int, int, int, int>)rk)(rbConfig, input, weights, bias, output, M, K, N, numTilesN, actCode);
            return;
        }

        if (!_fusedLinearLowPKernels.TryGetValue(typeof(T), out var k))
            _fusedLinearLowPKernels[typeof(T)] = k = _accelerator.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                int, int, int, int>(FusedLinearLowPImpl<T>);
        ((Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            int, int, int, int>)k)(M * N, input, weights, bias, output, M, K, N, actCode);
    }

    private static void FusedLinearLowPImpl<T>(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<T, Stride1D.Dense> weights,
        ArrayView1D<float, Stride1D.Dense> bias,
        ArrayView1D<float, Stride1D.Dense> output,
        int M, int K, int N, int activation)
        where T : unmanaged, INumber<T>
    {
        int row = idx / N;
        int col = idx % N;
        if (row >= M) return;

        float sum = 0f;
        for (int k = 0; k < K; k++)
            sum += input[row * K + k] * PrecisionConvert.ConvertToSingle(weights[k * N + col]);

        float x = sum + bias[col];
        if (activation == 1) { output[idx] = x > 0f ? x : 0f; return; }          // ReLU
        if (activation == 2) { output[idx] = GeluErf(x); return; }                // GELU (erf approx)
        if (activation == 5) { output[idx] = x / (1f + MathF.Exp(-x)); return; }  // SiLU
        output[idx] = x;                                                          // None
    }

    /// <summary>GELU erf approximation (A&amp;S 5-term), bit-faithful to <see cref="FusedLinearGeluImpl"/> /
    /// ElementWiseKernels.GELUImpl (clamp tails: x&gt;10 → x, x&lt;-10 → 0). Shared by the native low-p fused path.</summary>
    private static float GeluErf(float x)
    {
        if (x > 10f) return x;
        if (x < -10f) return 0f;
        const float INV_SQRT2 = 0.7071067811865475f;
        float z = x * INV_SQRT2;
        float az = z < 0f ? -z : z;
        const float p = 0.3275911f;
        const float a1 = 0.254829592f, a2 = -0.284496736f,
                    a3 = 1.421413741f, a4 = -1.453152027f, a5 = 1.061405429f;
        float t = 1f / (1f + p * az);
        float t2 = t * t, t3 = t2 * t, t4 = t3 * t, t5 = t4 * t;
        float erfAbs = 1f - (a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * MathF.Exp(-az * az);
        float erf = z < 0f ? -erfAbs : erfAbs;
        return 0.5f * x * (1f + erf);
    }

    // ── Register-blocked fused GEMM (the performant None/GELU path) ──
    // Mirrors RegisterBlockedMatMul.RegBlockedImpl (BLOCK=16, REG=4, 64×64 tile, each of 256 threads
    // computes a 4×4 register block) and fuses bias-Add + activation into the write-back. Verified
    // bit-equivalent to the per-element path against a CPU reference on WebGPU/Wasm/CUDA/OpenCL
    // (SpawnDev.ILGPU FusedFFN_RegBlocked* tests); CPU + WebGL are routed away from it in Forward.

    // Fused bias-add + activation, shared by all 16 register write-backs. Codes match the per-element kernels
    // (FusedLinearLowPImpl): 0 = linear (bias only), 1 = ReLU, 2 = GELU (A&S 5-term erf), 5 = SiLU (x·sigmoid).
    private static float FusedActivate(float acc, float bias, int activation)
    {
        float v = acc + bias;
        if (activation == 1) return v > 0f ? v : 0f;            // ReLU
        if (activation == 5) return v / (1f + XMath.Exp(-v));   // SiLU = x·sigmoid(x)
        if (activation == 2)
        {
            // GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2))) — matches ElementWiseKernels.GELUImpl + the
            // per-element FusedLinearGeluImpl above (clamp tails: x>10 → x, x<-10 → 0).
            float x = v;
            if (x > 10f) return x;
            if (x < -10f) return 0f;
            const float INV_SQRT2 = 0.7071067811865475f;
            float z = x * INV_SQRT2;
            float az = z < 0f ? -z : z;
            const float p = 0.3275911f;
            const float a1 = 0.254829592f, a2 = -0.284496736f,
                        a3 = 1.421413741f, a4 = -1.453152027f, a5 = 1.061405429f;
            float t = 1f / (1f + p * az);
            float t2 = t * t, t3 = t2 * t, t4 = t3 * t, t5 = t4 * t;
            float erfAbs = 1f - (a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * XMath.Exp(-az * az);
            float erf = z < 0f ? -erfAbs : erfAbs;
            return 0.5f * x * (1f + erf);
        }
        return v; // activation == 0: linear (bias only)
    }

    private static void FusedRegBlockedLinearActivation(
        ArrayView1D<float, Stride1D.Dense> X,
        ArrayView1D<float, Stride1D.Dense> W,
        ArrayView1D<float, Stride1D.Dense> Bias,
        ArrayView1D<float, Stride1D.Dense> Y,
        int M, int K, int N, int numTilesN, int activation)
    {
        const int BLOCK = 16;
        const int REG = 4;
        const int TILE = BLOCK * REG; // 64
        var aTile = SharedMemory.Allocate<float>(TILE * BLOCK); // 64×16
        var bTile = SharedMemory.Allocate<float>(BLOCK * TILE); // 16×64

        int tileIdx = Grid.IdxX;
        int tileRow = tileIdx / numTilesN;
        int tileCol = tileIdx % numTilesN;

        int localIdx = Group.IdxX;
        int threadRow = localIdx / BLOCK; // 0..15
        int threadCol = localIdx % BLOCK; // 0..15

        float c00 = 0, c01 = 0, c02 = 0, c03 = 0;
        float c10 = 0, c11 = 0, c12 = 0, c13 = 0;
        float c20 = 0, c21 = 0, c22 = 0, c23 = 0;
        float c30 = 0, c31 = 0, c32 = 0, c33 = 0;

        int numKTiles = (K + BLOCK - 1) / BLOCK;
        for (int t = 0; t < numKTiles; t++)
        {
            for (int r = 0; r < REG; r++)
            {
                int aRow = tileRow * TILE + threadRow * REG + r;
                int aCol = t * BLOCK + threadCol;
                int sIdx = (threadRow * REG + r) * BLOCK + threadCol;
                aTile[sIdx] = (aRow < M && aCol < K) ? X[aRow * K + aCol] : 0f;
            }
            for (int r = 0; r < REG; r++)
            {
                int bRow = t * BLOCK + threadRow;
                int bCol = tileCol * TILE + threadCol * REG + r;
                int sIdx = threadRow * TILE + threadCol * REG + r;
                bTile[sIdx] = (bRow < K && bCol < N) ? W[bRow * N + bCol] : 0f;
            }
            Group.Barrier();

            for (int k = 0; k < BLOCK; k++)
            {
                float a0 = aTile[(threadRow * REG + 0) * BLOCK + k];
                float a1 = aTile[(threadRow * REG + 1) * BLOCK + k];
                float a2 = aTile[(threadRow * REG + 2) * BLOCK + k];
                float a3 = aTile[(threadRow * REG + 3) * BLOCK + k];
                float b0 = bTile[k * TILE + threadCol * REG + 0];
                float b1 = bTile[k * TILE + threadCol * REG + 1];
                float b2 = bTile[k * TILE + threadCol * REG + 2];
                float b3 = bTile[k * TILE + threadCol * REG + 3];
                c00 += a0 * b0; c01 += a0 * b1; c02 += a0 * b2; c03 += a0 * b3;
                c10 += a1 * b0; c11 += a1 * b1; c12 += a1 * b2; c13 += a1 * b3;
                c20 += a2 * b0; c21 += a2 * b1; c22 += a2 * b2; c23 += a2 * b3;
                c30 += a3 * b0; c31 += a3 * b1; c32 += a3 * b2; c33 += a3 * b3;
            }
            Group.Barrier();
        }

        int baseRow = tileRow * TILE + threadRow * REG;
        int baseCol = tileCol * TILE + threadCol * REG;

        // Fused bias-Add + activation in the write-back (bias indexed by output column).
        if (baseRow + 0 < M)
        {
            if (baseCol + 0 < N) Y[(baseRow + 0) * N + baseCol + 0] = FusedActivate(c00, Bias[baseCol + 0], activation);
            if (baseCol + 1 < N) Y[(baseRow + 0) * N + baseCol + 1] = FusedActivate(c01, Bias[baseCol + 1], activation);
            if (baseCol + 2 < N) Y[(baseRow + 0) * N + baseCol + 2] = FusedActivate(c02, Bias[baseCol + 2], activation);
            if (baseCol + 3 < N) Y[(baseRow + 0) * N + baseCol + 3] = FusedActivate(c03, Bias[baseCol + 3], activation);
        }
        if (baseRow + 1 < M)
        {
            if (baseCol + 0 < N) Y[(baseRow + 1) * N + baseCol + 0] = FusedActivate(c10, Bias[baseCol + 0], activation);
            if (baseCol + 1 < N) Y[(baseRow + 1) * N + baseCol + 1] = FusedActivate(c11, Bias[baseCol + 1], activation);
            if (baseCol + 2 < N) Y[(baseRow + 1) * N + baseCol + 2] = FusedActivate(c12, Bias[baseCol + 2], activation);
            if (baseCol + 3 < N) Y[(baseRow + 1) * N + baseCol + 3] = FusedActivate(c13, Bias[baseCol + 3], activation);
        }
        if (baseRow + 2 < M)
        {
            if (baseCol + 0 < N) Y[(baseRow + 2) * N + baseCol + 0] = FusedActivate(c20, Bias[baseCol + 0], activation);
            if (baseCol + 1 < N) Y[(baseRow + 2) * N + baseCol + 1] = FusedActivate(c21, Bias[baseCol + 1], activation);
            if (baseCol + 2 < N) Y[(baseRow + 2) * N + baseCol + 2] = FusedActivate(c22, Bias[baseCol + 2], activation);
            if (baseCol + 3 < N) Y[(baseRow + 2) * N + baseCol + 3] = FusedActivate(c23, Bias[baseCol + 3], activation);
        }
        if (baseRow + 3 < M)
        {
            if (baseCol + 0 < N) Y[(baseRow + 3) * N + baseCol + 0] = FusedActivate(c30, Bias[baseCol + 0], activation);
            if (baseCol + 1 < N) Y[(baseRow + 3) * N + baseCol + 1] = FusedActivate(c31, Bias[baseCol + 1], activation);
            if (baseCol + 2 < N) Y[(baseRow + 3) * N + baseCol + 2] = FusedActivate(c32, Bias[baseCol + 2], activation);
            if (baseCol + 3 < N) Y[(baseRow + 3) * N + baseCol + 3] = FusedActivate(c33, Bias[baseCol + 3], activation);
        }
    }

    // Register-blocked fused linear with a NATIVE low-p weight W. Identical to FusedRegBlockedLinearActivation
    // except W is read as T and decoded to float ONCE as it enters the (float) shared tile — the one line that
    // differs — so the decode is amortized over the 4× register reuse and the hot math is byte-identical.
    private static void FusedRegBlockedLowPActivation<T>(
        ArrayView1D<float, Stride1D.Dense> X,
        ArrayView1D<T, Stride1D.Dense> W,
        ArrayView1D<float, Stride1D.Dense> Bias,
        ArrayView1D<float, Stride1D.Dense> Y,
        int M, int K, int N, int numTilesN, int activation)
        where T : unmanaged, INumber<T>
    {
        const int BLOCK = 16;
        const int REG = 4;
        const int TILE = BLOCK * REG; // 64
        var aTile = SharedMemory.Allocate<float>(TILE * BLOCK);
        var bTile = SharedMemory.Allocate<float>(BLOCK * TILE);

        int tileIdx = Grid.IdxX;
        int tileRow = tileIdx / numTilesN;
        int tileCol = tileIdx % numTilesN;
        int localIdx = Group.IdxX;
        int threadRow = localIdx / BLOCK;
        int threadCol = localIdx % BLOCK;

        float c00 = 0, c01 = 0, c02 = 0, c03 = 0;
        float c10 = 0, c11 = 0, c12 = 0, c13 = 0;
        float c20 = 0, c21 = 0, c22 = 0, c23 = 0;
        float c30 = 0, c31 = 0, c32 = 0, c33 = 0;

        int numKTiles = (K + BLOCK - 1) / BLOCK;
        for (int t = 0; t < numKTiles; t++)
        {
            for (int r = 0; r < REG; r++)
            {
                int aRow = tileRow * TILE + threadRow * REG + r;
                int aCol = t * BLOCK + threadCol;
                int sIdx = (threadRow * REG + r) * BLOCK + threadCol;
                aTile[sIdx] = (aRow < M && aCol < K) ? X[aRow * K + aCol] : 0f;
            }
            for (int r = 0; r < REG; r++)
            {
                int bRow = t * BLOCK + threadRow;
                int bCol = tileCol * TILE + threadCol * REG + r;
                int sIdx = threadRow * TILE + threadCol * REG + r;
                bTile[sIdx] = (bRow < K && bCol < N) ? PrecisionConvert.ConvertToSingle(W[bRow * N + bCol]) : 0f;
            }
            Group.Barrier();

            for (int k = 0; k < BLOCK; k++)
            {
                float a0 = aTile[(threadRow * REG + 0) * BLOCK + k];
                float a1 = aTile[(threadRow * REG + 1) * BLOCK + k];
                float a2 = aTile[(threadRow * REG + 2) * BLOCK + k];
                float a3 = aTile[(threadRow * REG + 3) * BLOCK + k];
                float b0 = bTile[k * TILE + threadCol * REG + 0];
                float b1 = bTile[k * TILE + threadCol * REG + 1];
                float b2 = bTile[k * TILE + threadCol * REG + 2];
                float b3 = bTile[k * TILE + threadCol * REG + 3];
                c00 += a0 * b0; c01 += a0 * b1; c02 += a0 * b2; c03 += a0 * b3;
                c10 += a1 * b0; c11 += a1 * b1; c12 += a1 * b2; c13 += a1 * b3;
                c20 += a2 * b0; c21 += a2 * b1; c22 += a2 * b2; c23 += a2 * b3;
                c30 += a3 * b0; c31 += a3 * b1; c32 += a3 * b2; c33 += a3 * b3;
            }
            Group.Barrier();
        }

        int baseRow = tileRow * TILE + threadRow * REG;
        int baseCol = tileCol * TILE + threadCol * REG;
        if (baseRow + 0 < M)
        {
            if (baseCol + 0 < N) Y[(baseRow + 0) * N + baseCol + 0] = FusedActivate(c00, Bias[baseCol + 0], activation);
            if (baseCol + 1 < N) Y[(baseRow + 0) * N + baseCol + 1] = FusedActivate(c01, Bias[baseCol + 1], activation);
            if (baseCol + 2 < N) Y[(baseRow + 0) * N + baseCol + 2] = FusedActivate(c02, Bias[baseCol + 2], activation);
            if (baseCol + 3 < N) Y[(baseRow + 0) * N + baseCol + 3] = FusedActivate(c03, Bias[baseCol + 3], activation);
        }
        if (baseRow + 1 < M)
        {
            if (baseCol + 0 < N) Y[(baseRow + 1) * N + baseCol + 0] = FusedActivate(c10, Bias[baseCol + 0], activation);
            if (baseCol + 1 < N) Y[(baseRow + 1) * N + baseCol + 1] = FusedActivate(c11, Bias[baseCol + 1], activation);
            if (baseCol + 2 < N) Y[(baseRow + 1) * N + baseCol + 2] = FusedActivate(c12, Bias[baseCol + 2], activation);
            if (baseCol + 3 < N) Y[(baseRow + 1) * N + baseCol + 3] = FusedActivate(c13, Bias[baseCol + 3], activation);
        }
        if (baseRow + 2 < M)
        {
            if (baseCol + 0 < N) Y[(baseRow + 2) * N + baseCol + 0] = FusedActivate(c20, Bias[baseCol + 0], activation);
            if (baseCol + 1 < N) Y[(baseRow + 2) * N + baseCol + 1] = FusedActivate(c21, Bias[baseCol + 1], activation);
            if (baseCol + 2 < N) Y[(baseRow + 2) * N + baseCol + 2] = FusedActivate(c22, Bias[baseCol + 2], activation);
            if (baseCol + 3 < N) Y[(baseRow + 2) * N + baseCol + 3] = FusedActivate(c23, Bias[baseCol + 3], activation);
        }
        if (baseRow + 3 < M)
        {
            if (baseCol + 0 < N) Y[(baseRow + 3) * N + baseCol + 0] = FusedActivate(c30, Bias[baseCol + 0], activation);
            if (baseCol + 1 < N) Y[(baseRow + 3) * N + baseCol + 1] = FusedActivate(c31, Bias[baseCol + 1], activation);
            if (baseCol + 2 < N) Y[(baseRow + 3) * N + baseCol + 2] = FusedActivate(c32, Bias[baseCol + 2], activation);
            if (baseCol + 3 < N) Y[(baseRow + 3) * N + baseCol + 3] = FusedActivate(c33, Bias[baseCol + 3], activation);
        }
    }
}

/// <summary>
/// Fused Scaled MatMul: Output = MatMul(A, B^T) * scale
/// Used in attention: scores = (Q * K^T) / sqrt(d_k)
/// Fuses the transpose, MatMul, and scaling into one kernel dispatch.
/// Eliminates 2 dispatches (transpose + scale) from the attention hot path.
/// </summary>
public class FusedScaledMatMulKernel
{
    private readonly Accelerator _accelerator;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, int, int, int, float>? _kernel;

    public FusedScaledMatMulKernel(Accelerator accelerator) => _accelerator = accelerator;

    /// <summary>
    /// Compute output[i,j] = sum_k(A[i,k] * B[j,k]) * scale
    /// Note: B is accessed as transposed (B[j,k] not B[k,j]).
    /// A: [M, K], B: [N, K] (stored row-major, accessed as B^T), Output: [M, N]
    /// </summary>
    public void Forward(
        ArrayView1D<float, Stride1D.Dense> A,
        ArrayView1D<float, Stride1D.Dense> B,
        ArrayView1D<float, Stride1D.Dense> output,
        int M, int K, int N, float scale)
    {
        _kernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, int, int, int, float>(ScaledMatMulTransBImpl);
        _kernel(M * N, A, B, output, M, K, N, scale);
    }

    private static void ScaledMatMulTransBImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> A,
        ArrayView1D<float, Stride1D.Dense> B,
        ArrayView1D<float, Stride1D.Dense> output,
        int M, int K, int N, float scale)
    {
        int row = idx / N;
        int col = idx % N;

        float sum = 0f;
        for (int k = 0; k < K; k++)
            sum += A[row * K + k] * B[col * K + k]; // B transposed: B[col, k]

        output[idx] = sum * scale;
    }
}

/// <summary>
/// Activation function for fused linear layers.
/// </summary>
public enum FusedActivation
{
    None,
    ReLU,
    GELU,
    SiLU,
    Sigmoid,
    Tanh,
}
