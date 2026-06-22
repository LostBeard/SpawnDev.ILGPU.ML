using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using SpawnDev.ILGPU.ML.GGUF;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// Fused weight dequantization inside MatMul — dequantize GGUF-quantized weights
/// as they load into registers, never expanding the full weight matrix.
///
/// Standard path: dequantize ALL weights (2.3GB → 4.6GB) → MatMul
/// Fused path: load quantized block → dequantize in register → accumulate → next block
///
/// Memory bandwidth saved: weights stay compressed in GPU memory.
/// Only expand to FP32 in registers during the actual computation.
///
/// WEIGHT ORIENTATION CONTRACT: the quantized bytes are the GGUF tensor's raw storage,
/// which for a linear weight is [N rows][K contiguous] (ggml ne = [K, N], ne0 fastest).
/// The kernels read it that way — output[m,n] = Σ_k input[m,k] · W[n,k] — which IS the
/// ONNX MatMul with B declared [K, N]. Do NOT "fix" the indexing to read [K rows][N]:
/// the transposed read is the contract that lets raw GGUF bytes serve as MatMul B.
///
/// BLOCK LAYOUTS are exact ports of ggml-quants.c dequantize_row_* (fetched + verified
/// 2026-06-11, see _DevComms seven P1 thread). The element ORDER inside a block is part
/// of the format: Q4_0/Q4_K take low nibbles of a byte run first, then high nibbles —
/// an interleaved read permutes the weights and produces silent garbage. The CPU
/// reference (GGUFModel.Dequantize*) implements the same layouts; the unit tests assert
/// GPU == CPU per type.
///
/// Supported here: Q4_0 (18B/32), Q8_0 (34B/32), Q4_K (144B/256), Q6_K (210B/256) —
/// the gemma4 Q4_K_M mix plus the common legacy types. Anything else must be
/// CPU-dequantized to F32 at load (see GGUFGraphBuilder.ExtractWeight); the loader
/// gates on <see cref="Supports"/> so an unsupported type can never reach Forward.
///
/// All backends use ONE kernel per type reading bytes from int32 words (Cast&lt;byte,int&gt;):
/// browser backends require it (ArrayView&lt;byte&gt; WGSL transpilation reads packed words),
/// and on desktop the extra shift/mask is noise next to memory bandwidth. One
/// implementation per type means one place to be wrong. NOTE: callers must upload
/// quantized buffers padded to a 4-byte multiple or Cast truncates the tail bytes
/// (InferenceSession.CreateFromGGUF does).
/// </summary>
public class FusedDequantMatMul : IDisposable
{
    private readonly Accelerator _accelerator;

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>>? _kernelQ4_0, _kernelQ8_0, _kernelQ4_K, _kernelQ6_K, _kernelMXFP4;

    // ── M=1 (GEMV) coalesced path ──
    // At seq=1 decode, M=1, so the general one-thread-per-output-element kernel above launches only N
    // threads, each STREAMING an entire weight row — consecutive threads read consecutive rows (strided
    // by bytesPerRow), so the warp's loads are uncoalesced (~32x bandwidth waste), and N threads with no
    // K-parallelism leaves the GPU under-occupied. MEASURED: this is 96.9% of gemma4:12b decode time
    // (~25ms per MLP down-proj at ~0.26% of card bandwidth). The GEMV kernels below assign a GROUP of
    // GemvGroupSize threads to ONE output column n; thread tid strides k=tid,tid+G,... so consecutive
    // lanes read consecutive k = consecutive weight bytes (coalesced), then a shared-memory tree reduction
    // sums the per-thread partials. Same per-type decode (DecodeQ4KElement etc.) — only the parallel shape
    // changes — so the M>1 oracle correctness carries over. Dispatched only when M==1; M>1 keeps the GEMM.
    // Group size is 64 = the LOWEST max-group-per-dim across our backends (the ILGPU CPU accelerator caps
    // groupDim at 64; WebGPU/WebGL allow 256, CUDA 1024). A larger group would throw
    // "Invalid group dimensions (128,...) exceeds maximum (64,64,64)" on CPU. 64 = 2 warps, still fully
    // coalesced per warp and ample K-parallelism; must stay a power of two for the tree reduction.
    private const int GemvGroupSize = 64;

    // DIAGNOSTIC TOGGLE (env GGUF_GEMV_OFF=1): force M==1 onto the per-element M*N kernel instead of
    // the shared-memory/barrier GEMV. A/B switch for isolating the M=1 GEMV as a suspect in the CPU
    // KV-decode non-determinism investigation (2026-06-15). Read once; zero cost in production.
    internal static readonly bool ForcePerElementGemv =
        Environment.GetEnvironmentVariable("GGUF_GEMV_OFF") == "1";

    private Action<KernelConfig, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>>? _gemvQ4_K, _gemvQ6_K, _gemvQ8_0, _gemvQ4_0, _gemvMXFP4, _gemvWarpQ4_K, _gemvWarpQ6_K;

    // Warp-cooperative GEMV (opt-in GGUF_GEMV_V2=1): the default GEMV pays two Group.Barrier() per super-block
    // (scale-decode + before-reuse) plus a shared-mem tree reduction — those syncs cap M=1 decode at ~10% of the
    // card's bandwidth. The warp kernel shares sub-block scales + reduces via Warp.Shuffle (warp-synchronous, NO
    // barrier, NO shared mem). Q4_K first (the qwen decode weight). Gated to warp>=32 GPUs (CUDA/OpenCL); CPU/Wasm
    // (small warp) keep the portable shared-mem kernel. Bit-identical math (same DecodeQ4KNibble + folded scales).
    public static bool EnableWarpGemv =
        Environment.GetEnvironmentVariable("GGUF_GEMV_V2") == "1";

    // dp4a int8-activation decode GEMV (opt-in GGUF_GEMV_DP4A=1): the llama.cpp/Ollama MMVQ technique. The
    // activation vector is quantized to int8 per 32-block (block_q8_1: int8 quants + scale d + s=d·Σq) once per
    // matmul, then the dot runs in the INTEGER domain via dp4a (4x int8 MAC/instr) — 8 float FMAs/word become
    // 2 dp4a, freeing the issue slots so the 32-bit weight loads saturate bandwidth (the path past our warp
    // GEMV's ~26%). NUMERICS: int8-approximate (the activation quant), NOT float-exact — same approximation
    // Ollama uses; validate vs the int8-activation reference / Ollama oracle, not the float GEMV. CUDA only
    // (dp4a is a CUDA inline-PTX intrinsic). Q4_K first (qwen's main decode weight).
    public static bool EnableDp4aGemv =
        Environment.GetEnvironmentVariable("GGUF_GEMV_DP4A") == "1";

    // Cached int8-quantized-activation temp buffers, per K (a model uses a few distinct K). Member-owned (never
    // method-local) so a pending dispatch never references a freed buffer (CLAUDE.md "Never Dispose Before Flush").
    private readonly Dictionary<int, MemoryBuffer1D<int, Stride1D.Dense>> _actQs = new();
    private readonly Dictionary<int, MemoryBuffer1D<float, Stride1D.Dense>> _actDs = new();
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, int>? _quantActKernel;
    private Action<KernelConfig, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>>? _gemvDp4aQ4_K, _gemvDp4aQ6_K;

    // Multi-row dequant GEMM for M>1 (prefill): dequant each weight element ONCE, reuse across GemmMTile rows —
    // kills the O(M) redundant dequant of the general per-element kernel (the prefill bottleneck). Output is
    // BIT-IDENTICAL to the per-element kernel (a faster MatMul, not a semantic change); A/B-VERIFIED on
    // qwen2.5-coder:7b Q4_K_M (CUDA, 50-tok prompt) — token+logit identical, prefill 34.0s→2.0s (~16.6x).
    // DEFAULT ON (2026-06-22): the full 6-backend sweep is GREEN — PMT FusedDequantMatMul_{MultiRow,RegBlocked}
    // _MatchesOracle_{Q4_K,Q6_K} 56/56 across CPU/CUDA/OpenCL/WebGPU/WebGL/Wasm vs the ggml CPU reference.
    // Env GGUF_GEMM_MR=0 forces it OFF (per-element fallback — the A/B diagnostic). Q4_K/Q6_K only; other
    // types' M>1 still use the per-element kernel; browser GPU keeps per-element by design (see dispatch).
    public static bool EnableMultiRowGemm =
        Environment.GetEnvironmentVariable("GGUF_GEMM_MR") != "0";
    private Action<KernelConfig, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>>? _gemmMRQ4_K, _gemmMRQ6_K, _rbQ4_K, _rbQ6_K;

    // Params buffers cached per (M,K,N). Never disposed mid-session: a WebGPU dispatch
    // that referenced the buffer may still be pending in the command encoder, and
    // disposing before the flush makes the GPU read freed memory (see CLAUDE.md
    // "Never Dispose Buffers Before Flush"). A model uses a handful of distinct shapes.
    private readonly Dictionary<(int M, int K, int N), MemoryBuffer1D<int, Stride1D.Dense>> _paramsBufs = new();

    public FusedDequantMatMul(Accelerator accelerator) => _accelerator = accelerator;

    /// <summary>Quantization types this kernel can dequantize in-register. The GGUF
    /// loader routes only these to the fused path; everything else dequantizes to F32
    /// on the CPU at load time.</summary>
    public static bool Supports(GGMLType type) =>
        type is GGMLType.Q4_0 or GGMLType.Q8_0 or GGMLType.Q4_K or GGMLType.Q6_K or GGMLType.MXFP4;

    /// <summary>Elements per quantization block for a supported type.</summary>
    private static int BlockElements(GGMLType type) =>
        type is GGMLType.Q4_K or GGMLType.Q6_K ? 256 : 32;

    /// <summary>
    /// MatMul with a GGUF-quantized weight matrix.
    /// input [M, K] (float) × weight (raw GGUF blocks, [N rows][K contiguous]) → output [M, N].
    /// </summary>
    public void Forward(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<byte, Stride1D.Dense> weightQuant,
        ArrayView1D<float, Stride1D.Dense> output,
        int M, int K, int N, GGMLType type)
    {
        if (!Supports(type))
            throw new NotSupportedException(
                $"FusedDequantMatMul does not support GGML type {type}. The loader must " +
                "CPU-dequantize this tensor to F32 (gate on FusedDequantMatMul.Supports).");
        int blockElems = BlockElements(type);
        if (K % blockElems != 0)
            throw new ArgumentException(
                $"K={K} is not a multiple of the {type} block size {blockElems} — not a valid " +
                "GGUF quantized tensor row.");

        if (!_paramsBufs.TryGetValue((M, K, N), out var paramsBuf))
        {
            paramsBuf = _accelerator.Allocate1D(new[] { M, K, N });
            _paramsBufs[(M, K, N)] = paramsBuf;
        }

        // All backends read bytes from packed int words (see class doc). The upload
        // must be padded to a 4-byte multiple or the tail bytes vanish in the Cast.
        var intView = weightQuant.Cast<byte, int>();

        // M==1 GEMV: coalesced group-per-column path (see field comment). Per-type; types without a
        // GEMV kernel yet fall through to the general M*N kernel below (correct, just not coalesced).
        //
        // The GEMV uses SharedMemory.Allocate + Group.Barrier (a tree reduction). Verified CORRECT on
        // CPU / CUDA / OpenCL / Wasm (oracle-green) AND now WebGPU too (4.12.1-local.2 fixed the WGSL
        // grid-stride emitter bug). But both browser-GPU backends are EXCLUDED for different reasons:
        //  - WebGL: NO workgroup shared memory / barriers (GLSL ES 3.0 Transform-Feedback vertex path) -
        //    a hard capability wall; the GLSL codegen would throw UnsupportedKernelFeatureException.
        //  - WebGPU: PERF, not correctness. The shared-mem/barrier cooperative GEMV is now correct on
        //    WebGPU but ~75x SLOWER than the per-element fallback there (MEASURED 2026-06-13, M=1
        //    K=4096 N=8192 Q4_K, consistent across per-iter-sync / batched-same-output / batched-diff-
        //    output: WebGPU GEMV ~530 ms/dispatch vs WebGL per-element ~7 ms and CUDA GEMV ~1 ms). The
        //    workgroup-reduction maps catastrophically onto Tint/Dawn. Until that's fixed, M==1 on
        //    WebGPU stays on the per-element kernel below. Tracked with Geordi (WGSL workgroup perf).
        // Both fall through to the general per-element M*N kernel below (correct; the pre-GEMV path).
        bool gpuBrowser = _accelerator.AcceleratorType == AcceleratorType.WebGL
                       || _accelerator.AcceleratorType == AcceleratorType.WebGPU;
        if (M == 1 && !gpuBrowser && !ForcePerElementGemv)
        {
            var gemvConfig = new KernelConfig(N, GemvGroupSize);
            // dp4a int8-activation Q4_K GEMV: quantize the activation to int8 (block_q8_1) once, then dot in the
            // integer domain via dp4a (the llama.cpp MMVQ path). CUDA only (dp4a inline-PTX intrinsic).
            if (EnableDp4aGemv && type == GGMLType.Q4_K && _accelerator.AcceleratorType == AcceleratorType.Cuda)
            {
                int nBlk = K / 32;
                if (!_actQs.TryGetValue(K, out var qsBuf)) { qsBuf = _accelerator.Allocate1D<int>(nBlk * 8); _actQs[K] = qsBuf; }
                if (!_actDs.TryGetValue(K, out var dsBuf)) { dsBuf = _accelerator.Allocate1D<float>(nBlk * 2); _actDs[K] = dsBuf; }
                _quantActKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
                    ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int>(QuantizeActQ8_1Impl);
                _quantActKernel(nBlk, input, qsBuf.View, dsBuf.View, nBlk);
                _gemvDp4aQ4_K ??= _accelerator.LoadStreamKernel<ArrayView1D<int, Stride1D.Dense>,
                    ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
                    ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>(GemvDp4aQ4_KImpl);
                _gemvDp4aQ4_K(new KernelConfig(N, _accelerator.WarpSize), qsBuf.View, dsBuf.View, intView, output, paramsBuf.View);
                return;
            }
            if (EnableDp4aGemv && type == GGMLType.Q6_K && _accelerator.AcceleratorType == AcceleratorType.Cuda)
            {
                int nBlk = K / 32;
                if (!_actQs.TryGetValue(K, out var qsBuf)) { qsBuf = _accelerator.Allocate1D<int>(nBlk * 8); _actQs[K] = qsBuf; }
                if (!_actDs.TryGetValue(K, out var dsBuf)) { dsBuf = _accelerator.Allocate1D<float>(nBlk * 2); _actDs[K] = dsBuf; }
                _quantActKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
                    ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int>(QuantizeActQ8_1Impl);
                _quantActKernel(nBlk, input, qsBuf.View, dsBuf.View, nBlk);
                _gemvDp4aQ6_K ??= _accelerator.LoadStreamKernel<ArrayView1D<int, Stride1D.Dense>,
                    ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
                    ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>(GemvDp4aQ6_KImpl);
                _gemvDp4aQ6_K(new KernelConfig(N, _accelerator.WarpSize), qsBuf.View, dsBuf.View, intView, output, paramsBuf.View);
                return;
            }
            // Warp-cooperative Q4_K GEMV: one warp per output column, Warp.Shuffle scale-broadcast + reduction,
            // NO Group.Barrier / NO shared mem (the default kernel's per-super-block barriers cap its bandwidth).
            // CUDA only: warp==32 + Warp.Shuffle always available. ILGPU's OpenCL backend needs the
            // cl_khr_subgroup_shuffle / cl_intel_subgroups extension (absent on NVIDIA's OpenCL → "Invalid code
            // generation"), so OpenCL keeps the portable GEMV until that capability path lands (ILGPU/Geordi).
            if (EnableWarpGemv && type == GGMLType.Q4_K && _accelerator.AcceleratorType == AcceleratorType.Cuda)
            {
                var warpCfg = new KernelConfig(N, _accelerator.WarpSize);
                _gemvWarpQ4_K ??= _accelerator.LoadStreamKernel<ArrayView1D<float, Stride1D.Dense>,
                    ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                    ArrayView1D<int, Stride1D.Dense>>(GemvDequantQ4_KWarpImpl);
                _gemvWarpQ4_K(warpCfg, input, intView, output, paramsBuf.View);
                return;
            }
            if (EnableWarpGemv && type == GGMLType.Q6_K && _accelerator.AcceleratorType == AcceleratorType.Cuda)
            {
                var warpCfg = new KernelConfig(N, _accelerator.WarpSize);
                _gemvWarpQ6_K ??= _accelerator.LoadStreamKernel<ArrayView1D<float, Stride1D.Dense>,
                    ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                    ArrayView1D<int, Stride1D.Dense>>(GemvDequantQ6_KWarpImpl);
                _gemvWarpQ6_K(warpCfg, input, intView, output, paramsBuf.View);
                return;
            }
            switch (type)
            {
                case GGMLType.Q4_K:
                    _gemvQ4_K ??= _accelerator.LoadStreamKernel<ArrayView1D<float, Stride1D.Dense>,
                        ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                        ArrayView1D<int, Stride1D.Dense>>(GemvDequantQ4_KImpl);
                    _gemvQ4_K(gemvConfig, input, intView, output, paramsBuf.View);
                    return;
                case GGMLType.Q6_K:
                    _gemvQ6_K ??= _accelerator.LoadStreamKernel<ArrayView1D<float, Stride1D.Dense>,
                        ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                        ArrayView1D<int, Stride1D.Dense>>(GemvDequantQ6_KImpl);
                    _gemvQ6_K(gemvConfig, input, intView, output, paramsBuf.View);
                    return;
                case GGMLType.Q8_0:
                    _gemvQ8_0 ??= _accelerator.LoadStreamKernel<ArrayView1D<float, Stride1D.Dense>,
                        ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                        ArrayView1D<int, Stride1D.Dense>>(GemvDequantQ8_0Impl);
                    _gemvQ8_0(gemvConfig, input, intView, output, paramsBuf.View);
                    return;
                case GGMLType.Q4_0:
                    _gemvQ4_0 ??= _accelerator.LoadStreamKernel<ArrayView1D<float, Stride1D.Dense>,
                        ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                        ArrayView1D<int, Stride1D.Dense>>(GemvDequantQ4_0Impl);
                    _gemvQ4_0(gemvConfig, input, intView, output, paramsBuf.View);
                    return;
                case GGMLType.MXFP4:
                    _gemvMXFP4 ??= _accelerator.LoadStreamKernel<ArrayView1D<float, Stride1D.Dense>,
                        ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                        ArrayView1D<int, Stride1D.Dense>>(GemvDequantMXFP4Impl);
                    _gemvMXFP4(gemvConfig, input, intView, output, paramsBuf.View);
                    return;
            }
        }

        // Multi-row dequant GEMM for M>1 (prefill) — opt-in (GGUF_GEMM_MR=1) until A/B-verified. Same shared-mem
        // /barrier shape as the GEMV, so it's gated to non-browser-GPU backends (WebGL has no workgroup shared
        // memory; WebGPU's workgroup reduction is slow on Tint/Dawn — both keep the per-element kernel below).
        if (M > 1 && EnableMultiRowGemm && !gpuBrowser && (type == GGMLType.Q4_K || type == GGMLType.Q6_K))
        {
            // M>=RB_TILE (large prefill): FLOP-efficient register-blocked tiled dequant GEMM. Smaller M: multi-row.
            // The register-blocked kernel launches RB_BLOCK*RB_BLOCK (256) threads per group; the ILGPU CPU
            // accelerator caps a group dimension at 64 (same reason GemvGroupSize=64), so on any backend whose
            // max group-X is below 256 (CPU) we fall through to the multi-row kernel (group 64) — which handles
            // any M correctly, just less FLOP-efficiently. CUDA/OpenCL/Wasm allow >=256 and take the fast path.
            if (M >= RB_TILE && _accelerator.MaxGroupSize.X >= RB_BLOCK * RB_BLOCK)
            {
                var rbConfig = new KernelConfig(
                    ((M + RB_TILE - 1) / RB_TILE) * ((N + RB_TILE - 1) / RB_TILE), RB_BLOCK * RB_BLOCK);
                if (type == GGMLType.Q4_K)
                {
                    _rbQ4_K ??= _accelerator.LoadStreamKernel<ArrayView1D<float, Stride1D.Dense>,
                        ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                        ArrayView1D<int, Stride1D.Dense>>(RegBlockedDequantQ4_KImpl);
                    _rbQ4_K(rbConfig, input, intView, output, paramsBuf.View);
                }
                else
                {
                    _rbQ6_K ??= _accelerator.LoadStreamKernel<ArrayView1D<float, Stride1D.Dense>,
                        ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                        ArrayView1D<int, Stride1D.Dense>>(RegBlockedDequantQ6_KImpl);
                    _rbQ6_K(rbConfig, input, intView, output, paramsBuf.View);
                }
                return;
            }

            var mrConfig = new KernelConfig(N * ((M + GemmMTile - 1) / GemmMTile), GemvGroupSize);
            if (type == GGMLType.Q4_K)
            {
                _gemmMRQ4_K ??= _accelerator.LoadStreamKernel<ArrayView1D<float, Stride1D.Dense>,
                    ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                    ArrayView1D<int, Stride1D.Dense>>(GemmDequantQ4_K_MultiRowImpl);
                _gemmMRQ4_K(mrConfig, input, intView, output, paramsBuf.View);
            }
            else
            {
                _gemmMRQ6_K ??= _accelerator.LoadStreamKernel<ArrayView1D<float, Stride1D.Dense>,
                    ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                    ArrayView1D<int, Stride1D.Dense>>(GemmDequantQ6_K_MultiRowImpl);
                _gemmMRQ6_K(mrConfig, input, intView, output, paramsBuf.View);
            }
            return;
        }

        switch (type)
        {
            case GGMLType.Q4_0:
                _kernelQ4_0 ??= LoadKernel(FusedDequantQ4_0Impl);
                _kernelQ4_0(M * N, input, intView, output, paramsBuf.View);
                break;
            case GGMLType.Q8_0:
                _kernelQ8_0 ??= LoadKernel(FusedDequantQ8_0Impl);
                _kernelQ8_0(M * N, input, intView, output, paramsBuf.View);
                break;
            case GGMLType.Q4_K:
                _kernelQ4_K ??= LoadKernel(FusedDequantQ4_KImpl);
                _kernelQ4_K(M * N, input, intView, output, paramsBuf.View);
                break;
            case GGMLType.Q6_K:
                _kernelQ6_K ??= LoadKernel(FusedDequantQ6_KImpl);
                _kernelQ6_K(M * N, input, intView, output, paramsBuf.View);
                break;
            case GGMLType.MXFP4:
                _kernelMXFP4 ??= LoadKernel(FusedDequantMXFP4Impl);
                _kernelMXFP4(M * N, input, intView, output, paramsBuf.View);
                break;
        }
    }

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>> LoadKernel(
        Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>> impl) =>
        _accelerator.LoadAutoGroupedStreamKernel(impl);

    // ─────────────────────────────────────────────────────────────────────────
    //  Q4_0: 18 bytes per 32 values — [d:fp16][16 × packed nibbles]
    //  Element j (0..15) = low nibble of byte j; element j+16 = HIGH nibble of
    //  byte j (ggml split order, NOT interleaved). value = (nibble - 8) * d.
    // ─────────────────────────────────────────────────────────────────────────
    private static void FusedDequantQ4_0Impl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<int, Stride1D.Dense> w,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int M = p[0], K = p[1], N = p[2];
        int m = idx / N;
        int n = idx % N;
        if (m >= M) return;

        int blocksPerRow = K / 32;
        int bytesPerRow = blocksPerRow * 18;
        int inBase = m * K;

        float sum = 0f;
        for (int block = 0; block < blocksPerRow; block++)
        {
            int bOff = n * bytesPerRow + block * 18;
            float d = HalfToFloat(ReadByte(w, bOff) | (ReadByte(w, bOff + 1) << 8));
            int kBase = block * 32;

            for (int j = 0; j < 16; j++)
            {
                int packed = ReadByte(w, bOff + 2 + j);
                sum += input[inBase + kBase + j] * (((packed & 0xF) - 8) * d);
                sum += input[inBase + kBase + j + 16] * (((packed >> 4) - 8) * d);
            }
        }
        output[idx] = sum;
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  MXFP4 (ggml GGML_TYPE_MXFP4 = 39, used by gpt-oss): 17 bytes per 32 values —
    //  [e:E8M0 (1 byte)][16 × packed FP4 nibbles]. Element j (0..15) = LOW nibble of
    //  byte j; element j+16 = HIGH nibble of byte j (ggml split order, like Q4_0).
    //  value = E2M1[nibble] · 2^(e-127) — the canonical MX form (OCP E2M1 element ×
    //  E8M0 shared scale). This is ggml dequantize_row_mxfp4 with its doubled
    //  kvalues_mxfp4 table folded into the canonical scale: ggml uses {0,1,2,3,4,6,8,
    //  12,...} (= 2× E2M1) paired with the HALVED scale 2^(e-128); 2×E2M1·2^(e-128) ≡
    //  E2M1·2^(e-127). We compose the VERIFIED library E2M1 decode
    //  Float4E2M1Extensions.RawBitsToFloat (bit-exact ml_dtypes.float4_e2m1fn, all 6
    //  backends, pure bit-math = no struct ctor) instead of a hand-rolled table —
    //  single source of truth (Rule 2). Scale = the verified library E8M0 decode
    //  Float8E8M0Extensions.RawBitsToFloat (2^(e-127), e==0xFF→NaN; bit-exact
    //  ml_dtypes.float8_e8m0fnu, all 6 backends, pure bit-math = no struct ctor).
    // ─────────────────────────────────────────────────────────────────────────
    private static void FusedDequantMXFP4Impl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<int, Stride1D.Dense> w,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int M = p[0], K = p[1], N = p[2];
        int m = idx / N;
        int n = idx % N;
        if (m >= M) return;

        int blocksPerRow = K / 32;
        int bytesPerRow = blocksPerRow * 17;
        int inBase = m * K;

        float sum = 0f;
        for (int block = 0; block < blocksPerRow; block++)
        {
            int bOff = n * bytesPerRow + block * 17;
            float d = Float8E8M0Extensions.RawBitsToFloat(ReadByte(w, bOff));
            int kBase = block * 32;

            for (int j = 0; j < 16; j++)
            {
                int packed = ReadByte(w, bOff + 1 + j);
                sum += input[inBase + kBase + j] * (Float4E2M1Extensions.RawBitsToFloat(packed & 0xF) * d);
                sum += input[inBase + kBase + j + 16] * (Float4E2M1Extensions.RawBitsToFloat(packed >> 4) * d);
            }
        }
        output[idx] = sum;
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Q8_0: 34 bytes per 32 values — [d:fp16][32 × int8]. value = q * d.
    // ─────────────────────────────────────────────────────────────────────────
    private static void FusedDequantQ8_0Impl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<int, Stride1D.Dense> w,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int M = p[0], K = p[1], N = p[2];
        int m = idx / N;
        int n = idx % N;
        if (m >= M) return;

        int blocksPerRow = K / 32;
        int bytesPerRow = blocksPerRow * 34;
        int inBase = m * K;

        float sum = 0f;
        for (int block = 0; block < blocksPerRow; block++)
        {
            int bOff = n * bytesPerRow + block * 34;
            float d = HalfToFloat(ReadByte(w, bOff) | (ReadByte(w, bOff + 1) << 8));
            int kBase = block * 32;

            for (int j = 0; j < 32; j++)
            {
                int q = SignExtend8(ReadByte(w, bOff + 2 + j));
                sum += input[inBase + kBase + j] * (q * d);
            }
        }
        output[idx] = sum;
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Q4_K: 144 bytes per 256 values — [d:fp16][dmin:fp16][scales:12B][qs:128B]
    //  Four 64-element chunks; chunk t reads 32 bytes: LOW nibbles are elements
    //  64t..64t+31 (6-bit scale/min pair 2t), HIGH nibbles are elements
    //  64t+32..64t+63 (pair 2t+1). value = d·sc·nibble − dmin·m.
    //
    //  SHAPE WARNING: this kernel must stay ONE FLAT DYNAMIC k-loop with per-element
    //  decode (the same shape as GatherQ4_KImpl, which is proven small + green on
    //  WebGL). Chunk-structured forms - constant 4×32 nested loops, with the scale
    //  decode hoisted per chunk - explode the WebGL GLSL emitter: nested per-access
    //  bounds-check branches × unrolled iterations × sequential-loops-in-loop block
    //  re-emission produced 31.5MB (if/else scales), 253MB (ternary scales), 12MB
    //  (straight-line scales), 5.6MB (dynamic lane bounds) of GLSL vs ~50KB for this
    //  form - all measured offline via ShaderCompiler.Generate, 2026-06-11. The
    //  per-element re-decode costs ALU, not bandwidth; revisit only after the ILGPU
    //  WebGL emitter fix (filed to the ILGPU lane) and only with the probe re-run.
    // ─────────────────────────────────────────────────────────────────────────
    private static void FusedDequantQ4_KImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<int, Stride1D.Dense> w,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int M = p[0], K = p[1], N = p[2];
        int m = idx / N;
        int n = idx % N;
        if (m >= M) return;

        int bytesPerRow = K / 256 * 144;
        int rowBase = n * bytesPerRow;
        int inBase = m * K;

        float sum = 0f;
        for (int k = 0; k < K; k++)
            sum += input[inBase + k] * DecodeQ4KElement(w, rowBase, k);
        output[idx] = sum;
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Q4_K GEMV (M==1): one thread GROUP per output column n. Thread tid accumulates
    //  input[k]·W[n,k] over k = tid, tid+G, tid+2G, … (consecutive lanes → consecutive k →
    //  coalesced weight bytes), then a shared-memory tree reduction sums the partials → output[n].
    //  Reuses DecodeQ4KElement (same layout inverse as the M>1 kernel), so GPU==CPU oracle holds.
    //  M is 1 by construction (GEMV dispatch), so input base is 0.
    // ─────────────────────────────────────────────────────────────────────────
    private static void GemvDequantQ4_KImpl(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<int, Stride1D.Dense> w,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int K = p[1], N = p[2];
        int n = Grid.IdxX;       // one group per output column
        int tid = Group.IdxX;    // 0..GemvGroupSize-1

        var sh = SharedMemory.Allocate<float>(GemvGroupSize);
        // Per-block sub-block-scale cache: a Q4_K block has 8 sub-blocks (j=0..7), each with a 6-bit (sc,mn)
        // shared by 32 elements. Extracting them per element (get_scale_min_k4 = ~15 ALU) was the GEMV's
        // bottleneck (MEASURED: Q4_K GEMV 33 GB/s vs trivial-dequant Q8_0 86 GB/s on a 4070 — ALU-bound, not
        // bandwidth). Decode the 8 folded {d·sc, dmin·mn} ONCE per block into shared, then the per-element work
        // is a nibble fetch + 1 mul + 1 sub (DecodeQ4KNibble). Bit-identical to DecodeQ4KScaled.
        var shDsc = SharedMemory.Allocate<float>(8);
        var shDmm = SharedMemory.Allocate<float>(8);
        float partial = 0f;
        if (n < N)
        {
            int bytesPerRow = K / 256 * 144;
            int rowBase = n * bytesPerRow;
            int numBlocks = K / 256;
            int perThread = 256 / GemvGroupSize;
            for (int blk = 0; blk < numBlocks; blk++)
            {
                int sbOff = rowBase + blk * 144;
                // 8 threads decode the block's 8 sub-block scales once (d/dmin block-constant, folded into d·sc/dmin·mn).
                if (tid < 8)
                {
                    float d = HalfToFloatFinite(ReadByte(w, sbOff) | (ReadByte(w, sbOff + 1) << 8));
                    float dmin = HalfToFloatFinite(ReadByte(w, sbOff + 2) | (ReadByte(w, sbOff + 3) << 8));
                    int j = tid, scOff = sbOff + 4;
                    int lowBit = ((j - 4) >> 31) & 1, hiBit = 1 - lowBit;
                    int bj = ReadByte(w, scOff + j), bj4 = ReadByte(w, scOff + j + 4), bjAlt = ReadByte(w, scOff + j - 4 * hiBit);
                    float sc = lowBit * (bj & 63) + hiBit * ((bj4 & 0xF) | ((bjAlt >> 6) << 4));
                    float mn = lowBit * (bj4 & 63) + hiBit * ((bj4 >> 4) | ((bj >> 6) << 4));
                    shDsc[j] = d * sc; shDmm[j] = dmin * mn;
                }
                Group.Barrier();
                int kBase = blk * 256;
                for (int sub = 0; sub < perThread; sub++)
                {
                    int r = tid + sub * GemvGroupSize;
                    int j = 2 * (r >> 6) + ((r >> 5) & 1);
                    partial += input[kBase + r] * DecodeQ4KNibble(w, sbOff, r, shDsc[j], shDmm[j]);
                }
                Group.Barrier(); // before the next block overwrites shDsc/shDmm
            }
        }
        sh[tid] = partial;
        Group.Barrier();

        // Tree reduction over the group (GemvGroupSize is a power of two).
        for (int stride = GemvGroupSize / 2; stride > 0; stride >>= 1)
        {
            if (tid < stride) sh[tid] += sh[tid + stride];
            Group.Barrier();
        }
        if (tid == 0 && n < N) output[n] = sh[0];
    }

    // Warp-cooperative VECTORIZED Q4_K GEMV (M==1 decode hot path): one 32-lane WARP per output column. The
    // default GemvDequantQ4_KImpl reads each 4-bit weight via ReadByte — loading a full 32-bit word to use ONE
    // nibble = 8× redundant loads (MEASURED the bottleneck: warp-shuffle reduction alone barely moved it, ~10%
    // of card bandwidth). Here each lane loads its 32-bit word of nibbles EXACTLY ONCE and decodes all 8 nibbles
    // in it (4 bytes × 2 nibbles = 8 elements). A super-block's 128 nibble-bytes = 32 words = one word per lane.
    // The 8 sub-block folded {d·sc, dmin·mn} are decoded by lanes<8 and shared via Warp.Shuffle; the final
    // reduction is Warp.ShuffleDown — ZERO Group.Barrier, ZERO shared memory. Same dequant math as DecodeQ4KNibble
    // (dsc·nibble − dmm); accumulation order differs (per-lane), so results match the per-element kernel to GEMV
    // float-reduction precision (the existing GEMV already differs from a serial sum at that level; argmax-identical,
    // oracle-verified to 2e-3). Requires warp size == 32 (gated at dispatch); CPU/Wasm/other keep the portable kernel.
    private static void GemvDequantQ4_KWarpImpl(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<int, Stride1D.Dense> w,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int K = p[1], N = p[2];
        int n = Grid.IdxX;          // one warp per output column
        int lane = Group.IdxX;      // 0..31
        float partial = 0f;
        if (n < N)
        {
            int bytesPerRow = K / 256 * 144;
            int rowBase = n * bytesPerRow;
            int numBlocks = K / 256;
            // Lane→data map (32 nibble-words/super-block, 1 per lane): the nibble region is 4 sub-block-pairs of
            // 32 bytes; lane's 4 bytes lie within ONE pair t = lane>>3 at l = 4·(lane&7)+bi (bi=0..3). t selects
            // sub-blocks 2t (nibble hi=0) and 2t+1 (nibble hi=1) — both shared in via Warp.Shuffle.
            int t = lane >> 3;            // 0..3  → sub-block pair {2t, 2t+1}
            int lBase = 4 * (lane & 7);   // 0,4,...,28  → first of this lane's 4 nibble columns
            for (int blk = 0; blk < numBlocks; blk++)
            {
                int sbOff = rowBase + blk * 144;
                // Lanes<8 decode their sub-block's folded {d·sc, dmin·mn} (mirrors the default kernel's shDsc/shDmm).
                float myDsc = 0f, myDmm = 0f;
                if (lane < 8)
                {
                    float d = HalfToFloatFinite(ReadByte(w, sbOff) | (ReadByte(w, sbOff + 1) << 8));
                    float dmin = HalfToFloatFinite(ReadByte(w, sbOff + 2) | (ReadByte(w, sbOff + 3) << 8));
                    int j = lane, scOff = sbOff + 4;
                    int lowBit = ((j - 4) >> 31) & 1, hiBit = 1 - lowBit;
                    int bj = ReadByte(w, scOff + j), bj4 = ReadByte(w, scOff + j + 4), bjAlt = ReadByte(w, scOff + j - 4 * hiBit);
                    float sc = lowBit * (bj & 63) + hiBit * ((bj4 & 0xF) | ((bjAlt >> 6) << 4));
                    float mn = lowBit * (bj4 & 63) + hiBit * ((bj4 >> 4) | ((bj >> 6) << 4));
                    myDsc = d * sc; myDmm = dmin * mn;
                }
                float dsc0 = Warp.Shuffle(myDsc, 2 * t), dmm0 = Warp.Shuffle(myDmm, 2 * t);
                float dsc1 = Warp.Shuffle(myDsc, 2 * t + 1), dmm1 = Warp.Shuffle(myDmm, 2 * t + 1);

                // Load this lane's 4 nibble-bytes as ONE 32-bit word (sbOff+16 is 4-aligned; +32t+lBase too).
                int word = w[(sbOff + 16 + 32 * t + lBase) >> 2];
                int kHi0 = blk * 256 + (t << 6) + lBase;   // global index of hi=0 element at l=lBase
                for (int bi = 0; bi < 4; bi++)
                {
                    int b = (word >> (bi * 8)) & 0xFF;
                    int nib0 = b & 0xF;          // hi=0 nibble → sub-block 2t
                    int nib1 = (b >> 4) & 0xF;   // hi=1 nibble → sub-block 2t+1
                    partial += input[kHi0 + bi] * (dsc0 * nib0 - dmm0);
                    partial += input[kHi0 + 32 + bi] * (dsc1 * nib1 - dmm1);
                }
            }
        }
        // Warp-shuffle reduction (no shared mem / no barrier): lane 0 ends with the column's dot product.
        for (int off = 16; off > 0; off >>= 1)
            partial += Warp.ShuffleDown(partial, off);
        if (lane == 0 && n < N) output[n] = partial;
    }

    // Warp-cooperative Q6_K decode GEMV (M==1): one 32-lane warp per output column. The default GemvDequantQ6_KImpl
    // calls DecodeQ6KScaled per element, which RE-reads the same qh / ql / scale bytes for each of the 4 elements
    // they encode (~4x redundant loads). Here lane = l (0..31) reads each block's qh[l] + ql[l] + ql[l+32] ONCE and
    // decodes all 4 variants (q1..q4) — the EXACT decode of the verified Q6_K multi-row GEMM (GemmDequantQ6_K_*), so
    // bit-for-tolerance identical. 32 consecutive lanes read 32 consecutive bytes → coalesced. Final reduction via
    // Warp.ShuffleDown (no Group.Barrier, no shared mem). CUDA only (Warp.Shuffle); others keep the portable kernel.
    private static void GemvDequantQ6_KWarpImpl(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<int, Stride1D.Dense> w,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int K = p[1], N = p[2];
        int n = Grid.IdxX;
        int l = Group.IdxX;     // lane == element index l (0..31) within a half
        float partial = 0f;
        if (n < N)
        {
            int bytesPerRow = K / 256 * 210;
            int rowBase = n * bytesPerRow;
            int numBlocks = K / 256;
            for (int blk = 0; blk < numBlocks; blk++)
            {
                int sbOff = rowBase + blk * 210;
                float d = HalfToFloatFinite(ReadByte(w, sbOff + 208) | (ReadByte(w, sbOff + 209) << 8));
                int kBase = blk * 256;
                for (int half = 0; half < 2; half++)
                {
                    int ql = sbOff + 64 * half;
                    int qh = sbOff + 128 + 32 * half;
                    int sc = sbOff + 192 + 8 * half;
                    int y = kBase + half * 128;
                    int isIdx = l >> 4;                 // l/16 -> 0 or 1 (scale sub-index)
                    int hb = ReadByte(w, qh + l);
                    int lo0 = ReadByte(w, ql + l);      // 4 elements per (l,half) from 1 qh + 2 ql bytes (no redundancy)
                    int lo32 = ReadByte(w, ql + l + 32);
                    int q1 = ((lo0 & 0xF) | ((hb & 3) << 4)) - 32;
                    int q2 = ((lo32 & 0xF) | (((hb >> 2) & 3) << 4)) - 32;
                    int q3 = ((lo0 >> 4) | (((hb >> 4) & 3) << 4)) - 32;
                    int q4 = ((lo32 >> 4) | (((hb >> 6) & 3) << 4)) - 32;
                    float s0 = SignExtend8(ReadByte(w, sc + isIdx));
                    float s2 = SignExtend8(ReadByte(w, sc + isIdx + 2));
                    float s4 = SignExtend8(ReadByte(w, sc + isIdx + 4));
                    float s6 = SignExtend8(ReadByte(w, sc + isIdx + 6));
                    partial += input[y + l] * (d * s0 * q1);
                    partial += input[y + l + 32] * (d * s2 * q2);
                    partial += input[y + l + 64] * (d * s4 * q3);
                    partial += input[y + l + 96] * (d * s6 * q4);
                }
            }
        }
        for (int off = 16; off > 0; off >>= 1)
            partial += Warp.ShuffleDown(partial, off);
        if (l == 0 && n < N) output[n] = partial;
    }

    // Wrapper for the dp4a (4x int8 dot-accumulate) PTX intrinsic: d = c + Σ a.s8[i]·b.s8[i]. CUDA-only.
    private static int Dp4a(int a, int b, int c)
    {
        CudaAsm.Emit("dp4a.s32.s32 %0, %1, %2, %3;", out int r, a, b, c);
        return r;
    }

    // Quantize the activation vector to block_q8_1 (32-element blocks): per block, int8 quants packed 4/word
    // (8 words/block) + d (= amax/127) + s (= d·Σq, the term for the Q4_K dmin·mn offset). One thread per block.
    private static void QuantizeActQ8_1Impl(Index1D idx, ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<int, Stride1D.Dense> qs, ArrayView1D<float, Stride1D.Dense> ds, int nBlk)
    {
        int blk = idx.X;
        if (blk >= nBlk) return;
        int b = blk * 32;
        float amax = 0f;
        for (int j = 0; j < 32; j++) { float a = input[b + j]; a = a < 0f ? -a : a; if (a > amax) amax = a; }
        float d = amax * (1f / 127f);
        float invd = amax > 0f ? 127f / amax : 0f;
        int sum = 0;
        for (int wi = 0; wi < 8; wi++)
        {
            int packed = 0;
            for (int e = 0; e < 4; e++)
            {
                float v = input[b + wi * 4 + e] * invd;
                int q = (int)(v + (v >= 0f ? 0.5f : -0.5f));
                q = q < -127 ? -127 : (q > 127 ? 127 : q);
                sum += q;
                packed |= (q & 0xFF) << (e * 8);
            }
            qs[blk * 8 + wi] = packed;
        }
        ds[blk * 2 + 0] = d;
        ds[blk * 2 + 1] = d * sum;
    }

    // Folded Q4_K sub-block scale {d·sc, dmin·mn} for sub-block j (0..7) at super-block byte sbOff — same 6-bit
    // extraction as DecodeQ4KScaled / the warp kernel. Returns d·sc; dmin·mn via out.
    private static float DecodeQ4KDsc(ArrayView1D<int, Stride1D.Dense> w, int sbOff, int j, out float dmm)
    {
        float d = HalfToFloatFinite(ReadByte(w, sbOff) | (ReadByte(w, sbOff + 1) << 8));
        float dmin = HalfToFloatFinite(ReadByte(w, sbOff + 2) | (ReadByte(w, sbOff + 3) << 8));
        int scOff = sbOff + 4;
        int lowBit = ((j - 4) >> 31) & 1, hiBit = 1 - lowBit;
        int bj = ReadByte(w, scOff + j), bj4 = ReadByte(w, scOff + j + 4), bjAlt = ReadByte(w, scOff + j - 4 * hiBit);
        float sc = lowBit * (bj & 63) + hiBit * ((bj4 & 0xF) | ((bjAlt >> 6) << 4));
        float mn = lowBit * (bj4 & 63) + hiBit * ((bj4 >> 4) | ((bj >> 6) << 4));
        dmm = dmin * mn;
        return d * sc;
    }

    // dp4a int8-activation Q4_K decode GEMV (the llama.cpp MMVQ path): one 32-lane warp per output column; each
    // lane owns whole "t-units" (one t = sub-blocks 2t,2t+1 = 64 elements), reading its 8 weight nibble-words
    // ONCE, nibble-masking to two int4-packed words, and dp4a-ing against the int8 activation quants. Dot in
    // int32, then folded to float by the weight scale × activation scale:
    //   partial += (d·sc)·d_a·dot − (dmin·mn)·s_a   per sub-block.   Warp-reduce. CUDA only (dp4a).
    private static void GemvDp4aQ4_KImpl(
        ArrayView1D<int, Stride1D.Dense> qs,
        ArrayView1D<float, Stride1D.Dense> ds,
        ArrayView1D<int, Stride1D.Dense> w,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int K = p[1], N = p[2];
        int n = Grid.IdxX;
        int lane = Group.IdxX;
        float partial = 0f;
        if (n < N)
        {
            int bytesPerRow = K / 256 * 144;
            int rowBase = n * bytesPerRow;
            int nTU = K / 64;
            for (int tu = lane; tu < nTU; tu += 32)
            {
                int super = tu >> 2;
                int t = tu & 3;
                int sbOff = rowBase + super * 144;
                float dmm0, dmm1;
                float dsc0 = DecodeQ4KDsc(w, sbOff, 2 * t, out dmm0);
                float dsc1 = DecodeQ4KDsc(w, sbOff, 2 * t + 1, out dmm1);
                int ablk0 = super * 8 + 2 * t, ablk1 = ablk0 + 1;
                float da0 = ds[ablk0 * 2], sa0 = ds[ablk0 * 2 + 1];
                float da1 = ds[ablk1 * 2], sa1 = ds[ablk1 * 2 + 1];
                int wWordBase = (sbOff + 16 + 32 * t) >> 2;
                int dot0 = 0, dot1 = 0;
                for (int wi = 0; wi < 8; wi++)
                {
                    int wword = w[wWordBase + wi];
                    dot0 = Dp4a((wword >> 0) & 0x0F0F0F0F, qs[ablk0 * 8 + wi], dot0);
                    dot1 = Dp4a((wword >> 4) & 0x0F0F0F0F, qs[ablk1 * 8 + wi], dot1);
                }
                partial += dsc0 * da0 * dot0 - dmm0 * sa0;
                partial += dsc1 * da1 * dot1 - dmm1 * sa1;
            }
        }
        for (int off = 16; off > 0; off >>= 1)
            partial += Warp.ShuffleDown(partial, off);
        if (lane == 0 && n < N) output[n] = partial;
    }

    // Misaligned 4-byte load: the 4 bytes starting at arbitrary byteOff as an int (Q6_K's 210-byte super-blocks
    // are not 4-aligned). At most 2 int loads.
    private static int Load4Bytes(ArrayView1D<int, Stride1D.Dense> w, int byteOff)
    {
        int wi = byteOff >> 2;
        int sh = (byteOff & 3) * 8;
        int lo = w[wi];
        if (sh == 0) return lo;
        int hi = w[wi + 1];
        return (int)(((uint)lo >> sh) | ((uint)hi << (32 - sh)));
    }

    // Pack 4 Q6_K values of one variant into an int32 of 4 int8: q = (qlNib | (qhBits<<4)) − 32, q∈[-32,31].
    private static int PackQ6(int qlInt, int qhInt, int nibShift, int qhShift)
    {
        int packed = 0;
        for (int e = 0; e < 4; e++)
        {
            int qlNib = ((qlInt >> (e * 8 + nibShift)) & 0xF);
            int qhBits = ((qhInt >> (e * 8 + qhShift)) & 3);
            int q = (qlNib | (qhBits << 4)) - 32;
            packed |= (q & 0xFF) << (e * 8);
        }
        return packed;
    }

    // dp4a int8-activation Q6_K decode GEMV. Q6_K is SYMMETRIC (q∈[-32,31], value=d·sc·q, NO dmin), with 16-elem
    // int8-scaled sub-blocks and 6-bit values split ql(4)+qh(2). Each lane owns whole "(half, l-group-of-4)"
    // units: reads ql[l..l+3], ql[l+32..l+35], qh[l..l+3] ONCE (3 misaligned 4-byte loads), decodes all 4
    // variants × 4 elements, packs each variant's 4 q's, dp4a's against the int8 activation quants (variant v →
    // activation block super*8+half*4+v). partial += d·sc_v·d_a·dot_v. Same int8-act numerics as Q4_K dp4a; warp-
    // reduce. CUDA only (dp4a).
    private static void GemvDp4aQ6_KImpl(
        ArrayView1D<int, Stride1D.Dense> qs,
        ArrayView1D<float, Stride1D.Dense> ds,
        ArrayView1D<int, Stride1D.Dense> w,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int K = p[1], N = p[2];
        int n = Grid.IdxX;
        int lane = Group.IdxX;
        float partial = 0f;
        if (n < N)
        {
            int bytesPerRow = K / 256 * 210;
            int rowBase = n * bytesPerRow;
            int nUnits = K / 16;     // 16 units per 256-super-block
            for (int u = lane; u < nUnits; u += 32)
            {
                int super = u >> 4;
                int half = (u >> 3) & 1;
                int lgroup = u & 7;
                int l = lgroup * 4;
                int sbOff = rowBase + super * 210;
                float dw = HalfToFloatFinite(ReadByte(w, sbOff + 208) | (ReadByte(w, sbOff + 209) << 8));
                int qlBase = sbOff + 64 * half;
                int qhBase = sbOff + 128 + 32 * half;
                int scBase = sbOff + 192 + 8 * half;
                int qlA = Load4Bytes(w, qlBase + l);
                int qlB = Load4Bytes(w, qlBase + 32 + l);
                int qhW = Load4Bytes(w, qhBase + l);
                int scIdx = lgroup >> 2;                 // l>>4
                int ablkBase = super * 8 + half * 4;
                int d0 = Dp4a(PackQ6(qlA, qhW, 0, 0), qs[(ablkBase + 0) * 8 + lgroup], 0);
                int d1 = Dp4a(PackQ6(qlB, qhW, 0, 2), qs[(ablkBase + 1) * 8 + lgroup], 0);
                int d2 = Dp4a(PackQ6(qlA, qhW, 4, 4), qs[(ablkBase + 2) * 8 + lgroup], 0);
                int d3 = Dp4a(PackQ6(qlB, qhW, 4, 6), qs[(ablkBase + 3) * 8 + lgroup], 0);
                partial += dw * SignExtend8(ReadByte(w, scBase + scIdx + 0)) * ds[(ablkBase + 0) * 2] * d0;
                partial += dw * SignExtend8(ReadByte(w, scBase + scIdx + 2)) * ds[(ablkBase + 1) * 2] * d1;
                partial += dw * SignExtend8(ReadByte(w, scBase + scIdx + 4)) * ds[(ablkBase + 2) * 2] * d2;
                partial += dw * SignExtend8(ReadByte(w, scBase + scIdx + 6)) * ds[(ablkBase + 3) * 2] * d3;
            }
        }
        for (int off = 16; off > 0; off >>= 1)
            partial += Warp.ShuffleDown(partial, off);
        if (lane == 0 && n < N) output[n] = partial;
    }

    /// <summary>Decode element <paramref name="r"/> (0..255) within a Q4_K super-block at byte
    /// <paramref name="sbOff"/>, given the block's already-decoded fp16 <paramref name="d"/>/<paramref name="dmin"/>
    /// (hoisted out of the per-element loop in the GEMV). The 6-bit sub-block scale/min + nibble are identical
    /// to <see cref="DecodeQ4KElement"/>; only the redundant fp16 decode is lifted.</summary>
    private static float DecodeQ4KScaled(ArrayView1D<int, Stride1D.Dense> w, int sbOff, int r, float d, float dmin)
    {
        int t = r >> 6;
        int c = r & 63;
        int l = c & 31;
        int hi = (c >> 5) & 1;
        int scOff = sbOff + 4;
        int j = 2 * t + hi;
        int lowBit = ((j - 4) >> 31) & 1;
        int hiBit = 1 - lowBit;
        int bj = ReadByte(w, scOff + j);
        int bj4 = ReadByte(w, scOff + j + 4);
        int bjAlt = ReadByte(w, scOff + j - 4 * hiBit);
        int sc = lowBit * (bj & 63) + hiBit * ((bj4 & 0xF) | ((bjAlt >> 6) << 4));
        int mn = lowBit * (bj4 & 63) + hiBit * ((bj4 >> 4) | ((bj >> 6) << 4));
        int packed = ReadByte(w, sbOff + 16 + 32 * t + l);
        int nibble = (packed >> (4 * hi)) & 0xF;
        return d * sc * nibble - dmin * mn;
    }

    /// <summary>Decode Q4_K element <paramref name="r"/> (0..255) within a super-block at byte <paramref name="sbOff"/>,
    /// given the per-column products already folded: <paramref name="dsc"/> = d·sc and <paramref name="dminmn"/> =
    /// dmin·mn (both constant across a K-tile, which lies within one Q4_K sub-block). Only the per-element nibble is
    /// read here, so the per-element dequant is a nibble fetch + one multiply + one subtract. Bit-identical to
    /// <see cref="DecodeQ4KElement"/>: d·sc·nibble − dmin·mn == (d·sc)·nibble − (dmin·mn), same float op order.</summary>
    private static float DecodeQ4KNibble(ArrayView1D<int, Stride1D.Dense> w, int sbOff, int r, float dsc, float dminmn)
    {
        int t = r >> 6;
        int l = (r & 63) & 31;
        int hi = (r >> 5) & 1;
        int packed = ReadByte(w, sbOff + 16 + 32 * t + l);
        int nibble = (packed >> (4 * hi)) & 0xF;
        return dsc * nibble - dminmn;
    }

    /// <summary>Decode Q6_K element <paramref name="r"/> (0..255) within a super-block at byte <paramref name="sbOff"/>,
    /// given the per-column product <paramref name="dsc"/> = d·sc already folded (constant across a K-tile, which lies
    /// within one Q6_K scale group of 16). Only the 6-bit quant is read here. Bit-identical to
    /// <see cref="DecodeQ6KElement"/>: d·sc·q == (d·sc)·q, same float op order.</summary>
    private static float DecodeQ6KNibble(ArrayView1D<int, Stride1D.Dense> w, int sbOff, int r, float dsc)
    {
        int half = r >> 7;
        int rh = r & 127;
        int variant = rh >> 5;
        int l = rh & 31;
        int qlByte = ReadByte(w, sbOff + 64 * half + l + (variant & 1) * 32);
        int qh = ReadByte(w, sbOff + 128 + 32 * half + l);
        int isHigh = variant >> 1;
        int qlNib = isHigh == 1 ? (qlByte >> 4) : (qlByte & 0xF);
        int qhBits = (qh >> (2 * variant)) & 3;
        int q = (qlNib | (qhBits << 4)) - 32;
        return dsc * q;
    }

    // Number of output rows (M) a single multi-row-GEMM group handles. Each weight element is dequantized
    // ONCE and reused across these GemmMTile input rows — killing the O(M) redundant dequant of the general
    // per-element kernel (the prefill bottleneck). Small enough that the per-thread accumulators stay in regs.
    private const int GemmMTile = 8;

    // Register-blocked dequant GEMM (the FLOP-efficient prefill path for M>=RB_TILE). Mirrors the verified f32
    // RegisterBlockedMatMul (16x16 block, each thread a 4x4 register tile, 64x64 output tile) — the ONLY change
    // is the B (weight) tile is dequantized on load into shared memory, then reused REG×BLOCK times from
    // registers. So a weight element is dequantized once per K-tile and amortized across the whole output tile.
    private const int RB_BLOCK = 16;  // 16x16 = 256 threads
    private const int RB_REG = 4;     // each thread computes REG x REG outputs
    private const int RB_TILE = RB_BLOCK * RB_REG; // 64x64 output tile

    /// <summary>
    /// Q4_K dequant GEMM for M&gt;1 (prefill). Group-per-(output column n, M-tile): the group walks the column's
    /// Q4_K blocks coalesced (identical to the M=1 GEMV), dequantizes each element ONCE via
    /// <see cref="DecodeQ4KScaled"/>, and accumulates it into <see cref="GemmMTile"/> input rows at once — so a
    /// weight is dequantized M/GemmMTile times instead of M times. Bit-compatible with the per-element kernel
    /// (same decode, same fp32 accumulation order per element). Grid = N·⌈M/GemmMTile⌉ groups.
    /// </summary>
    private static void GemmDequantQ4_K_MultiRowImpl(
        ArrayView1D<float, Stride1D.Dense> input,   // [M, K]
        ArrayView1D<int, Stride1D.Dense> w,
        ArrayView1D<float, Stride1D.Dense> output,   // [M, N]
        ArrayView1D<int, Stride1D.Dense> p)          // [M, K, N]
    {
        int M = p[0], K = p[1], N = p[2];
        int gi = Grid.IdxX;
        int n = gi % N;
        int mBase = (gi / N) * GemmMTile;
        int tid = Group.IdxX;

        var sh = SharedMemory.Allocate<float>(GemvGroupSize);
        var partial = new float[GemmMTile]; // per-thread M-tile accumulators (device-local array; correct on all
        for (int mi = 0; mi < GemmMTile; mi++) partial[mi] = 0f; // 6 backends since SpawnDev.ILGPU 4.15.1)

        if (n < N)
        {
            int bytesPerRow = K / 256 * 144;
            int rowBase = n * bytesPerRow;
            int numBlocks = K / 256;
            int perThread = 256 / GemvGroupSize;
            for (int blk = 0; blk < numBlocks; blk++)
            {
                int sbOff = rowBase + blk * 144;
                float d = HalfToFloatFinite(ReadByte(w, sbOff) | (ReadByte(w, sbOff + 1) << 8));
                float dmin = HalfToFloatFinite(ReadByte(w, sbOff + 2) | (ReadByte(w, sbOff + 3) << 8));
                int kBase = blk * 256;
                for (int sub = 0; sub < perThread; sub++)
                {
                    int r = tid + sub * GemvGroupSize;
                    float wval = DecodeQ4KScaled(w, sbOff, r, d, dmin); // dequant ONCE, reuse across rows
                    int kIdx = kBase + r;
                    for (int mi = 0; mi < GemmMTile; mi++)
                    {
                        int m = mBase + mi;
                        if (m < M) partial[mi] += input[m * K + kIdx] * wval;
                    }
                }
            }
        }

        // Reduce each row's partials across the group → output[m, n]. sh is reused per row (barrier between).
        for (int mi = 0; mi < GemmMTile; mi++)
        {
            sh[tid] = partial[mi];
            Group.Barrier();
            for (int stride = GemvGroupSize / 2; stride > 0; stride >>= 1)
            {
                if (tid < stride) sh[tid] += sh[tid + stride];
                Group.Barrier();
            }
            int m = mBase + mi;
            if (tid == 0 && n < N && m < M) output[m * N + n] = sh[0];
            Group.Barrier();
        }
    }

    /// <summary>Q6_K dequant GEMM for M&gt;1 — the Q6_K analogue of <see cref="GemmDequantQ4_K_MultiRowImpl"/>
    /// (210-byte super-blocks, scale at offset 208, no dmin, <see cref="DecodeQ6KScaled"/>). Frequently the
    /// output/logits projection, so this is a large prefill win.</summary>
    private static void GemmDequantQ6_K_MultiRowImpl(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<int, Stride1D.Dense> w,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int M = p[0], K = p[1], N = p[2];
        int gi = Grid.IdxX;
        int n = gi % N;
        int mBase = (gi / N) * GemmMTile;
        int tid = Group.IdxX;

        var sh = SharedMemory.Allocate<float>(GemvGroupSize);
        var partial = new float[GemmMTile];
        for (int mi = 0; mi < GemmMTile; mi++) partial[mi] = 0f;

        if (n < N)
        {
            int bytesPerRow = K / 256 * 210;
            int rowBase = n * bytesPerRow;
            int numBlocks = K / 256;
            int perThread = 256 / GemvGroupSize;
            for (int blk = 0; blk < numBlocks; blk++)
            {
                int sbOff = rowBase + blk * 210;
                float d = HalfToFloatFinite(ReadByte(w, sbOff + 208) | (ReadByte(w, sbOff + 209) << 8));
                int kBase = blk * 256;
                for (int sub = 0; sub < perThread; sub++)
                {
                    int r = tid + sub * GemvGroupSize;
                    float wval = DecodeQ6KScaled(w, sbOff, r, d); // dequant ONCE, reuse across rows
                    int kIdx = kBase + r;
                    for (int mi = 0; mi < GemmMTile; mi++)
                    {
                        int m = mBase + mi;
                        if (m < M) partial[mi] += input[m * K + kIdx] * wval;
                    }
                }
            }
        }

        for (int mi = 0; mi < GemmMTile; mi++)
        {
            sh[tid] = partial[mi];
            Group.Barrier();
            for (int stride = GemvGroupSize / 2; stride > 0; stride >>= 1)
            {
                if (tid < stride) sh[tid] += sh[tid + stride];
                Group.Barrier();
            }
            int m = mBase + mi;
            if (tid == 0 && n < N && m < M) output[m * N + n] = sh[0];
            Group.Barrier();
        }
    }

    /// <summary>Register-blocked dequant GEMM for Q4_K, M&gt;=RB_TILE (prefill). The verified f32
    /// RegisterBlockedMatMul with ONLY the B(weight) tile dequantized on load: W is stored [N,K], so the
    /// GEMM's B[k][n] = W[n][k] = <see cref="DecodeQ4KElement"/>(w, n·bytesPerRow, k). Each weight element is
    /// dequantized ONCE per K-tile and amortized across the 64×64 output tile (16 MACs/k from registers).</summary>
    private static void RegBlockedDequantQ4_KImpl(
        ArrayView1D<float, Stride1D.Dense> A,
        ArrayView1D<int, Stride1D.Dense> w,
        ArrayView1D<float, Stride1D.Dense> C,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int M = p[0], K = p[1], N = p[2];
        int numTilesN = (N + RB_TILE - 1) / RB_TILE;
        int bytesPerRow = K / 256 * 144;

        var aTile = SharedMemory.Allocate<float>(RB_TILE * RB_BLOCK);
        var bTile = SharedMemory.Allocate<float>(RB_BLOCK * RB_TILE);
        // Per-column Q4_K metadata for the current K-tile: the folded products {d·sc, dmin·mn} × RB_TILE columns.
        // A K-tile is 16 k deep and 16-aligned, so it lies entirely within ONE Q4_K sub-block (32 elems) for every
        // column — hence d/dmin (block-constant) AND the 6-bit sc/mn (sub-block-constant), and therefore d·sc and
        // dmin·mn, are the SAME for all 16 k. Decode + fold them ONCE per tile (64 cooperating threads) instead of
        // re-decoding per B-tile element (the old DecodeQ4KElement redid the fp16 d/dmin + 6-bit scale extraction
        // + the d·sc/dmin·mn products 16× redundantly per column). The per-element B load is then a nibble fetch +
        // one multiply + one subtract (DecodeQ4KNibble) — bit-identical.
        var bMeta = SharedMemory.Allocate<float>(RB_TILE * 2);

        int tileIdx = Grid.IdxX;
        int tileRow = tileIdx / numTilesN;
        int tileCol = tileIdx % numTilesN;
        int threadRow = Group.IdxX / RB_BLOCK;
        int threadCol = Group.IdxX % RB_BLOCK;

        float c00 = 0, c01 = 0, c02 = 0, c03 = 0, c10 = 0, c11 = 0, c12 = 0, c13 = 0;
        float c20 = 0, c21 = 0, c22 = 0, c23 = 0, c30 = 0, c31 = 0, c32 = 0, c33 = 0;

        int numKTiles = (K + RB_BLOCK - 1) / RB_BLOCK;
        for (int t = 0; t < numKTiles; t++)
        {
            for (int r = 0; r < RB_REG; r++)
            {
                int aRow = tileRow * RB_TILE + threadRow * RB_REG + r;
                int aCol = t * RB_BLOCK + threadCol;
                aTile[(threadRow * RB_REG + r) * RB_BLOCK + threadCol] = (aRow < M && aCol < K) ? A[aRow * K + aCol] : 0f;
            }
            // Decode this K-tile's per-column metadata once (one thread per column; the rest wait at the barrier).
            // r = k0 & 255 picks any element in the tile — the sub-block scale index j is constant across the tile.
            if (Group.IdxX < RB_TILE)
            {
                int metaCol = Group.IdxX;
                int n = tileCol * RB_TILE + metaCol;
                float dsc = 0f, dminmn = 0f;
                if (n < N)
                {
                    int k0 = t * RB_BLOCK;
                    int sbOff = n * bytesPerRow + (k0 >> 8) * 144;
                    int r = k0 & 255;
                    int tt = r >> 6;
                    int hi = (r >> 5) & 1;
                    float md = HalfToFloatFinite(ReadByte(w, sbOff) | (ReadByte(w, sbOff + 1) << 8));
                    float mdmin = HalfToFloatFinite(ReadByte(w, sbOff + 2) | (ReadByte(w, sbOff + 3) << 8));
                    int scOff = sbOff + 4;
                    int j = 2 * tt + hi;
                    int lowBit = ((j - 4) >> 31) & 1;
                    int hiBit = 1 - lowBit;
                    int bj = ReadByte(w, scOff + j);
                    int bj4 = ReadByte(w, scOff + j + 4);
                    int bjAlt = ReadByte(w, scOff + j - 4 * hiBit);
                    float msc = lowBit * (bj & 63) + hiBit * ((bj4 & 0xF) | ((bjAlt >> 6) << 4));
                    float mmn = lowBit * (bj4 & 63) + hiBit * ((bj4 >> 4) | ((bj >> 6) << 4));
                    dsc = md * msc;       // d·sc folded once per column (was recomputed per element)
                    dminmn = mdmin * mmn; // dmin·mn folded once per column
                }
                bMeta[metaCol * 2 + 0] = dsc;
                bMeta[metaCol * 2 + 1] = dminmn;
            }
            Group.Barrier();
            for (int r = 0; r < RB_REG; r++)
            {
                int bRow = t * RB_BLOCK + threadRow;                    // k
                int metaCol = threadCol * RB_REG + r;                   // column within tile (0..RB_TILE-1)
                int bCol = tileCol * RB_TILE + metaCol;                 // n
                int sbOff = bCol * bytesPerRow + (bRow >> 8) * 144;
                bTile[threadRow * RB_TILE + metaCol] =
                    (bRow < K && bCol < N)
                        ? DecodeQ4KNibble(w, sbOff, bRow & 255, bMeta[metaCol * 2 + 0], bMeta[metaCol * 2 + 1])
                        : 0f;
            }
            Group.Barrier();
            for (int k = 0; k < RB_BLOCK; k++)
            {
                float a0 = aTile[(threadRow * RB_REG + 0) * RB_BLOCK + k];
                float a1 = aTile[(threadRow * RB_REG + 1) * RB_BLOCK + k];
                float a2 = aTile[(threadRow * RB_REG + 2) * RB_BLOCK + k];
                float a3 = aTile[(threadRow * RB_REG + 3) * RB_BLOCK + k];
                float b0 = bTile[k * RB_TILE + threadCol * RB_REG + 0];
                float b1 = bTile[k * RB_TILE + threadCol * RB_REG + 1];
                float b2 = bTile[k * RB_TILE + threadCol * RB_REG + 2];
                float b3 = bTile[k * RB_TILE + threadCol * RB_REG + 3];
                c00 += a0 * b0; c01 += a0 * b1; c02 += a0 * b2; c03 += a0 * b3;
                c10 += a1 * b0; c11 += a1 * b1; c12 += a1 * b2; c13 += a1 * b3;
                c20 += a2 * b0; c21 += a2 * b1; c22 += a2 * b2; c23 += a2 * b3;
                c30 += a3 * b0; c31 += a3 * b1; c32 += a3 * b2; c33 += a3 * b3;
            }
            Group.Barrier();
        }
        WriteRegTile(C, M, N, tileRow * RB_TILE + threadRow * RB_REG, tileCol * RB_TILE + threadCol * RB_REG,
            c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22, c23, c30, c31, c32, c33);
    }

    /// <summary>Q6_K register-blocked dequant GEMM (Q6_K analogue: 210-byte blocks, <see cref="DecodeQ6KElement"/>).</summary>
    private static void RegBlockedDequantQ6_KImpl(
        ArrayView1D<float, Stride1D.Dense> A,
        ArrayView1D<int, Stride1D.Dense> w,
        ArrayView1D<float, Stride1D.Dense> C,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int M = p[0], K = p[1], N = p[2];
        int numTilesN = (N + RB_TILE - 1) / RB_TILE;
        int bytesPerRow = K / 256 * 210;

        var aTile = SharedMemory.Allocate<float>(RB_TILE * RB_BLOCK);
        var bTile = SharedMemory.Allocate<float>(RB_BLOCK * RB_TILE);
        // Per-column folded product d·sc for the current K-tile (one float per column). A K-tile (16 deep,
        // 16-aligned) lies within one Q6_K scale group of 16, so d (block-constant) and sc (group-constant)
        // — and thus d·sc — are the same for all 16 k. Decode once per tile; per-element B load is then just
        // the 6-bit quant fetch × dsc (DecodeQ6KNibble) — bit-identical to the per-element DecodeQ6KElement.
        var bMeta = SharedMemory.Allocate<float>(RB_TILE);

        int tileIdx = Grid.IdxX;
        int tileRow = tileIdx / numTilesN;
        int tileCol = tileIdx % numTilesN;
        int threadRow = Group.IdxX / RB_BLOCK;
        int threadCol = Group.IdxX % RB_BLOCK;

        float c00 = 0, c01 = 0, c02 = 0, c03 = 0, c10 = 0, c11 = 0, c12 = 0, c13 = 0;
        float c20 = 0, c21 = 0, c22 = 0, c23 = 0, c30 = 0, c31 = 0, c32 = 0, c33 = 0;

        int numKTiles = (K + RB_BLOCK - 1) / RB_BLOCK;
        for (int t = 0; t < numKTiles; t++)
        {
            for (int r = 0; r < RB_REG; r++)
            {
                int aRow = tileRow * RB_TILE + threadRow * RB_REG + r;
                int aCol = t * RB_BLOCK + threadCol;
                aTile[(threadRow * RB_REG + r) * RB_BLOCK + threadCol] = (aRow < M && aCol < K) ? A[aRow * K + aCol] : 0f;
            }
            if (Group.IdxX < RB_TILE)
            {
                int metaCol = Group.IdxX;
                int n = tileCol * RB_TILE + metaCol;
                float dsc = 0f;
                if (n < N)
                {
                    int k0 = t * RB_BLOCK;
                    int sbOff = n * bytesPerRow + (k0 >> 8) * 210;
                    int r = k0 & 255;
                    int half = r >> 7;
                    int rh = r & 127;
                    int variant = rh >> 5;
                    int l = rh & 31;
                    float md = HalfToFloatFinite(ReadByte(w, sbOff + 208) | (ReadByte(w, sbOff + 209) << 8));
                    float msc = SignExtend8(ReadByte(w, sbOff + 192 + 8 * half + (l >> 4) + 2 * variant));
                    dsc = md * msc; // d·sc folded once per column (was recomputed per element)
                }
                bMeta[metaCol] = dsc;
            }
            Group.Barrier();
            for (int r = 0; r < RB_REG; r++)
            {
                int bRow = t * RB_BLOCK + threadRow;
                int metaCol = threadCol * RB_REG + r;
                int bCol = tileCol * RB_TILE + metaCol;
                int sbOff = bCol * bytesPerRow + (bRow >> 8) * 210;
                bTile[threadRow * RB_TILE + metaCol] =
                    (bRow < K && bCol < N) ? DecodeQ6KNibble(w, sbOff, bRow & 255, bMeta[metaCol]) : 0f;
            }
            Group.Barrier();
            for (int k = 0; k < RB_BLOCK; k++)
            {
                float a0 = aTile[(threadRow * RB_REG + 0) * RB_BLOCK + k];
                float a1 = aTile[(threadRow * RB_REG + 1) * RB_BLOCK + k];
                float a2 = aTile[(threadRow * RB_REG + 2) * RB_BLOCK + k];
                float a3 = aTile[(threadRow * RB_REG + 3) * RB_BLOCK + k];
                float b0 = bTile[k * RB_TILE + threadCol * RB_REG + 0];
                float b1 = bTile[k * RB_TILE + threadCol * RB_REG + 1];
                float b2 = bTile[k * RB_TILE + threadCol * RB_REG + 2];
                float b3 = bTile[k * RB_TILE + threadCol * RB_REG + 3];
                c00 += a0 * b0; c01 += a0 * b1; c02 += a0 * b2; c03 += a0 * b3;
                c10 += a1 * b0; c11 += a1 * b1; c12 += a1 * b2; c13 += a1 * b3;
                c20 += a2 * b0; c21 += a2 * b1; c22 += a2 * b2; c23 += a2 * b3;
                c30 += a3 * b0; c31 += a3 * b1; c32 += a3 * b2; c33 += a3 * b3;
            }
            Group.Barrier();
        }
        WriteRegTile(C, M, N, tileRow * RB_TILE + threadRow * RB_REG, tileCol * RB_TILE + threadCol * RB_REG,
            c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22, c23, c30, c31, c32, c33);
    }

    private static void WriteRegTile(ArrayView1D<float, Stride1D.Dense> C, int M, int N, int baseRow, int baseCol,
        float c00, float c01, float c02, float c03, float c10, float c11, float c12, float c13,
        float c20, float c21, float c22, float c23, float c30, float c31, float c32, float c33)
    {
        if (baseRow + 0 < M && baseCol + 0 < N) C[(baseRow + 0) * N + baseCol + 0] = c00;
        if (baseRow + 0 < M && baseCol + 1 < N) C[(baseRow + 0) * N + baseCol + 1] = c01;
        if (baseRow + 0 < M && baseCol + 2 < N) C[(baseRow + 0) * N + baseCol + 2] = c02;
        if (baseRow + 0 < M && baseCol + 3 < N) C[(baseRow + 0) * N + baseCol + 3] = c03;
        if (baseRow + 1 < M && baseCol + 0 < N) C[(baseRow + 1) * N + baseCol + 0] = c10;
        if (baseRow + 1 < M && baseCol + 1 < N) C[(baseRow + 1) * N + baseCol + 1] = c11;
        if (baseRow + 1 < M && baseCol + 2 < N) C[(baseRow + 1) * N + baseCol + 2] = c12;
        if (baseRow + 1 < M && baseCol + 3 < N) C[(baseRow + 1) * N + baseCol + 3] = c13;
        if (baseRow + 2 < M && baseCol + 0 < N) C[(baseRow + 2) * N + baseCol + 0] = c20;
        if (baseRow + 2 < M && baseCol + 1 < N) C[(baseRow + 2) * N + baseCol + 1] = c21;
        if (baseRow + 2 < M && baseCol + 2 < N) C[(baseRow + 2) * N + baseCol + 2] = c22;
        if (baseRow + 2 < M && baseCol + 3 < N) C[(baseRow + 2) * N + baseCol + 3] = c23;
        if (baseRow + 3 < M && baseCol + 0 < N) C[(baseRow + 3) * N + baseCol + 0] = c30;
        if (baseRow + 3 < M && baseCol + 1 < N) C[(baseRow + 3) * N + baseCol + 1] = c31;
        if (baseRow + 3 < M && baseCol + 2 < N) C[(baseRow + 3) * N + baseCol + 2] = c32;
        if (baseRow + 3 < M && baseCol + 3 < N) C[(baseRow + 3) * N + baseCol + 3] = c33;
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Q6_K GEMV (M==1): same group-per-column + coalesced strided-k + shared-mem
    //  reduction as Q4_K, using the per-element DecodeQ6KElement layout inverse.
    // ─────────────────────────────────────────────────────────────────────────
    private static void GemvDequantQ6_KImpl(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<int, Stride1D.Dense> w,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int K = p[1], N = p[2];
        int n = Grid.IdxX;
        int tid = Group.IdxX;

        var sh = SharedMemory.Allocate<float>(GemvGroupSize);
        float partial = 0f;
        if (n < N)
        {
            int bytesPerRow = K / 256 * 210;
            int rowBase = n * bytesPerRow;
            // Block-structured (same as the Q4_K GEMV): decode the block's fp16 d ONCE per 256-block instead
            // of per element (Pathology #3); each thread owns 256/G=4 elements per block, same coalesced access.
            int numBlocks = K / 256;
            int perThread = 256 / GemvGroupSize;
            for (int blk = 0; blk < numBlocks; blk++)
            {
                int sbOff = rowBase + blk * 210;
                float d = HalfToFloatFinite(ReadByte(w, sbOff + 208) | (ReadByte(w, sbOff + 209) << 8));
                int kBase = blk * 256;
                for (int sub = 0; sub < perThread; sub++)
                {
                    int r = tid + sub * GemvGroupSize;
                    partial += input[kBase + r] * DecodeQ6KScaled(w, sbOff, r, d);
                }
            }
        }
        sh[tid] = partial;
        Group.Barrier();
        for (int stride = GemvGroupSize / 2; stride > 0; stride >>= 1)
        {
            if (tid < stride) sh[tid] += sh[tid + stride];
            Group.Barrier();
        }
        if (tid == 0 && n < N) output[n] = sh[0];
    }

    /// <summary>Decode element <paramref name="r"/> (0..255) within a Q6_K super-block at byte
    /// <paramref name="sbOff"/>, given the block's already-decoded fp16 <paramref name="d"/> (hoisted out of
    /// the per-element GEMV loop). Quant + sub-block scale math identical to <see cref="DecodeQ6KElement"/>.</summary>
    private static float DecodeQ6KScaled(ArrayView1D<int, Stride1D.Dense> w, int sbOff, int r, float d)
    {
        int half = r >> 7;
        int rh = r & 127;
        int variant = rh >> 5;
        int l = rh & 31;
        int qlBase = sbOff + 64 * half;
        int qhBase = sbOff + 128 + 32 * half;
        int scBase = sbOff + 192 + 8 * half;
        int qlByte = ReadByte(w, qlBase + l + (variant & 1) * 32);
        int qh = ReadByte(w, qhBase + l);
        int isHigh = variant >> 1;
        int qlNib = isHigh == 1 ? (qlByte >> 4) : (qlByte & 0xF);
        int qhBits = (qh >> (2 * variant)) & 3;
        int q = (qlNib | (qhBits << 4)) - 32;
        int sc = SignExtend8(ReadByte(w, scBase + (l >> 4) + 2 * variant));
        return d * sc * q;
    }

    /// <summary>Decode element <paramref name="col"/> of a Q6_K-quantized row whose blocks start at
    /// byte <paramref name="rowByteBase"/>. The single-element inverse of the FusedDequantQ6_KImpl block
    /// layout (210B/256: [ql:128][qh:64][scales:16×int8][d:fp16@208], two 128-elem halves). Used by the
    /// M==1 GEMV. PURE INTEGER ARITHMETIC + the branchless HalfToFloatFinite (no branches/selects), same
    /// WebGL-emitter discipline as DecodeQ4KElement.</summary>
    internal static float DecodeQ6KElement(
        ArrayView1D<int, Stride1D.Dense> w, int rowByteBase, int col)
    {
        int sbOff = rowByteBase + (col >> 8) * 210;
        int r = col & 255;
        int half = r >> 7;           // 0 or 1 (which 128-element half)
        int rh = r & 127;            // 0..127 within the half
        int variant = rh >> 5;       // 0..3 -> el l / l+32 / l+64 / l+96
        int l = rh & 31;             // 0..31

        int qlBase = sbOff + 64 * half;
        int qhBase = sbOff + 128 + 32 * half;
        int scBase = sbOff + 192 + 8 * half;

        // ql index: variant 0,2 use byte l; variant 1,3 use byte l+32.
        int qlByte = ReadByte(w, qlBase + l + (variant & 1) * 32);
        int qh = ReadByte(w, qhBase + l);
        // nibble: variant 0,1 -> low nibble; variant 2,3 -> high nibble.
        int isHigh = variant >> 1;
        int qlNib = isHigh == 1 ? (qlByte >> 4) : (qlByte & 0xF);
        // qh bits: variant v -> bits 2v..2v+1.
        int qhBits = (qh >> (2 * variant)) & 3;
        int q = ((qlNib | (qhBits << 4)) - 32);

        int sc = SignExtend8(ReadByte(w, scBase + (l >> 4) + 2 * variant));
        float d = HalfToFloatFinite(ReadByte(w, sbOff + 208) | (ReadByte(w, sbOff + 209) << 8));
        return d * sc * q;
    }

    // ── Q8_0 GEMV (M==1): 34B/32 = [d:fp16][32×int8], value = q·d ──
    private static void GemvDequantQ8_0Impl(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<int, Stride1D.Dense> w,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int K = p[1], N = p[2];
        int n = Grid.IdxX;
        int tid = Group.IdxX;
        var sh = SharedMemory.Allocate<float>(GemvGroupSize);
        float partial = 0f;
        if (n < N)
        {
            int rowBase = n * (K / 32 * 34);
            for (int k = tid; k < K; k += GemvGroupSize)
                partial += input[k] * DecodeQ8_0Element(w, rowBase, k);
        }
        sh[tid] = partial;
        Group.Barrier();
        for (int stride = GemvGroupSize / 2; stride > 0; stride >>= 1)
        {
            if (tid < stride) sh[tid] += sh[tid + stride];
            Group.Barrier();
        }
        if (tid == 0 && n < N) output[n] = sh[0];
    }

    /// <summary>Decode element <paramref name="col"/> of a Q8_0 row (34B/32: [d:fp16][32×int8]; value = q·d).</summary>
    internal static float DecodeQ8_0Element(ArrayView1D<int, Stride1D.Dense> w, int rowByteBase, int col)
    {
        int bOff = rowByteBase + (col >> 5) * 34;   // 32 values per block
        float d = HalfToFloatFinite(ReadByte(w, bOff) | (ReadByte(w, bOff + 1) << 8));
        int q = SignExtend8(ReadByte(w, bOff + 2 + (col & 31)));
        return q * d;
    }

    // ── Q4_0 GEMV (M==1): 18B/32 = [d:fp16][16 packed nibbles]; el j=low nibble of byte j, el j+16=high; value=(nib-8)·d ──
    private static void GemvDequantQ4_0Impl(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<int, Stride1D.Dense> w,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int K = p[1], N = p[2];
        int n = Grid.IdxX;
        int tid = Group.IdxX;
        var sh = SharedMemory.Allocate<float>(GemvGroupSize);
        float partial = 0f;
        if (n < N)
        {
            int rowBase = n * (K / 32 * 18);
            for (int k = tid; k < K; k += GemvGroupSize)
                partial += input[k] * DecodeQ4_0Element(w, rowBase, k);
        }
        sh[tid] = partial;
        Group.Barrier();
        for (int stride = GemvGroupSize / 2; stride > 0; stride >>= 1)
        {
            if (tid < stride) sh[tid] += sh[tid + stride];
            Group.Barrier();
        }
        if (tid == 0 && n < N) output[n] = sh[0];
    }

    // MXFP4 GEMV (M==1 decode hot path): one thread GROUP per output column n, coalesced strided-k read +
    // shared-mem tree reduction. Mirror of GemvDequantQ4_0Impl with the MXFP4 row stride (17B/block) and
    // decode. Browser-GPU backends are excluded by the caller (WebGL = no shared mem; WebGPU = perf) and
    // fall through to the per-element GEMM — same as every other type.
    private static void GemvDequantMXFP4Impl(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<int, Stride1D.Dense> w,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int K = p[1], N = p[2];
        int n = Grid.IdxX;
        int tid = Group.IdxX;
        var sh = SharedMemory.Allocate<float>(GemvGroupSize);
        float partial = 0f;
        if (n < N)
        {
            int rowBase = n * (K / 32 * 17);
            for (int k = tid; k < K; k += GemvGroupSize)
                partial += input[k] * DecodeMXFP4Element(w, rowBase, k);
        }
        sh[tid] = partial;
        Group.Barrier();
        for (int stride = GemvGroupSize / 2; stride > 0; stride >>= 1)
        {
            if (tid < stride) sh[tid] += sh[tid + stride];
            Group.Barrier();
        }
        if (tid == 0 && n < N) output[n] = sh[0];
    }

    /// <summary>Decode element <paramref name="col"/> of an MXFP4 row (17B/32: [e:E8M0][16 nibble bytes];
    /// el j = low nibble of byte j, el j+16 = high nibble of byte j; value = E2M1[nibble]·2^(e-127)).
    /// Shared by the MXFP4 GEMV; same layout inverse as the GEMM kernel.</summary>
    internal static float DecodeMXFP4Element(ArrayView1D<int, Stride1D.Dense> w, int rowByteBase, int col)
    {
        int within = col & 31;
        int bOff = rowByteBase + (col >> 5) * 17;
        float d = Float8E8M0Extensions.RawBitsToFloat(ReadByte(w, bOff));
        int packed = ReadByte(w, bOff + 1 + (within & 15));
        int nib = (within >> 4) == 1 ? (packed >> 4) : (packed & 0xF);  // within>=16 -> high nibble
        return Float4E2M1Extensions.RawBitsToFloat(nib) * d;
    }

    /// <summary>Decode element <paramref name="col"/> of a Q4_0 row (18B/32: [d:fp16][16 nibble bytes];
    /// el j = low nibble of byte j, el j+16 = high nibble of byte j; value = (nibble-8)·d).</summary>
    internal static float DecodeQ4_0Element(ArrayView1D<int, Stride1D.Dense> w, int rowByteBase, int col)
    {
        int within = col & 31;                       // 0..31 within the 32-value block
        int bOff = rowByteBase + (col >> 5) * 18;
        float d = HalfToFloatFinite(ReadByte(w, bOff) | (ReadByte(w, bOff + 1) << 8));
        int packed = ReadByte(w, bOff + 2 + (within & 15));
        int nib = (within >> 4) == 1 ? (packed >> 4) : (packed & 0xF);  // within>=16 -> high nibble
        return (nib - 8) * d;
    }

    /// <summary>Decode element <paramref name="col"/> of a Q4_K-quantized row whose
    /// blocks start at byte <paramref name="rowByteBase"/>. Shared by the fused MatMul
    /// and Gather kernels - ONE copy of the layout inverse. PURE INTEGER ARITHMETIC,
    /// zero branches AND zero selects: a single ternary in a dynamic-loop body multiplies
    /// the WebGL GLSL emitter's block duplication (5 selects here = 4.7MB of GLSL;
    /// this form ≈ 60KB; measured 2026-06-11 - see the kernel SHAPE WARNING above).</summary>
    internal static float DecodeQ4KElement(
        ArrayView1D<int, Stride1D.Dense> w, int rowByteBase, int col)
    {
        int sbOff = rowByteBase + (col >> 8) * 144;
        int r = col & 255;
        // chunk t = r/64; within-chunk c = r%64; lane l = c%32; hi = (c>=32) as 0/1.
        int t = r >> 6;
        int c = r & 63;
        int l = c & 31;
        int hi = (c >> 5) & 1;

        float d = HalfToFloatFinite(ReadByte(w, sbOff) | (ReadByte(w, sbOff + 1) << 8));
        float dmin = HalfToFloatFinite(ReadByte(w, sbOff + 2) | (ReadByte(w, sbOff + 3) << 8));

        // ggml get_scale_min_k4(j) for j = 2t + hi, as mask arithmetic. j in 0..7;
        // lowBit = 1 when j < 4 (sign of j-4), else 0:
        //   low form  (j<4):  sc = s[j] & 63;                    m = s[j+4] & 63
        //   high form (j>=4): sc = (s[j+4]&0xF) | ((s[j-4]>>6)<<4); m = (s[j+4]>>4) | ((s[j]>>6)<<4)
        int scOff = sbOff + 4;
        int j = 2 * t + hi;
        int lowBit = ((j - 4) >> 31) & 1;
        int hiBit = 1 - lowBit;
        int bj = ReadByte(w, scOff + j);
        int bj4 = ReadByte(w, scOff + j + 4);
        int bjAlt = ReadByte(w, scOff + j - 4 * hiBit); // j (low) or j-4 (high), always in range
        int sc = lowBit * (bj & 63) + hiBit * ((bj4 & 0xF) | ((bjAlt >> 6) << 4));
        int mn = lowBit * (bj4 & 63) + hiBit * ((bj4 >> 4) | ((bj >> 6) << 4));

        int packed = ReadByte(w, sbOff + 16 + 32 * t + l);
        int nibble = (packed >> (4 * hi)) & 0xF;
        return d * sc * nibble - dmin * mn;
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Q6_K: 210 bytes per 256 values — [ql:128B][qh:64B][scales:16×int8][d:fp16]
    //  Two 128-element halves (ql+=64, qh+=32, sc+=8). Within a half, for l in
    //  0..31: el l = ql[l].lo|qh[l]&3; el l+32 = ql[l+32].lo|qh bits 2-3;
    //  el l+64 = ql[l].hi|qh bits 4-5; el l+96 = ql[l+32].hi|qh bits 6-7;
    //  all −32, scale sc[l/16 + {0,2,4,6}] (signed). value = d·sc·q.
    // ─────────────────────────────────────────────────────────────────────────
    private static void FusedDequantQ6_KImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<int, Stride1D.Dense> w,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int M = p[0], K = p[1], N = p[2];
        int m = idx / N;
        int n = idx % N;
        if (m >= M) return;

        int sbPerRow = K / 256;
        int bytesPerRow = sbPerRow * 210;
        int inBase = m * K;

        float sum = 0f;
        for (int sb = 0; sb < sbPerRow; sb++)
        {
            int sbOff = n * bytesPerRow + sb * 210;
            float d = HalfToFloat(ReadByte(w, sbOff + 208) | (ReadByte(w, sbOff + 209) << 8));
            int ql = sbOff;
            int qh = sbOff + 128;
            int sc = sbOff + 192;
            int kBase = sb * 256;

            for (int half = 0; half < 2; half++)
            {
                int y = kBase + half * 128;
                for (int l = 0; l < 32; l++)
                {
                    int isIdx = l / 16;
                    int hb = ReadByte(w, qh + l);
                    int lo0 = ReadByte(w, ql + l);
                    int lo32 = ReadByte(w, ql + l + 32);
                    int q1 = ((lo0 & 0xF) | ((hb & 3) << 4)) - 32;
                    int q2 = ((lo32 & 0xF) | (((hb >> 2) & 3) << 4)) - 32;
                    int q3 = ((lo0 >> 4) | (((hb >> 4) & 3) << 4)) - 32;
                    int q4 = ((lo32 >> 4) | (((hb >> 6) & 3) << 4)) - 32;
                    float s0 = SignExtend8(ReadByte(w, sc + isIdx));
                    float s2 = SignExtend8(ReadByte(w, sc + isIdx + 2));
                    float s4 = SignExtend8(ReadByte(w, sc + isIdx + 4));
                    float s6 = SignExtend8(ReadByte(w, sc + isIdx + 6));
                    sum += input[inBase + y + l] * (d * s0 * q1);
                    sum += input[inBase + y + l + 32] * (d * s2 * q2);
                    sum += input[inBase + y + l + 64] * (d * s4 * q3);
                    sum += input[inBase + y + l + 96] * (d * s6 * q4);
                }
                ql += 64; qh += 32; sc += 8;
            }
        }
        output[idx] = sum;
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Helpers (inlined into kernels by the ILGPU frontend)
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>ggml get_scale_min_k4: j-th 6-bit (scale, min) pair from a Q4_K/Q5_K
    /// 12-byte scales array. Packed return: scale in bits 0-7, min in bits 8-15
    /// (kernels cannot use out params on all backends). Shared with
    /// <see cref="FusedDequantGather"/> - keep ONE copy of every block-decode helper.</summary>
    internal static int GetScaleMinK4(ArrayView1D<int, Stride1D.Dense> w, int scOff, int j)
    {
        // BRANCHLESS (ternary selects only, no if/else blocks): an if/else here, inlined
        // into the unrolled constant chunk loops of the K-quant kernels, triggered a
        // WebGL GLSL codegen path explosion - 8 sequential two-way branches per
        // super-block blew FusedDequantQ4_KImpl up to 31.5MB of GLSL (447k lines) and
        // OOM'd the browser marshaling it to the worker. Selects emit as expressions,
        // and they are the better GPU form anyway (no divergence).
        bool low = j < 4;
        int bj = ReadByte(w, scOff + j);
        int bj4 = ReadByte(w, scOff + j + 4);
        int bjAlt = ReadByte(w, scOff + (low ? j : j - 4));
        int sc = low ? (bj & 63) : ((bj4 & 0xF) | ((bjAlt >> 6) << 4));
        int mn = low ? (bj4 & 63) : ((bj4 >> 4) | ((bj >> 6) << 4));
        return sc | (mn << 8);
    }

    /// <summary>Extract a single byte from a packed int32 array.</summary>
    internal static int ReadByte(ArrayView1D<int, Stride1D.Dense> packed, int byteIndex)
    {
        int word = packed[byteIndex / 4];
        return (word >> ((byteIndex % 4) * 8)) & 0xFF;
    }

    /// <summary>Sign-extend an unsigned byte value (0..255) to a signed int — explicit
    /// int-domain arithmetic; sbyte reinterpret sign-extension is not reliable across
    /// browser backends.</summary>
    internal static int SignExtend8(int b) => (b & 0x80) != 0 ? b - 256 : b;

    /// <summary>Branchless FP16-bits → float for PER-ELEMENT loop bodies. Handles
    /// normals, subnormals, and signed zero exactly; Inf/NaN map to a large finite
    /// value (a valid GGUF weight scale is never Inf/NaN - the full-semantics
    /// <see cref="HalfToFloat"/> stays for per-block use). Zero branches/selects:
    /// the branchy version inlined twice per loop iteration multiplied the WebGL
    /// GLSL emitter's block duplication (1.2MB -> ~70KB; measured 2026-06-11).</summary>
    internal static float HalfToFloatFinite(int h)
    {
        int sign = (h >> 15) & 1;
        int exp = (h >> 10) & 0x1F;
        int mant = h & 0x3FF;
        int isNorm = ((-exp) >> 31) & 1; // 1 when exp > 0
        float frac = mant * (1f / 1024f);
        float magNorm = (1f + frac) * MathF.Pow(2f, exp - 15);
        float magSub = frac * (1f / 16384f);
        return (1 - 2 * sign) * (isNorm * magNorm + (1 - isNorm) * magSub);
    }

    /// <summary>Convert FP16 bits to float.</summary>
    internal static float HalfToFloat(int h)
    {
        int sign = (h >> 15) & 1;
        int exp = (h >> 10) & 0x1F;
        int mant = h & 0x3FF;

        if (exp == 0)
        {
            if (mant == 0) return sign == 1 ? -0f : 0f;
            // Subnormal
            float val = mant / 1024f * (1f / 16384f);
            return sign == 1 ? -val : val;
        }
        if (exp == 31)
        {
            return mant == 0
                ? (sign == 1 ? float.NegativeInfinity : float.PositiveInfinity)
                : float.NaN;
        }

        float result = (1f + mant / 1024f) * MathF.Pow(2, exp - 15);
        return sign == 1 ? -result : result;
    }

    public void Dispose()
    {
        foreach (var buf in _paramsBufs.Values) buf.Dispose();
        _paramsBufs.Clear();
        foreach (var buf in _actQs.Values) buf.Dispose();
        _actQs.Clear();
        foreach (var buf in _actDs.Values) buf.Dispose();
        _actDs.Clear();
    }
}
