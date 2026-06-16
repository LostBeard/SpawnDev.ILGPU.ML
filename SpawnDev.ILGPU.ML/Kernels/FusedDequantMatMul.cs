using ILGPU;
using ILGPU.Runtime;
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
        ArrayView1D<int, Stride1D.Dense>>? _kernelQ4_0, _kernelQ8_0, _kernelQ4_K, _kernelQ6_K;

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
        ArrayView1D<int, Stride1D.Dense>>? _gemvQ4_K, _gemvQ6_K, _gemvQ8_0, _gemvQ4_0;

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
        type is GGMLType.Q4_0 or GGMLType.Q8_0 or GGMLType.Q4_K or GGMLType.Q6_K;

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
            }
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
        float partial = 0f;
        if (n < N)
        {
            int bytesPerRow = K / 256 * 144;
            int rowBase = n * bytesPerRow;
            // BLOCK-STRUCTURED: with GemvGroupSize=64 each thread owns exactly 256/64=4 elements per 256-block
            // (r = tid, tid+64, tid+128, tid+192 — same coalesced access as the strided-k loop). The block's
            // fp16 d/dmin are IDENTICAL for all 256 elements, so decode them ONCE per block (was per element =
            // 2 HalfToFloatFinite × ~20 ALU ops wasted 256x — Pathology #3). The per-element work is then just
            // the 6-bit sub-block scale + the nibble.
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
                    partial += input[kBase + r] * DecodeQ4KScaled(w, sbOff, r, d, dmin);
                }
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
    }
}
