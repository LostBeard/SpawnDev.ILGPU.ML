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
