using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Text;

namespace SpawnDev.ILGPU.ML.Ltx2;

/// <summary>
/// The random stream of ComfyUI's seeded fp8 re-rounding of a LoRA-patched weight, reproduced BLACK BOX (measurements only, never ComfyUI's source; method and
/// evidence: Plans/ltx-2-19b-2026-11-17.md, tools/ltx2-fp8-seed-search). The whole rule, for a weight whose checkpoint tensor is F8_E4M3:
/// <list type="number">
/// <item>w = W + strength x (B A) in float32; the new scale = max|w| / 448;</item>
/// <item>y = float16(|w| / scale): the scaled weight passes through half precision;</item>
/// <item>l &lt;= y &lt; h the F8_E4M3 neighbours, frac = (y - l) / (h - l);</item>
/// <item>u_i = (word_i AND 0xFF) / 256 for element i in row-major order, the words from MT19937 seeded (init_genrand) with the CRC-32 of the UTF-8 key
/// <c>diffusion_model.&lt;tensor&gt;.weight</c>;</item>
/// <item>stored = sign(w) x (h if frac + u &gt;= 1 else l).</item>
/// </list>
/// MEASURED 2026-11-17 on every element of two 4096 x 4096 tensors (attn1.to_q at ten LoRA strengths, attn1.to_k): 33 to 70 of 16,777,216 elements differ (the last
/// float32 bit of B A, by summation order). The device side is <see cref="Ltx2Kernels.Fp8StochasticRound"/>.
/// </summary>
public static class Ltx2Fp8Rounding
{
    private static readonly uint[] CrcTable = BuildCrcTable();
    private static uint[] BuildCrcTable()
    {
        var t = new uint[256];
        for (uint n = 0; n < 256; n++)
        {
            uint c = n;
            for (int k = 0; k < 8; k++) c = (c & 1u) != 0u ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            t[n] = c;
        }
        return t;
    }

    /// <summary>CRC-32 (zlib's) of the text's UTF-8 bytes.</summary>
    public static uint Crc32(string text)
    {
        uint c = 0xFFFFFFFFu;
        foreach (byte b in Encoding.UTF8.GetBytes(text)) c = CrcTable[(c ^ b) & 0xFFu] ^ (c >> 8);
        return c ^ 0xFFFFFFFFu;
    }

    /// <summary>The key a transformer tensor's stream is seeded by: <c>diffusion_model.&lt;name&gt;.weight</c> (name without a prefix: <c>transformer_blocks.1.attn1.to_q</c>).</summary>
    public static string Key(string tensorName) => "diffusion_model." + tensorName + ".weight";

    /// <summary>The seed of a tensor's stream.</summary>
    public static uint Seed(string tensorName) => Crc32(Key(tensorName));

    /// <summary>
    /// The first <paramref name="count"/> 8-bit draws of MT19937 seeded with <paramref name="seed"/> (each the low 8 bits of one 32-bit output), packed four to a
    /// word into <paramref name="packed"/> (draw i in bits 8 (i AND 3) of word i / 4): the layout <see cref="Ltx2Kernels.Fp8StochasticRound"/> reads.
    /// </summary>
    public static void Low8(uint seed, Span<uint> packed, long count)
    {
        const int N = 624, M = 397;
        long words = (count + 3) / 4;
        if (packed.Length < words) throw new ArgumentException($"{count} draws need {words} words");
        if (!BitConverter.IsLittleEndian || !Vector128.IsHardwareAccelerated) { Low8Scalar(seed, packed, count); return; }
        // Vectorized: draw i is byte i of the packed words (little endian). The generator makes its 624 words a state refill at a time: the refill's first 227 words
        // read only the old state, the rest read words written 227 earlier, so both run eight at a time; the output mix is elementwise.
        // MEASURED 2026-11-17 (12 cores' one core): 4.0 ns a draw in the plain form (1.61 s for one block's 28 patched matrices), see Ltx2_Fp8CompatStream_Speed for this one.
        var bytes = MemoryMarshal.AsBytes(packed[..(int)words]);
        Span<uint> mt = stackalloc uint[N];
        mt[0] = seed;
        for (int i = 1; i < N; i++) mt[i] = 1812433253u * (mt[i - 1] ^ (mt[i - 1] >> 30)) + (uint)i;
        Span<byte> tail = stackalloc byte[N];
        var upper = Vector256.Create(0x80000000u); var lower = Vector256.Create(0x7FFFFFFFu); var matrix = Vector256.Create(0x9908B0DFu); var one = Vector256.Create(1u);
        var tb = Vector256.Create(0x9D2C5680u); var tc = Vector256.Create(0xEFC60000u); var low8 = Vector256.Create(0xFFu);
        ref uint s = ref MemoryMarshal.GetReference(mt);
        for (long done = 0; done < count; done += N)
        {
            int i = 0;
            for (; i + 8 <= N - M; i += 8)
            {
                var a = Vector256.LoadUnsafe(ref s, (nuint)i); var b = Vector256.LoadUnsafe(ref s, (nuint)(i + 1));
                var y = (a & upper) | (b & lower);
                (Vector256.LoadUnsafe(ref s, (nuint)(i + M)) ^ (y >>> 1) ^ (Vector256.Equals(y & one, one) & matrix)).StoreUnsafe(ref s, (nuint)i);
            }
            for (; i < N - M; i++) { uint y = (mt[i] & 0x80000000u) | (mt[i + 1] & 0x7FFFFFFFu); mt[i] = mt[i + M] ^ (y >> 1) ^ ((y & 1u) != 0u ? 0x9908B0DFu : 0u); }
            for (; i + 9 <= N; i += 8)                                                      // words i + 1 .. i + 8 are still the old state: i + 8 <= 623
            {
                var a = Vector256.LoadUnsafe(ref s, (nuint)i); var b = Vector256.LoadUnsafe(ref s, (nuint)(i + 1));
                var y = (a & upper) | (b & lower);
                (Vector256.LoadUnsafe(ref s, (nuint)(i + M - N)) ^ (y >>> 1) ^ (Vector256.Equals(y & one, one) & matrix)).StoreUnsafe(ref s, (nuint)i);
            }
            for (; i < N - 1; i++) { uint y = (mt[i] & 0x80000000u) | (mt[i + 1] & 0x7FFFFFFFu); mt[i] = mt[i + M - N] ^ (y >> 1) ^ ((y & 1u) != 0u ? 0x9908B0DFu : 0u); }
            { uint y = (mt[N - 1] & 0x80000000u) | (mt[0] & 0x7FFFFFFFu); mt[N - 1] = mt[M - 1] ^ (y >> 1) ^ ((y & 1u) != 0u ? 0x9908B0DFu : 0u); }
            // the output mix, 16 words to 16 bytes (624 = 39 x 16); a last partial refill goes through the stack
            long left = count - done;
            var dst = left >= N ? bytes.Slice((int)done, N) : tail;
            ref byte d = ref MemoryMarshal.GetReference(dst);
            for (int j = 0; j < N; j += 16)
            {
                var v0 = Vector256.LoadUnsafe(ref s, (nuint)j); var v1 = Vector256.LoadUnsafe(ref s, (nuint)(j + 8));
                v0 ^= v0 >>> 11; v0 ^= (v0 << 7) & tb; v0 ^= (v0 << 15) & tc; v0 ^= v0 >>> 18;
                v1 ^= v1 >>> 11; v1 ^= (v1 << 7) & tb; v1 ^= (v1 << 15) & tc; v1 ^= v1 >>> 18;
                // WORKAROUND, tracked publicly: https://github.com/LostBeard/SpawnDev.ILGPU.ML/issues/3 and Docs/workarounds.md (approved by TJ 2026-11-18).
                // A .NET runtime bug, not ours (upstream report still to be filed): the low byte is masked out BEFORE the narrowing, so a narrowing that
                // SATURATES gives the same bytes as one that truncates. Vector128.Narrow must truncate, and does on desktop and in the browser's interpreter, but
                // once this loop is hot the browser runtime (Mono WebAssembly, .NET 10: its hot-path compiler) returns saturated values: FF for every word above
                // 255. MEASURED 2026-11-17 on all three browser lanes: the unmasked form is right for 342 rounds, then 4,756,832 of the next 4.8 M bytes are FF
                // (the canary in Ltx2_Fp8CompatStream_MatchesPlainFormAndUploads reports it on every run: drop the mask when it says "equals the plain mix").
                v0 &= low8; v1 &= low8;
                Vector128.Narrow(Vector128.Narrow(v0.GetLower(), v0.GetUpper()), Vector128.Narrow(v1.GetLower(), v1.GetUpper())).StoreUnsafe(ref d, (nuint)j);
            }
            if (left < N) tail[..(int)left].CopyTo(bytes.Slice((int)done, (int)left));
        }
        bytes[(int)count..].Clear();                                                        // the last word's unused draws
    }

    /// <summary>The plain form of <see cref="Low8"/> (one word at a time): what it falls back to without vector hardware, and the reference it is tested against.</summary>
    public static void Low8Scalar(uint seed, Span<uint> packed, long count)
    {
        const int N = 624, M = 397;
        long words = (count + 3) / 4;
        if (packed.Length < words) throw new ArgumentException($"{count} draws need {words} words");
        Span<uint> mt = stackalloc uint[N];
        mt[0] = seed;
        for (int i = 1; i < N; i++) mt[i] = 1812433253u * (mt[i - 1] ^ (mt[i - 1] >> 30)) + (uint)i;
        int idx = N;
        long e = 0;
        for (long wi = 0; wi < words; wi++)
        {
            uint word = 0;
            for (int k = 0; k < 4 && e < count; k++, e++)
            {
                if (idx == N)
                {
                    for (int i = 0; i < N - M; i++) { uint y = (mt[i] & 0x80000000u) | (mt[i + 1] & 0x7FFFFFFFu); mt[i] = mt[i + M] ^ (y >> 1) ^ ((y & 1u) != 0u ? 0x9908B0DFu : 0u); }
                    for (int i = N - M; i < N - 1; i++) { uint y = (mt[i] & 0x80000000u) | (mt[i + 1] & 0x7FFFFFFFu); mt[i] = mt[i + M - N] ^ (y >> 1) ^ ((y & 1u) != 0u ? 0x9908B0DFu : 0u); }
                    { uint y = (mt[N - 1] & 0x80000000u) | (mt[0] & 0x7FFFFFFFu); mt[N - 1] = mt[M - 1] ^ (y >> 1) ^ ((y & 1u) != 0u ? 0x9908B0DFu : 0u); }
                    idx = 0;
                }
                uint v = mt[idx++];
                v ^= v >> 11; v ^= (v << 7) & 0x9D2C5680u; v ^= (v << 15) & 0xEFC60000u; v ^= v >> 18;
                word |= (v & 0xFFu) << (k << 3);
            }
            packed[(int)wi] = word;
        }
    }

    /// <summary>
    /// The rule on the host, one element (the reference the kernel is tested against): the F8_E4M3 value stored for the patched weight <paramref name="w"/> at
    /// <paramref name="scale"/> with the 8-bit draw <paramref name="draw"/> (0..255).
    /// </summary>
    public static float Round(float w, float scale, int draw)
    {
        float y = (float)(Half)(MathF.Abs(w) / scale);                                    // float32 division, then round to nearest even into half precision
        float l, h, frac;
        if (y >= 0.015625f)                                                               // normal F8_E4M3: 1.mmm x 2^e
        {
            uint b = BitConverter.SingleToUInt32Bits(y), lb = b & 0xFFF00000u;
            l = BitConverter.UInt32BitsToSingle(lb); h = BitConverter.UInt32BitsToSingle(lb + 0x100000u); frac = (b & 0xFFFFFu) * (1f / 1048576f);
        }
        else { float sc = y * 512f, t = MathF.Floor(sc); frac = sc - t; l = t * 0.001953125f; h = (t + 1f) * 0.001953125f; }   // below 2^-6: steps of 2^-9 from 0
        float q = frac * 256f + draw >= 256f ? h : l;
        return w < 0f ? -q : q;
    }

    /// <summary>
    /// The F8_E4M3 byte of <paramref name="v"/>, a value ON the E4M3 grid (what <see cref="Round"/> and the device rounding produce: |v| &lt;= 448, normal 1.mmm x 2^e
    /// with e in -6..8 or a multiple of 2^-9 below 2^-6); the sign bit follows the float's, so -0 is 0x80. Throws for a value off the grid.
    /// </summary>
    public static byte ToE4M3(float v)
    {
        uint bits = BitConverter.SingleToUInt32Bits(v), sign = (bits >> 24) & 0x80u;
        float a = MathF.Abs(v);
        if (a == 0f) return (byte)sign;
        if (a < 0.015625f)
        {
            float m = a * 512f;                                                               // subnormal E4M3: m x 2^-9, m = 1..7
            if (m != MathF.Floor(m) || m < 1f || m > 7f) throw new ArgumentException($"{v:R} is not an F8_E4M3 value");
            return (byte)(sign | (uint)m);
        }
        uint ab = bits & 0x7FFFFFFFu;
        int e = (int)(ab >> 23) - 127 + 7;
        if ((ab & 0xFFFFFu) != 0 || e < 1 || e > 15 || (e == 15 && ((ab >> 20) & 7u) == 7u)) throw new ArgumentException($"{v:R} is not an F8_E4M3 value");
        return (byte)(sign | ((uint)e << 3) | ((ab >> 20) & 7u));
    }
}
