using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.SkinTokens;
using SpawnDev.ILGPU.ML.Trellis2;

namespace SpawnDev.ILGPU.ML.Flux2;

using FV = ArrayView1D<float, Stride1D.Dense>;

/// <summary>
/// FLUX.2 klein's flow transformer (3.9B parameters, bf16 weights, fp32 activations): width 3072 = 24 heads of 128, 5 double-stream blocks (separate text and image streams joined only in
/// attention) then 20 single-stream blocks (one stream, attention and MLP in parallel), shared adaLN modulation (one vector from the timestep, not one per block), 4-axis rotary positions
/// (32 dims each, theta 2000, interleaved pairs) over (t, h, w, l) ids, no biases, no guidance embedding (the klein checkpoints are step-distilled).
/// <code>
/// temb = linear_2(silu(linear_1(sinusoid(1000 t))))                      mod_img, mod_txt = double_stream_modulation_{img,txt}(silu(temb)),  mod_single = single_stream_modulation(silu(temb))
/// img = x_embedder(latents),  txt = context_embedder(prompt_embeds)
/// 5 x double { a_img, a_txt = joint_attention(rope(qk_norm(qkv(adaln(img))), qk_norm(add_qkv(adaln(txt)))))
///              img += gate * to_out(a_img) ; img += gate * ff(adaln(img)) ;   txt likewise with to_add_out / ff_context }
/// h = [txt | img];  20 x single { h += gate * to_out([attention(rope(qk_norm(q, k), v)) | swiglu(mlp)]) }  where (q, k, v, mlp) = to_qkv_mlp_proj(adaln(h))
/// out = proj_out(adaln_continuous(h[txt:], temb))
/// </code>
/// Weights from tools/unirig-harness/reference/export_flux2.py (transformer.safetensors). Checked against the reference's own step outputs (golden_klein.py).
/// </summary>
public sealed class Flux2Transformer : IDisposable
{
    public const int Dim = 3072, Heads = 24, HeadDim = 128, MlpHidden = 9216, DoubleBlocks = 5, SingleBlocks = 20, InChannels = 128, JointDim = 7680, TimeChannels = 256;
    public const float Eps = 1e-6f, RopeTheta = 2000f;
    public static readonly int[] RopeAxes = { 32, 32, 32, 32 };

    private readonly Accelerator _acc;
    private readonly Flux2Weights _w;
    private readonly DiTKernels _k;
    private readonly SkinTokensKernels _u;
    private readonly Flux2Kernels _f;
    private readonly int _nTxt, _nImg, _n;
    private readonly List<IDisposable> _owned = new();
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _img, _txt, _normI, _normT, _qkv, _attn, _attnImg, _attnTxt, _proj, _gateUp, _act, _single, _scores;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _temb, _tembAct, _tproj, _thid, _modImg, _modTxt, _modSingle, _modOut, _cosT, _sinT, _out, _inLat, _inTxt;

    /// <summary>
    /// The timestep the model sees is rounded to bf16 three times by the reference (t as a bf16 tensor, then t/1000, then *1000), because the pipeline runs the whole model in bf16. Matching the reference
    /// means doing the same (default); turn it off to feed the exact float.
    /// </summary>
    public static bool RoundTimestepToBf16 = true;

    public Flux2Transformer(Accelerator accelerator, Flux2Weights weights, int textTokens, int imageTokens)
    {
        _acc = accelerator; _w = weights; _nTxt = textTokens; _nImg = imageTokens; _n = textTokens + imageTokens;
        _k = new DiTKernels(accelerator); _u = new SkinTokensKernels(accelerator); _f = new Flux2Kernels(accelerator);
        long N = _n, I = _nImg, T = _nTxt;
        _img = Alloc<float>(I * Dim); _txt = Alloc<float>(T * Dim); _normI = Alloc<float>(I * Dim); _normT = Alloc<float>(T * Dim);
        _qkv = Alloc<float>(N * 3 * Dim); _attn = Alloc<float>(N * Dim); _attnImg = Alloc<float>(I * Dim); _attnTxt = Alloc<float>(T * Dim);
        _proj = Alloc<float>(N * Dim); _gateUp = Alloc<float>(N * 2 * MlpHidden); _act = Alloc<float>(N * MlpHidden);
        _single = Alloc<float>(N * (3 * Dim + 2 * MlpHidden)); _scores = Alloc<float>(_k.UsesFlashAttention(HeadDim) ? 1 : (long)Heads * N * N);   // the flash kernels (tensor-core on CUDA, portable tiled elsewhere) need no score matrix
        _temb = Alloc<float>(Dim); _tembAct = Alloc<float>(Dim); _tproj = Alloc<float>(TimeChannels); _thid = Alloc<float>(Dim);
        _modImg = Alloc<float>(6 * Dim); _modTxt = Alloc<float>(6 * Dim); _modSingle = Alloc<float>(3 * Dim); _modOut = Alloc<float>(2 * Dim);
        _cosT = Alloc<float>(N * HeadDim); _sinT = Alloc<float>(N * HeadDim); _out = Alloc<float>(I * InChannels);
        _inLat = Alloc<float>(I * InChannels); _inTxt = Alloc<float>(T * JointDim);
    }

    private MemoryBuffer1D<T, Stride1D.Dense> Alloc<T>(long n) where T : unmanaged { var b = _acc.Allocate1D<T>(Math.Max(1, n)); _owned.Add(b); return b; }

    /// <summary>Device views of the last <see cref="Forward"/> result, [imageTokens, 128].</summary>
    public FV Output => _out.View;

    /// <summary>Intermediate checkpoints for tests: name, device view, rows, columns.</summary>
    public Action<string, FV, int, int>? Probe;

    private static float BF16Round(float f)
    {
        uint u = BitConverter.SingleToUInt32Bits(f);
        if ((u & 0x7FFFFFFFu) > 0x7F800000u) return f;
        u += 0x7FFFu + ((u >> 16) & 1u);
        return BitConverter.UInt32BitsToSingle(u & 0xFFFF0000u);
    }

    /// <summary>Rotary tables for (t, h, w, l) ids [rows, 4]: cos and sin [rows, 128], each axis' 16 frequencies repeated for the interleaved pairs.</summary>
    public static (float[] Cos, float[] Sin) RopeTables(int[] ids, int rows)
    {
        var cos = new float[rows * HeadDim]; var sin = new float[rows * HeadDim];
        for (int r = 0; r < rows; r++)
        {
            int o = 0;
            for (int a = 0; a < RopeAxes.Length; a++)
            {
                int dim = RopeAxes[a];
                for (int j = 0; j < dim / 2; j++)
                {
                    double freq = 1.0 / Math.Pow(RopeTheta, 2.0 * j / dim);
                    double ang = ids[r * 4 + a] * freq;
                    float c = (float)Math.Cos(ang), s = (float)Math.Sin(ang);
                    cos[r * HeadDim + o] = c; cos[r * HeadDim + o + 1] = c;
                    sin[r * HeadDim + o] = s; sin[r * HeadDim + o + 1] = s;
                    o += 2;
                }
            }
        }
        return (cos, sin);
    }

    /// <summary>x[rows, K] times w[N, K]^T: the general GEMM, the coalesced GEMV for one row (the timestep).</summary>
    private void Lin(FV x, string weight, FV o, int rows, int K, int N)
    {
        if (rows == 1) _u.Gemv(x, _w.Matrix(weight), null, o, K, N);
        else _k.Linear(x, _w.Matrix(weight), null, o, rows, K, N);
    }

    /// <summary>
    /// One denoising step's model call. <paramref name="latents"/> [imageTokens, 128] and <paramref name="promptEmbeds"/> [textTokens, 7680] are host arrays, <paramref name="t"/> the scheduler timestep
    /// (sigma * 1000), <paramref name="imgIds"/> / <paramref name="txtIds"/> the [rows, 4] position ids. The prediction is left in <see cref="Output"/>.
    /// </summary>
    public void Forward(float[] latents, float[] promptEmbeds, float t, int[] imgIds, int[] txtIds)
    {
        _inLat.View.SubView(0, (long)_nImg * InChannels).CopyFromCPU(latents);
        _inTxt.View.SubView(0, (long)_nTxt * JointDim).CopyFromCPU(promptEmbeds);
        var all = new int[_n * 4]; Array.Copy(txtIds, all, _nTxt * 4); Array.Copy(imgIds, 0, all, _nTxt * 4, _nImg * 4);
        var (cos, sin) = RopeTables(all, _n);
        _cosT.View.SubView(0, cos.Length).CopyFromCPU(cos); _sinT.View.SubView(0, sin.Length).CopyFromCPU(sin);
        Forward(_inLat.View, _inTxt.View, t);
    }

    /// <summary>The model call with the latents and prompt embeddings already on the device and the rotary tables set (see the host overload).</summary>
    public void Forward(FV latents, FV promptEmbeds, float t)
    {
        float tt = t;
        if (RoundTimestepToBf16) { tt = BF16Round(t); tt = BF16Round(tt / 1000f); tt = BF16Round(tt * 1000f); }
        else { tt = t; }
        // timestep embedding: sinusoid (cos | sin, 128 frequencies), linear_1, silu, linear_2
        var sin256 = new float[TimeChannels];
        for (int i = 0; i < TimeChannels / 2; i++)
        {
            double f = Math.Exp(-Math.Log(10000.0) * i / (TimeChannels / 2));
            double a = tt * f;
            sin256[i] = RoundTimestepToBf16 ? BF16Round((float)Math.Cos(a)) : (float)Math.Cos(a);
            sin256[TimeChannels / 2 + i] = RoundTimestepToBf16 ? BF16Round((float)Math.Sin(a)) : (float)Math.Sin(a);
        }
        _tproj.View.CopyFromCPU(sin256);
        Lin(_tproj.View, "time_guidance_embed.timestep_embedder.linear_1.weight", _thid.View, 1, TimeChannels, Dim);
        _k.Silu(_thid.View, _tembAct.View, Dim);
        Lin(_tembAct.View, "time_guidance_embed.timestep_embedder.linear_2.weight", _temb.View, 1, Dim, Dim);
        Probe?.Invoke("temb", _temb.View, 1, Dim);
        _k.Silu(_temb.View, _tembAct.View, Dim);
        Lin(_tembAct.View, "double_stream_modulation_img.linear.weight", _modImg.View, 1, Dim, 6 * Dim);
        Lin(_tembAct.View, "double_stream_modulation_txt.linear.weight", _modTxt.View, 1, Dim, 6 * Dim);
        Lin(_tembAct.View, "single_stream_modulation.linear.weight", _modSingle.View, 1, Dim, 3 * Dim);
        Lin(_tembAct.View, "norm_out.linear.weight", _modOut.View, 1, Dim, 2 * Dim);

        Lin(latents, "x_embedder.weight", _img.View, _nImg, InChannels, Dim);
        Lin(promptEmbeds, "context_embedder.weight", _txt.View, _nTxt, JointDim, Dim);
        Probe?.Invoke("embedded_img", _img.View, _nImg, Dim);
        Probe?.Invoke("embedded_txt", _txt.View, _nTxt, Dim);

        for (int l = 0; l < DoubleBlocks; l++)
        {
            DoubleBlock(l);
            Probe?.Invoke($"double{l + 1}_img", _img.View, _nImg, Dim);
            Probe?.Invoke($"double{l + 1}_txt", _txt.View, _nTxt, Dim);
        }
        // one stream: [txt | img]
        _proj.View.SubView(0, (long)_nTxt * Dim).CopyFrom(_txt.View.SubView(0, (long)_nTxt * Dim));
        _proj.View.SubView((long)_nTxt * Dim, (long)_nImg * Dim).CopyFrom(_img.View.SubView(0, (long)_nImg * Dim));
        // the joint stream lives in _proj; the single blocks normalise into _attn
        for (int l = 0; l < SingleBlocks; l++)
        {
            SingleBlock(l);
            if (Probe != null) Probe($"single{l + 1}", _proj.View, _n, Dim);
        }
        // final adaln_continuous on the image rows, then proj_out
        var imgRows = _proj.View.SubView((long)_nTxt * Dim, (long)_nImg * Dim);
        _k.LayerNormModulate(imgRows, _normI.View, _modOut.View, Dim, 0, _nImg, Dim, Eps);          // out = LN(x) * (1 + scale) + shift, chunk order (scale, shift)
        Lin(_normI.View, "proj_out.weight", _out.View, _nImg, Dim, InChannels);
    }

    private void DoubleBlock(int l)
    {
        string b = $"transformer_blocks.{l}.";
        // adaln: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp  (each Dim wide)
        _k.LayerNormModulate(_img.View, _normI.View, _modImg.View, 0, Dim, _nImg, Dim, Eps);
        _k.LayerNormModulate(_txt.View, _normT.View, _modTxt.View, 0, Dim, _nTxt, Dim, Eps);
        // joint qkv: rows [txt | img], 3 * Dim columns (q | k | v)
        Lin(_normT.View, b + "attn.add_qkv.weight", _qkv.View.SubView(0, (long)_nTxt * 3 * Dim), _nTxt, Dim, 3 * Dim);
        Lin(_normI.View, b + "attn.qkv.weight", _qkv.View.SubView((long)_nTxt * 3 * Dim, (long)_nImg * 3 * Dim), _nImg, Dim, 3 * Dim);
        var qkvT = _qkv.View.SubView(0, (long)_nTxt * 3 * Dim); var qkvI = _qkv.View.SubView((long)_nTxt * 3 * Dim, (long)_nImg * 3 * Dim);
        _u.RmsNormHeads(qkvT, _w.Vector(b + "attn.norm_added_q.weight"), _nTxt, 3 * Dim, 0, Heads, HeadDim, Eps);
        _u.RmsNormHeads(qkvT, _w.Vector(b + "attn.norm_added_k.weight"), _nTxt, 3 * Dim, Dim, Heads, HeadDim, Eps);
        _u.RmsNormHeads(qkvI, _w.Vector(b + "attn.norm_q.weight"), _nImg, 3 * Dim, 0, Heads, HeadDim, Eps);
        _u.RmsNormHeads(qkvI, _w.Vector(b + "attn.norm_k.weight"), _nImg, 3 * Dim, Dim, Heads, HeadDim, Eps);
        _f.RopeInterleaved(_qkv.View, _cosT.View, _sinT.View, _n, 3 * Dim, 0, Heads, HeadDim);
        _f.RopeInterleaved(_qkv.View, _cosT.View, _sinT.View, _n, 3 * Dim, Dim, Heads, HeadDim);
        _k.Attention(_qkv.View, 3 * Dim, 0, _qkv.View, 3 * Dim, Dim, 2 * Dim, _scores.View, _attn.View, _n, _n, Heads, HeadDim);
        Lin(_attn.View.SubView(0, (long)_nTxt * Dim), b + "attn.to_add_out.weight", _attnTxt.View, _nTxt, Dim, Dim);
        Lin(_attn.View.SubView((long)_nTxt * Dim, (long)_nImg * Dim), b + "attn.to_out.weight", _attnImg.View, _nImg, Dim, Dim);
        _k.GateResidual(_img.View, _attnImg.View, _modImg.View, 2 * Dim, _nImg * Dim, Dim);
        _k.GateResidual(_txt.View, _attnTxt.View, _modTxt.View, 2 * Dim, _nTxt * Dim, Dim);
        // MLPs
        _k.LayerNormModulate(_img.View, _normI.View, _modImg.View, 3 * Dim, 4 * Dim, _nImg, Dim, Eps);
        Lin(_normI.View, b + "ff.linear_in.weight", _gateUp.View, _nImg, Dim, 2 * MlpHidden);
        _u.Swiglu(_gateUp.View, _act.View, _nImg, MlpHidden);
        Lin(_act.View, b + "ff.linear_out.weight", _attnImg.View, _nImg, MlpHidden, Dim);
        _k.GateResidual(_img.View, _attnImg.View, _modImg.View, 5 * Dim, _nImg * Dim, Dim);
        _k.LayerNormModulate(_txt.View, _normT.View, _modTxt.View, 3 * Dim, 4 * Dim, _nTxt, Dim, Eps);
        Lin(_normT.View, b + "ff_context.linear_in.weight", _gateUp.View, _nTxt, Dim, 2 * MlpHidden);
        _u.Swiglu(_gateUp.View, _act.View, _nTxt, MlpHidden);
        Lin(_act.View, b + "ff_context.linear_out.weight", _attnTxt.View, _nTxt, MlpHidden, Dim);
        _k.GateResidual(_txt.View, _attnTxt.View, _modTxt.View, 5 * Dim, _nTxt * Dim, Dim);
    }

    private void SingleBlock(int l)
    {
        string b = $"single_transformer_blocks.{l}.";
        int cols = 3 * Dim + 2 * MlpHidden;                                           // q | k | v | gate | up
        var h = _proj.View;
        _k.LayerNormModulate(h, _attn.View, _modSingle.View, 0, Dim, _n, Dim, Eps);   // shift, scale, gate
        // fused projection [N, 27648] into _single (viewed N x cols): needs N * cols floats
        Lin(_attn.View, b + "attn.to_qkv_mlp_proj.weight", _single.View, _n, Dim, cols);
        _u.RmsNormHeads(_single.View, _w.Vector(b + "attn.norm_q.weight"), _n, cols, 0, Heads, HeadDim, Eps);
        _u.RmsNormHeads(_single.View, _w.Vector(b + "attn.norm_k.weight"), _n, cols, Dim, Heads, HeadDim, Eps);
        _f.RopeInterleaved(_single.View, _cosT.View, _sinT.View, _n, cols, 0, Heads, HeadDim);
        _f.RopeInterleaved(_single.View, _cosT.View, _sinT.View, _n, cols, Dim, Heads, HeadDim);
        _k.Attention(_single.View, cols, 0, _single.View, cols, Dim, 2 * Dim, _scores.View, _attn.View, _n, _n, Heads, HeadDim);
        // [attention | swiglu(mlp)] -> to_out
        _f.SwigluCols(_single.View, cols, 3 * Dim, _gateUp.View, Dim + MlpHidden, Dim, _n, MlpHidden);
        _f.CopyCols(_attn.View, _gateUp.View, _n, Dim, Dim, Dim + MlpHidden, 0);
        Lin(_gateUp.View, b + "attn.to_out.weight", _attn.View, _n, Dim + MlpHidden, Dim);
        _k.GateResidual(h, _attn.View, _modSingle.View, 2 * Dim, _n * Dim, Dim);
    }

    public void Dispose()
    {
        foreach (var o in _owned) o.Dispose();
        _owned.Clear();
    }
}
