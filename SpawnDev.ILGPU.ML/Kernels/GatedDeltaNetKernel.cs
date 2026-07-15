using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// Qwen3-Next "Gated DeltaNet" linear-attention recurrence (the 24 linear layers of GGUF arch qwen35).
///
/// Per value-head hv (num_v_heads, e.g. 32) with a recurrent state S[hv] of shape [head_k_dim, head_v_dim]
/// (e.g. [128,128]), for each token t in order (delta rule, from HF modeling_qwen3_next.py verbatim):
///     S      = S * exp(g_t[hv])                        # per-head scalar log-decay g (g&lt;=0)
///     kv[v]  = sum_k S[k,v] * k_t[kh][k]               # kh = hv % num_k_heads (ggml_repeat tile)
///     delta  = (v_t[hv][v] - kv[v]) * beta_t[hv]
///     S[k,v] = S[k,v] + k_t[kh][k] * delta[v]          # outer-product update
///     out_t[hv][v] = sum_k S[k,v] * (q_t[kh][k] / sqrt(head_k_dim))   # HF scales q by 1/sqrt(head_k_dim)
/// q and k are L2-normalized per head BEFORE this (done by the caller/op); the 1/sqrt(head_k_dim) query
/// scale is applied HERE. The output then goes through a gated RMSNorm (·silu(z)) and the out projection
/// (in the operator, not here). Verified against an independent numpy reference (GGUF qwen35): " Paris"
/// is rank-0 for "The capital of France is" with kh=hv%num_k_heads + the q-scale (both were bugs before).
///
/// Threading: one thread per (hv, v) = num_v_heads*head_v_dim threads; each owns the state COLUMN S[hv][:][v]
/// (head_k_dim values) and loops tokens sequentially (the recurrence is inherently causal/sequential over t,
/// which is exactly why linear-attention decode is O(1)/token). State layout is [hv][v][k] (k innermost →
/// each thread's head_k_dim values are contiguous). The state buffer PERSISTS across the call: pass a zeroed
/// state for a fresh prefill (it emerges holding the final state = the decode-start state); pass the carried
/// state for a 1-token decode step. This is the recurrent analogue of ShortConvStateCache.
///
/// NOTE: this first version reads/writes the state in global memory (2 passes over head_k_dim per token per
/// thread). Correctness first; a shared/register-blocked tiling is the perf follow-up (Rule 4).
/// </summary>
public sealed class GatedDeltaNetKernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,   // q     [seq, numKHeads, headKDim]  (L2-normed)
        ArrayView1D<float, Stride1D.Dense>,   // k     [seq, numKHeads, headKDim]  (L2-normed)
        ArrayView1D<float, Stride1D.Dense>,   // v     [seq, numVHeads, headVDim]
        ArrayView1D<float, Stride1D.Dense>,   // a     [seq, numVHeads]  (ssm_alpha proj)
        ArrayView1D<float, Stride1D.Dense>,   // b     [seq, numVHeads]  (ssm_beta proj)
        ArrayView1D<float, Stride1D.Dense>,   // ssmA  [numVHeads]  (A_log)
        ArrayView1D<float, Stride1D.Dense>,   // ssmDt [numVHeads]  (dt_bias)
        ArrayView1D<float, Stride1D.Dense>,   // outp  [seq, numVHeads, headVDim]
        ArrayView1D<float, Stride1D.Dense>,   // state [numVHeads, headVDim, headKDim]  (persists)
        ArrayView1D<int, Stride1D.Dense>>? _kernel;   // params [seq, numKHeads, headKDim, numVHeads, headVDim]

    private MemoryBuffer1D<int, Stride1D.Dense>? _paramsBuf;
    private readonly List<MemoryBuffer1D<int, Stride1D.Dense>> _oldParamsBufs = new();

    public GatedDeltaNetKernel(Accelerator accelerator) => _accelerator = accelerator;

    /// <summary>Run the delta-rule scan. q/k must be L2-normed per head already. The per-head gates are folded
    /// in: g = -exp(ssmA[hv]) * softplus(a[t,hv] + ssmDt[hv]); beta = sigmoid(b[t,hv]). <paramref name="state"/>
    /// is the persistent recurrent state [numVHeads, headVDim, headKDim] — zeroed for a fresh sequence, carried
    /// across decode steps. Produces <paramref name="outp"/> [seq, numVHeads, headVDim].</summary>
    public void Forward(
        ArrayView1D<float, Stride1D.Dense> q,
        ArrayView1D<float, Stride1D.Dense> k,
        ArrayView1D<float, Stride1D.Dense> v,
        ArrayView1D<float, Stride1D.Dense> a,
        ArrayView1D<float, Stride1D.Dense> b,
        ArrayView1D<float, Stride1D.Dense> ssmA,
        ArrayView1D<float, Stride1D.Dense> ssmDt,
        ArrayView1D<float, Stride1D.Dense> outp,
        ArrayView1D<float, Stride1D.Dense> state,
        int seq, int numKHeads, int headKDim, int numVHeads, int headVDim)
    {
        _kernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>(ScanImpl);

        var p = new[] { seq, numKHeads, headKDim, numVHeads, headVDim };
        if (_paramsBuf != null) _oldParamsBufs.Add(_paramsBuf);
        _paramsBuf = _accelerator.Allocate1D(p);

        _kernel(numVHeads * headVDim, q, k, v, a, b, ssmA, ssmDt, outp, state, _paramsBuf.View);
    }

    private static void ScanImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> q,
        ArrayView1D<float, Stride1D.Dense> k,
        ArrayView1D<float, Stride1D.Dense> v,
        ArrayView1D<float, Stride1D.Dense> a,
        ArrayView1D<float, Stride1D.Dense> b,
        ArrayView1D<float, Stride1D.Dense> ssmA,
        ArrayView1D<float, Stride1D.Dense> ssmDt,
        ArrayView1D<float, Stride1D.Dense> outp,
        ArrayView1D<float, Stride1D.Dense> state,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int seq = p[0], numKHeads = p[1], headKDim = p[2], numVHeads = p[3], headVDim = p[4];

        int hv = idx / headVDim;        // value head
        int vd = idx % headVDim;        // value-dim within the head
        // k-head for this v-head: llama.cpp expands 16 k-heads -> 32 v-heads via ggml_repeat_4d
        // (TILE, not repeat_interleave) => kh = hv % numKHeads. Corroborated by GGUF
        // ssm.v_head_reordered=True. numpy ref verified: " Paris" rank 0 for "The capital of France is".
        int kh = hv % numKHeads;
        // HF torch_recurrent_gated_delta_rule scales the query by 1/sqrt(head_k_dim) before the scan.
        float qScale = 1f / MathF.Sqrt(headKDim);

        // This thread owns the state column S[hv][vd][0..headKDim-1] (k innermost, contiguous).
        int sBase = (hv * headVDim + vd) * headKDim;
        int qkHeadStride = numKHeads * headKDim;   // per-token stride of q/k
        int vHeadStride = numVHeads * headVDim;     // per-token stride of v/out

        // Per-head decay/beta constants that don't vary within the head. dt_bias is per v-head.
        // ssm_a is used DIRECTLY: the GGUF tensor is already -exp(A_log), pre-transformed at conversion
        // (llama.cpp ggml_mul(softplus, ssm_a); numpy g = softplus*ssm_a — both multiply the tensor as-is).
        // Applying -exp() here again was a bug (double transform) — caught by bit-verify vs the numpy ref.
        float aCoeff = ssmA[hv];
        float dtBias = ssmDt[hv];

        for (int t = 0; t < seq; t++)
        {
            // g = -exp(A_log) * softplus(a + dt_bias);  softplus(x)=log(1+exp(x)) (x>20 → x, avoid overflow).
            float ax = a[t * numVHeads + hv] + dtBias;
            float sp = ax > 20f ? ax : MathF.Log(1f + MathF.Exp(ax));
            float expG = MathF.Exp(aCoeff * sp);            // exp(g), g<=0 → decay in (0,1]
            float betaT = 1f / (1f + MathF.Exp(-b[t * numVHeads + hv]));   // sigmoid(b)
            int qkTok = t * qkHeadStride + kh * headKDim;   // base of k_t[kh] and q_t[kh]
            int vTok = t * vHeadStride + hv * headVDim;      // base of v_t[hv] and out_t[hv]

            // Pass 1: decay S, accumulate kv = sum_k S[k,vd] * k_t[k].
            float kv = 0f;
            for (int kk = 0; kk < headKDim; kk++)
            {
                float s = state[sBase + kk] * expG;
                state[sBase + kk] = s;
                kv += s * k[qkTok + kk];
            }
            float delta = (v[vTok + vd] - kv) * betaT;

            // Pass 2: S += k_t ⊗ delta, accumulate out = sum_k S[k,vd] * q_t[k].
            float outAcc = 0f;
            for (int kk = 0; kk < headKDim; kk++)
            {
                float s = state[sBase + kk] + k[qkTok + kk] * delta;
                state[sBase + kk] = s;
                outAcc += s * (q[qkTok + kk] * qScale);   // q pre-scaled by 1/sqrt(head_k_dim)
            }
            outp[vTok + vd] = outAcc;
        }
    }

    public void Dispose()
    {
        _paramsBuf?.Dispose(); _paramsBuf = null;
        foreach (var b in _oldParamsBufs) b.Dispose();
        _oldParamsBufs.Clear();
    }
}
