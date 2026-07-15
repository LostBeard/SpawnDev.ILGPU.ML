using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;

namespace SpawnDev.ILGPU.ML.Operators;

/// <summary>
/// Registry of ONNX operator implementations.
/// Automatically registers all built-in operators on construction.
/// Custom operators can be added via Register().
/// </summary>
public class OperatorRegistry : IDisposable
{
    private readonly Dictionary<string, IOnnxOperator> _ops = new(StringComparer.OrdinalIgnoreCase);
    private readonly Accelerator _accelerator;
    public Accelerator Accelerator => _accelerator;

    /// <summary>
    /// Canonical, accelerator-free manifest of every ONNX op-type registered by
    /// <see cref="RegisterBuiltins"/>. Consumers that must answer "do we support this op?"
    /// WITHOUT constructing a full registry (e.g. the streaming Model Inspector, which is
    /// structure-only and never touches an accelerator) read this set instead.
    ///
    /// SINGLE SOURCE OF TRUTH: this list and the live registration calls are locked together
    /// by <c>MLTestBase.Op_BuiltinOpTypes_MatchesLiveRegistry</c>, which constructs a real
    /// registry and asserts <see cref="SupportedOps"/> set-equals this manifest. Add or remove
    /// a <c>Register(...)</c> call and forget to mirror it here (or vice versa) and that test
    /// fails. Do NOT maintain a second, divergent op list anywhere — point it at this one.
    /// (Stale parallel lists are exactly what made GPT-2 falsely report 90% compatibility.)
    /// </summary>
    public static readonly IReadOnlySet<string> BuiltinOpTypes = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
    {
        "Abs", "Acos", "Acosh", "Add", "AddRMSNorm", "AffineGrid", "And", "ArgMax", "ArgMin",
        "Asin", "Asinh", "Atan", "Atanh", "AveragePool", "BatchNormalization", "Bernoulli", "BitShift",
        "BitwiseAnd", "BitwiseNot", "BitwiseOr", "BitwiseXor", "BlackmanWindow", "Cast", "CastLike", "Ceil",
        "Celu", "CenterCropPad", "Clip", "Col2Im", "Compress", "Concat", "ConcatFromSequence", "Constant",
        "ConstantOfShape", "Conv", "ConvInteger", "ConvTranspose", "Cos", "Cosh", "CumSum", "DFT",
        "DeformConv", "DepthToSpace", "DequantizeLinear", "Det", "Div", "Dropout", "DynamicQuantizeLinear", "Einsum",
        "Elu", "Equal", "Erf", "Exp", "Expand", "EyeLike", "Flatten", "Floor",
        "FusedAttention", "FusedLinear", "FusedScaledMatMul", "GRU", "Gather", "GatherElements", "GatherND", "Gelu", "Gemm",
        "GlobalAveragePool", "GlobalLpPool", "GlobalMaxPool", "Greater", "GreaterOrEqual", "GridSample", "GroupNormalization", "HammingWindow",
        "HannWindow", "HardSigmoid", "HardSwish", "Hardmax", "Identity", "If", "ImageDecoder", "InstanceNormalization",
        "IsInf", "IsNaN", "LRN", "LSTM", "LayerNormalization", "LeakyRelu", "Less", "LessOrEqual",
        "Log", "LogSoftmax", "Loop", "LpNormalization", "LpPool", "MatMul", "MatMulInteger", "Max",
        "MaxPool", "MaxRoiPool", "MaxUnpool", "Mean", "MeanVarianceNormalization", "MelWeightMatrix", "Min", "Mish",
        "MoE", "Mod", "Mul", "Multinomial", "Neg", "NegativeLogLikelihoodLoss", "NonMaxSuppression", "NonZero", "Not",
        "OneHot", "Optional", "OptionalGetElement", "OptionalHasElement", "Or", "PRelu", "Pad", "Pow",
        "QLinearConv", "QLinearMatMul", "QuantizeLinear", "RNN", "RandomNormal", "RandomNormalLike", "RandomUniform", "RandomUniformLike",
        "Range", "Reciprocal", "ReduceL1", "ReduceL2", "ReduceLogSum", "ReduceLogSumExp", "ReduceMax", "ReduceMean",
        "RoPE", "ShortConv", "GatedDeltaNet",
        "ReduceMin", "ReduceProd", "ReduceSum", "ReduceSumSquare", "Relu", "Reshape", "Resize", "ReverseSequence",
        "RMSNormalization",
        "RoiAlign", "Round", "STFT", "Scan", "Scatter", "ScatterElements", "ScatterND", "Selu",
        "SequenceAt", "SequenceConstruct", "SequenceEmpty", "SequenceErase", "SequenceInsert", "SequenceLength", "SequenceMap", "Shape",
        "Shrink", "SiLU", "Sigmoid", "Sign", "Sin", "Sinh", "Size", "Slice",
        "Softmax", "SoftmaxCrossEntropyLoss", "Softplus", "Softsign", "SpaceToDepth", "Split", "SplitToSequence", "Sqrt", "SwiGLU",
        "Squeeze", "StringConcat", "StringNormalizer", "StringSplit", "Sub", "Sum", "Tan", "Tanh",
        "ThresholdedRelu", "Tile", "TopK", "Transpose", "Trilu", "Unique", "Unsqueeze", "Upsample",
        "Where", "Xor",
    };

    // Kernel instances (shared across operators)
    public MatMulKernel MatMul { get; }
    public LayerNormKernel LayerNorm { get; }
    public SoftmaxKernel Softmax { get; }
    public ElementWiseKernels ElementWise { get; }
    public Conv2DKernel Conv2D { get; }
    public Conv1DKernel Conv1D { get; }
    public ShortConvKernel ShortConv { get; }
    public GatedDeltaNetKernel GatedDeltaNetScan { get; }
    public GatedDeltaNetOps GatedDeltaNetOps { get; }
    public ActivationKernels Activations { get; }
    public ReductionKernels Reductions { get; }
    public PoolingKernels Pooling { get; }
    public NormalizationKernels Normalization { get; }
    public TransposeKernel Transpose { get; }
    public GatherKernel Gather { get; }
    public PadKernel Pad { get; }
    public ConvTranspose2DKernel ConvTranspose { get; }
    public Kernels.FusedDequantMatMul FusedDequant { get; }
    public Kernels.FusedDequantGather FusedDequantGather { get; }
    public Kernels.MoEKernels MoE { get; }
    public Kernels.RoPEKernel RoPE { get; }
    public Kernels.FusedAttentionKernel FusedAttention { get; }
    public Kernels.SliceKernel Slice { get; }
    public Kernels.ConcatKernel Concat { get; }
    public Kernels.MissingElementWiseKernels MissingElementWise { get; }
    /// <summary>Approach-(i) precision-aware op kernels (read+write low-p activations, no fp32 temp).
    /// Used by <see cref="IPrecisionAwareOperator"/> implementations under the F16 executor path.</summary>
    public Kernels.PrecisionAwareKernels PrecisionAware { get; }

    /// <summary>
    /// GGML quantization type per quantized-weight tensor name, set by the GGUF loader
    /// alongside <c>OnnxOpContext.QuantizedWeights</c> (which carries only the raw byte
    /// views). Operators that route a tensor to a fused dequant kernel MUST resolve its
    /// type here - a quantized view without a type is an error, never "assume Q4_0"
    /// (decoding one GGML layout as another produces silent garbage; see the K-quant
    /// landmine, DevComms seven P1 thread 2026-06-11). Session-scoped: lives on the
    /// registry because the registry already flows to every operator, keeping
    /// GraphExecutor's plumbing untouched.
    /// </summary>
    public IReadOnlyDictionary<string, GGUF.GGMLType>? QuantizedWeightTypes { get; set; }

    public OperatorRegistry(Accelerator accelerator)
    {
        _accelerator = accelerator;

        // Create kernel instances
        MatMul = new MatMulKernel(accelerator);
        LayerNorm = new LayerNormKernel(accelerator);
        Softmax = new SoftmaxKernel(accelerator);
        ElementWise = new ElementWiseKernels(accelerator);
        Conv2D = new Conv2DKernel(accelerator);
        Conv1D = new Conv1DKernel(accelerator);
        ShortConv = new ShortConvKernel(accelerator);
        GatedDeltaNetScan = new GatedDeltaNetKernel(accelerator);
        GatedDeltaNetOps = new GatedDeltaNetOps(accelerator);
        Activations = new ActivationKernels(accelerator);
        Reductions = new ReductionKernels(accelerator);
        Pooling = new PoolingKernels(accelerator);
        Normalization = new NormalizationKernels(accelerator);
        Transpose = new TransposeKernel(accelerator);
        Gather = new GatherKernel(accelerator);
        Pad = new PadKernel(accelerator);
        ConvTranspose = new ConvTranspose2DKernel(accelerator);
        FusedDequant = new Kernels.FusedDequantMatMul(accelerator);
        FusedDequantGather = new Kernels.FusedDequantGather(accelerator);
        MoE = new Kernels.MoEKernels(accelerator);
        RoPE = new Kernels.RoPEKernel(accelerator);
        FusedAttention = new Kernels.FusedAttentionKernel(accelerator);
        Slice = new Kernels.SliceKernel(accelerator);
        Concat = new Kernels.ConcatKernel(accelerator);
        MissingElementWise = new Kernels.MissingElementWiseKernels(accelerator);
        PrecisionAware = new Kernels.PrecisionAwareKernels(accelerator);

        // Register built-in operators
        RegisterBuiltins();
    }

    // Zero-bias cache for Conv ops without an explicit bias input.
    // Keyed by outC so any number of biasless convs with the same channel count
    // share a single GPU zero buffer. Lifetime is bound to the registry — buffers
    // are disposed in Dispose(). This avoids leaking one permanent buffer per
    // biasless-conv invocation through a multi-inference session.
    private readonly Dictionary<int, MemoryBuffer1D<float, Stride1D.Dense>> _zeroBiasCache = new();

    public ArrayView1D<float, Stride1D.Dense> GetOrCreateZeroBias(int outC)
    {
        if (!_zeroBiasCache.TryGetValue(outC, out var buf))
        {
            buf = _accelerator.Allocate1D(new float[outC]);
            _zeroBiasCache[outC] = buf;
        }
        return buf.View;
    }

    public void Register(IOnnxOperator op) => _ops[op.OpType] = op;

    public IOnnxOperator Resolve(string opType)
        => _ops.TryGetValue(opType, out var op) ? op
           : throw new NotSupportedException($"Unsupported ONNX operator: {opType}");

    public bool IsSupported(string opType) => _ops.ContainsKey(opType);

    public IReadOnlyList<string> SupportedOps => _ops.Keys.ToList();

    private void RegisterBuiltins()
    {
        // Tier 1: Essential ops
        Register(new MatMulOperator(this));
        Register(new ReluOperator(this));
        Register(new GeluOperator(this));
        Register(new AddOperator(this));
        Register(new MulOperator(this));
        Register(new SubOperator(this));
        Register(new ReshapeOperator(this));
        Register(new TransposeOperator(this));
        Register(new SoftmaxOperator(this));
        Register(new LayerNormOperator(this));
        Register(new UnsqueezeOperator(this));
        Register(new SqueezeOperator(this));
        Register(new FlattenOperator(this));
        Register(new ConcatOperator(this));
        Register(new GatherOperator(this));
        Register(new ScatterNDOperator(this));
        Register(new ClipOperator(this));

        // Tier 2: Common ops
        Register(new SigmoidOperator(this));
        Register(new TanhOperator(this));
        Register(new BatchNormOperator(this));
        Register(new GlobalAvgPoolOperator(this));
        Register(new ReduceMeanOperator(this));
        Register(new ReduceSumOperator(this));
        Register(new SqrtOperator(this));
        Register(new SinOperator(this));
        Register(new CosOperator(this));
        Register(new TanOperator(this));
        Register(new ExpOperator(this));
        Register(new NegOperator(this));
        Register(new DivOperator(this));
        Register(new AbsOperator(this));
        Register(new ErfOperator(this));
        Register(new PowOperator(this));
        Register(new WhereOperator(this));
        Register(new ReciprocalOperator(this));
        Register(new MaxPoolOperator(this));
        Register(new AveragePoolOperator(this));
        Register(new ResizeOperator(this));
        Register(new PadOperator(this));
        Register(new ConvTransposeOperator(this));
        Register(new ArgMaxOperator(this));
        Register(new GatherNDOperator(this));
        Register(new ConvOperator(this));
        Register(new SplitOperator(this));
        Register(new SliceOperator(this));
        Register(new DropoutOperator(this));
        Register(new GemmOperator(this));
        Register(new MoEOperator(this));
        Register(new InstanceNormOperator(this));
        Register(new GroupNormOperator(_accelerator));
        Register(new ConstantOperator());
        Register(new CeilOperator(this));
        Register(new LogOperator(this));
        Register(new MinOperator(this));
        Register(new MaxOnnxOperator(this));
        Register(new ReduceMaxOperator(this));
        Register(new ReduceMinOperator(this));
        Register(new CastOperator(this));
        Register(new FloorOperator(this));
        Register(new UpsampleOperator(this));
        Register(new ShapeOperator(this));
        Register(new SiLUOperator(this));
        Register(new SwiGLUOperator(this));
        Register(new LeakyReluOperator(this));
        Register(new ExpandOperator(this));
        Register(new TileOperator(this));
        Register(new GatherElementsOperator(this));
        Register(new ModOperator(this));
        Register(new CumSumOperator(this));
        Register(new OneHotOperator(this));
        Register(new EqualOperator(this));
        Register(new GreaterOperator(this));
        Register(new LessOperator(this));
        Register(new LessOrEqualOperator(this));
        Register(new GreaterOrEqualOperator(this));
        Register(new AndOperator(this));
        Register(new OrOperator(this));
        Register(new XorOperator(this));
        Register(new IsNaNOperator(this));
        Register(new NotOperator(this));
        Register(new ConstantOfShapeOperator(this));
        Register(new RangeOperator(this));
        Register(new HardSigmoidOperator(this));
        Register(new HardSwishOperator(this));

        Register(new NonZeroOperator(this));

        // Operators from #2
        Register(new DepthToSpaceOperator(_accelerator));
        Register(new TopKOperator(_accelerator));
        Register(new SignOperator(_accelerator));

        // General tensor operations
        Register(new EinsumOperator(this));

        // Trig / hyperbolic
        Register(new AcosOperator(this));
        Register(new AcoshOperator(this));
        Register(new AsinOperator(this));
        Register(new AsinhOperator(this));
        Register(new AtanOperator(this));
        Register(new AtanhOperator(this));
        Register(new CoshOperator(this));
        Register(new SinhOperator(this));

        // Activations
        Register(new EluOperator(this));
        Register(new CeluOperator(this));
        Register(new SeluOperator(this));
        Register(new SoftplusOperator(this));
        Register(new SoftsignOperator(this));
        Register(new MishOperator(this));
        Register(new ThresholdedReluOperator(this));
        Register(new PReluOperator(this));
        Register(new LogSoftmaxOperator(this));
        Register(new HardmaxOperator(this));

        // Math / utility
        Register(new RoundOperator(this));
        Register(new IsInfOperator(this));
        Register(new ShrinkOperator(this));
        Register(new IdentityOperator(this));
        Register(new SizeOperator(this));
        Register(new ArgMinOperator(this));
        Register(new SumOperator(this));
        Register(new MeanOperator(this));

        // Reduce variants
        Register(new ReduceProdOperator(this));
        Register(new ReduceL1Operator(this));
        Register(new ReduceL2Operator(this));
        Register(new ReduceSumSquareOperator(this));
        Register(new ReduceLogSumOperator(this));
        Register(new ReduceLogSumExpOperator(this));

        // Pooling
        Register(new GlobalMaxPoolOperator(this));

        // Spatial
        Register(new SpaceToDepthOperator(this));
        Register(new TriluOperator(this));
        Register(new ScatterElementsOperator(this));
        Register(new NonMaxSuppressionOperator(this));

        // Bitwise
        Register(new BitwiseAndOperator(this));
        Register(new BitwiseOrOperator(this));
        Register(new BitwiseXorOperator(this));
        Register(new BitwiseNotOperator(this));
        Register(new BitShiftOperator(this));

        // Quantization
        Register(new DequantizeLinearOperator(this));
        Register(new QuantizeLinearOperator(this));
        Register(new DynamicQuantizeLinearOperator(this));

        // Utility
        Register(new CastLikeOperator(this));
        Register(new CompressOperator(this));
        Register(new EyeLikeOperator(this));
        Register(new LRNOperator(this));
        Register(new MeanVarianceNormalizationOperator(this));
        Register(new ReverseSequenceOperator(this));
        Register(new ScatterOperator(this));
        Register(new UniqueOperator(this));

        // Random
        Register(new RandomNormalOperator(this));
        Register(new RandomNormalLikeOperator(this));
        Register(new RandomUniformOperator(this));
        Register(new RandomUniformLikeOperator(this));
        Register(new MultinomialOperator(this));

        // Window functions
        Register(new HannWindowOperator(this));
        Register(new HammingWindowOperator(this));
        Register(new BlackmanWindowOperator(this));

        // Loss functions
        Register(new NegativeLogLikelihoodLossOperator(this));
        Register(new SoftmaxCrossEntropyLossOperator(this));

        // Remaining ONNX operators (batch 4 — full coverage)
        Register(new LpNormalizationOperator(this)); Register(new GlobalLpPoolOperator(this));
        Register(new LpPoolOperator(this)); Register(new DetOperator(this));
        Register(new BernoulliOperator(this)); Register(new CenterCropPadOperator(this));
        Register(new MaxRoiPoolOperator(this)); Register(new MaxUnpoolOperator(this));
        Register(new ImageDecoderOperator(this)); Register(new AffineGridOperator(this));
        Register(new GridSampleOperator(this)); Register(new Col2ImOperator(this));
        Register(new DeformConvOperator(this)); Register(new RoiAlignOperator(this));
        Register(new ConvIntegerOperator(this)); Register(new MatMulIntegerOperator(this));
        Register(new QLinearConvOperator(this)); Register(new QLinearMatMulOperator(this));
        Register(new DFTOperatorImpl(this)); Register(new STFTOperatorImpl(this));
        Register(new MelWeightMatrixOperatorImpl(this));
        Register(new SequenceConstructOperator(this)); Register(new SequenceEmptyOperator(this));
        Register(new SequenceAtOperator(this)); Register(new SequenceInsertOperator(this));
        Register(new SequenceEraseOperator(this)); Register(new SequenceLengthOperator(this));
        Register(new SequenceMapOperator(this)); Register(new ConcatFromSequenceOperator(this));
        Register(new SplitToSequenceOperator(this));
        Register(new OptionalOperator(this)); Register(new OptionalGetElementOperator(this));
        Register(new OptionalHasElementOperator(this));
        Register(new StringConcatOperator(this)); Register(new StringNormalizerOperator(this));
        Register(new StringSplitOperator(this));
        Register(new IfOperator(this)); Register(new LoopOperator(this)); Register(new ScanOperator(this));
        Register(new RNNOperatorImpl(this)); Register(new LSTMOperatorImpl(this)); Register(new GRUOperatorImpl(this));

        // Fused operators (created by GraphOptimizer)
        Register(new FusedLinearOperator(this));
        Register(new FusedScaledMatMulOperator(this));
        // gemma4 fused-attention layer (graph builder emits per-layer nodes)
        Register(new RoPEOperator(this));
        Register(new FusedAttentionOperator(this));
        // True RMSNorm (every RMS decoder: llama/mistral/qwen/gemma). Distinct from the
        // mean-centered LayerNormalization — see RMSNormOperator.
        Register(new RMSNormOperator(this));
        Register(new ShortConvOperator(this));
        Register(new GatedDeltaNetOperator(this));
        Register(new AddRMSNormOperator(this));
    }

    public void Dispose()
    {
        // Dispose operator instances that hold GPU param buffers.
        foreach (var op in _ops.Values)
            if (op is IDisposable d) try { d.Dispose(); } catch { }
        // Dispose kernel instances explicitly. Do NOT use reflection —
        // it catches the Accelerator property (IDisposable) which we don't own.
        // Disposing the shared Accelerator destroys the GPU device and causes
        // "obj null or undefined" on WebGPU for all subsequent operations.
        try { (MatMul as IDisposable)?.Dispose(); } catch { }
        try { (LayerNorm as IDisposable)?.Dispose(); } catch { }
        try { (Softmax as IDisposable)?.Dispose(); } catch { }
        try { (ElementWise as IDisposable)?.Dispose(); } catch { }
        try { (Conv2D as IDisposable)?.Dispose(); } catch { }
        try { (Conv1D as IDisposable)?.Dispose(); } catch { }
        try { (ShortConv as IDisposable)?.Dispose(); } catch { }
        try { (GatedDeltaNetScan as IDisposable)?.Dispose(); } catch { }
        try { (GatedDeltaNetOps as IDisposable)?.Dispose(); } catch { }
        try { (Activations as IDisposable)?.Dispose(); } catch { }
        try { (Reductions as IDisposable)?.Dispose(); } catch { }
        try { (Pooling as IDisposable)?.Dispose(); } catch { }
        try { (Normalization as IDisposable)?.Dispose(); } catch { }
        try { (Transpose as IDisposable)?.Dispose(); } catch { }
        try { (Gather as IDisposable)?.Dispose(); } catch { }
        try { (Pad as IDisposable)?.Dispose(); } catch { }
        try { (ConvTranspose as IDisposable)?.Dispose(); } catch { }
        try { (FusedDequant as IDisposable)?.Dispose(); } catch { }
        try { (FusedDequantGather as IDisposable)?.Dispose(); } catch { }
        try { (RoPE as IDisposable)?.Dispose(); } catch { }
        try { (FusedAttention as IDisposable)?.Dispose(); } catch { }
        try { (Slice as IDisposable)?.Dispose(); } catch { }
        try { (Concat as IDisposable)?.Dispose(); } catch { }
        try { (MissingElementWise as IDisposable)?.Dispose(); } catch { }
        try { (PrecisionAware as IDisposable)?.Dispose(); } catch { }

        // Dispose cached zero-bias buffers (one per distinct outC seen).
        foreach (var b in _zeroBiasCache.Values) try { b.Dispose(); } catch { }
        _zeroBiasCache.Clear();
    }
}
