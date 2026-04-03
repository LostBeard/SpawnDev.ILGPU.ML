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

    // Kernel instances (shared across operators)
    public MatMulKernel MatMul { get; }
    public LayerNormKernel LayerNorm { get; }
    public SoftmaxKernel Softmax { get; }
    public ElementWiseKernels ElementWise { get; }
    public Conv2DKernel Conv2D { get; }
    public Conv1DKernel Conv1D { get; }
    public ActivationKernels Activations { get; }
    public ReductionKernels Reductions { get; }
    public PoolingKernels Pooling { get; }
    public NormalizationKernels Normalization { get; }
    public TransposeKernel Transpose { get; }
    public GatherKernel Gather { get; }
    public PadKernel Pad { get; }
    public ConvTranspose2DKernel ConvTranspose { get; }
    public Kernels.FusedDequantMatMul FusedDequant { get; }

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
        Activations = new ActivationKernels(accelerator);
        Reductions = new ReductionKernels(accelerator);
        Pooling = new PoolingKernels(accelerator);
        Normalization = new NormalizationKernels(accelerator);
        Transpose = new TransposeKernel(accelerator);
        Gather = new GatherKernel(accelerator);
        Pad = new PadKernel(accelerator);
        ConvTranspose = new ConvTranspose2DKernel(accelerator);
        FusedDequant = new Kernels.FusedDequantMatMul(accelerator);

        // Register built-in operators
        RegisterBuiltins();
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
        try { (Activations as IDisposable)?.Dispose(); } catch { }
        try { (Reductions as IDisposable)?.Dispose(); } catch { }
        try { (Pooling as IDisposable)?.Dispose(); } catch { }
        try { (Normalization as IDisposable)?.Dispose(); } catch { }
        try { (Transpose as IDisposable)?.Dispose(); } catch { }
        try { (Gather as IDisposable)?.Dispose(); } catch { }
        try { (Pad as IDisposable)?.Dispose(); } catch { }
        try { (ConvTranspose as IDisposable)?.Dispose(); } catch { }
        try { (FusedDequant as IDisposable)?.Dispose(); } catch { }
    }
}
