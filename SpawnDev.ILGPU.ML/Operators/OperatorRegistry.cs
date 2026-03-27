using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;

namespace SpawnDev.ILGPU.ML.Operators;

/// <summary>
/// Registry of ONNX operator implementations.
/// Automatically registers all built-in operators on construction.
/// Custom operators can be added via Register().
/// </summary>
public class OperatorRegistry
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
        Register(new EqualOperator(this));
        Register(new GreaterOperator(this));
        Register(new LessOperator(this));
        Register(new LessOrEqualOperator(this));
        Register(new AndOperator(this));
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

        // Fused operators (created by GraphOptimizer)
        Register(new FusedLinearOperator(this));
        Register(new FusedScaledMatMulOperator(this));
    }
}
