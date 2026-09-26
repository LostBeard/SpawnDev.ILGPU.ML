using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Training;

/// <summary>
/// A simple trainable multi-layer perceptron (MLP) built programmatically.
/// Supports forward pass, backward pass, and weight updates — all on GPU.
/// </summary>
/// <remarks>
/// Scratch buffers are allocated once in <see cref="Build"/> and reused. Prefer
/// <c>readLoss: false</c> on hot training loops, then read loss once per epoch —
/// per-step <c>CopyToHostAsync</c> is the dominant cost on WebGPU/Wasm.
/// </remarks>
public class TrainableModel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly TrainingKernels _kernels;
    private readonly ElementWiseKernels _elementWise;
    private readonly SoftmaxKernel _softmax;
    private readonly List<Layer> _layers = new();
    private bool _built;
    private int _maxBatch;

    // Persistent train/predict scratch — never allocated per step.
    private MemoryBuffer1D<float, Stride1D.Dense>? _inputBuf;
    private MemoryBuffer1D<int, Stride1D.Dense>? _targetBuf;
    private MemoryBuffer1D<float, Stride1D.Dense>? _probsBuf;
    private MemoryBuffer1D<float, Stride1D.Dense>? _lossBuf;
    private MemoryBuffer1D<float, Stride1D.Dense>? _gradLogitsBuf;

    public int InputSize { get; private set; }
    public int OutputSize { get; private set; }
    public int MaxBatchSize => _maxBatch;
    public int ParameterCount => _layers.Sum(l => l.ParameterCount);

    public TrainableModel(Accelerator accelerator)
    {
        _accelerator = accelerator;
        _kernels = new TrainingKernels(accelerator);
        _elementWise = new ElementWiseKernels(accelerator);
        _softmax = new SoftmaxKernel(accelerator);
    }

    public TrainableModel AddLinear(int inFeatures, int outFeatures)
    {
        _layers.Add(new LinearLayer(inFeatures, outFeatures));
        if (_layers.Count == 1) InputSize = inFeatures;
        OutputSize = outFeatures;
        return this;
    }

    public TrainableModel AddReLU()
    {
        _layers.Add(new ReLULayer());
        return this;
    }

    /// <summary>Allocate GPU buffers and initialize weights.</summary>
    public void Build(int maxBatchSize = 32)
    {
        if (_built) throw new InvalidOperationException("Already built.");
        if (maxBatchSize < 1) throw new ArgumentOutOfRangeException(nameof(maxBatchSize));
        _maxBatch = maxBatchSize;
        var rng = new Random(42);
        foreach (var layer in _layers)
            layer.Allocate(_accelerator, maxBatchSize, rng, _elementWise);

        _inputBuf = _accelerator.Allocate1D<float>(maxBatchSize * InputSize);
        _targetBuf = _accelerator.Allocate1D<int>(maxBatchSize);
        _probsBuf = _accelerator.Allocate1D<float>(maxBatchSize * OutputSize);
        _lossBuf = _accelerator.Allocate1D<float>(maxBatchSize);
        _gradLogitsBuf = _accelerator.Allocate1D<float>(maxBatchSize * OutputSize);
        _built = true;
    }

    /// <summary>
    /// One training step. When <paramref name="readLoss"/> is false, skips GPU sync and host
    /// readback (caller must eventually sync / read before relying on loss or disposing).
    /// </summary>
    public async Task<float> TrainStepAsync(
        float[] inputData, int[] targetLabels,
        int batchSize, float learningRate = 0.01f,
        bool readLoss = true)
    {
        if (!_built) throw new InvalidOperationException("Call Build() first.");
        if (batchSize < 1 || batchSize > _maxBatch)
            throw new ArgumentOutOfRangeException(nameof(batchSize), $"1..{_maxBatch}");
        if (inputData.Length != batchSize * InputSize)
            throw new ArgumentException($"Expected input length {batchSize * InputSize}, got {inputData.Length}.", nameof(inputData));
        if (targetLabels.Length != batchSize)
            throw new ArgumentException($"Expected {batchSize} labels, got {targetLabels.Length}.", nameof(targetLabels));

        UploadBatch(inputData, targetLabels, batchSize);
        TrainStepCore(batchSize, learningRate);

        if (!readLoss)
            return 0f;

        return await ReadLastLossAsync(batchSize).ConfigureAwait(false);
    }

    /// <summary>
    /// Sync fire-and-forget train step (no host loss). Prefer this in tight loops;
    /// call <see cref="ReadLastLossAsync"/> after the last step of an epoch.
    /// </summary>
    public void TrainStep(
        float[] inputData, int[] targetLabels,
        int batchSize, float learningRate)
    {
        if (!_built) throw new InvalidOperationException("Call Build() first.");
        if (batchSize < 1 || batchSize > _maxBatch)
            throw new ArgumentOutOfRangeException(nameof(batchSize));
        UploadBatch(inputData, targetLabels, batchSize);
        TrainStepCore(batchSize, learningRate);
    }

    private void UploadBatch(float[] inputData, int[] targetLabels, int batchSize)
    {
        int inElems = batchSize * InputSize;
        // SubView + CopyFromCPU requires T[] (IContiguousArrayView), not Span on SubView.
        // Callers must pass exact-length staging arrays (batch * InputSize).
        if (inputData.Length != inElems)
            throw new ArgumentException($"Expected input length {inElems}, got {inputData.Length}.", nameof(inputData));
        if (targetLabels.Length != batchSize)
            throw new ArgumentException($"Expected {batchSize} labels, got {targetLabels.Length}.", nameof(targetLabels));

        _inputBuf!.View.SubView(0, inElems).CopyFromCPU(inputData);
        _targetBuf!.View.SubView(0, batchSize).CopyFromCPU(targetLabels);
    }

    private void TrainStepCore(int batchSize, float learningRate)
    {
        int outputClasses = OutputSize;
        var current = _inputBuf!.View.SubView(0, batchSize * InputSize);
        foreach (var layer in _layers)
            current = layer.Forward(_kernels, _accelerator, current, batchSize);

        var probs = _probsBuf!.View.SubView(0, batchSize * outputClasses);
        var loss = _lossBuf!.View.SubView(0, batchSize);
        var targets = _targetBuf!.View.SubView(0, batchSize);
        _kernels.SoftmaxCrossEntropyForward(current, probs, loss, targets, batchSize, outputClasses);

        var gradLogits = _gradLogitsBuf!.View.SubView(0, batchSize * outputClasses);
        _kernels.SoftmaxCrossEntropyBackward(probs, targets, gradLogits, batchSize, outputClasses);

        var gradCurrent = gradLogits;
        for (int i = _layers.Count - 1; i >= 0; i--)
            gradCurrent = _layers[i].Backward(_kernels, _accelerator, gradCurrent, batchSize);

        foreach (var layer in _layers)
            layer.UpdateWeights(_kernels, learningRate);
    }

    /// <summary>Drain pending GPU work without host readback.</summary>
    public Task FlushAsync() => _accelerator.SynchronizeAsync();

    /// <summary>Read mean loss from the last train step that wrote the loss buffer (syncs first).</summary>
    public async Task<float> ReadLastLossAsync(int batchSize)
    {
        await _accelerator.SynchronizeAsync().ConfigureAwait(false);
        var lossData = await _lossBuf!.CopyToHostAsync<float>(0, batchSize).ConfigureAwait(false);
        float sum = 0f;
        for (int i = 0; i < batchSize; i++)
            sum += lossData[i];
        return sum / batchSize;
    }

    /// <summary>Run forward pass only (inference). Returns raw logits.</summary>
    public async Task<float[]> PredictAsync(float[] inputData, int batchSize)
    {
        if (!_built) throw new InvalidOperationException("Call Build() first.");
        if (batchSize < 1 || batchSize > _maxBatch)
            throw new ArgumentOutOfRangeException(nameof(batchSize));

        UploadInputOnly(inputData, batchSize);
        var current = _inputBuf!.View.SubView(0, batchSize * InputSize);
        foreach (var layer in _layers)
            current = layer.Forward(_kernels, _accelerator, current, batchSize);

        int outputSize = batchSize * OutputSize;
        _elementWise.Scale(current.SubView(0, outputSize), _probsBuf!.View.SubView(0, outputSize), outputSize, 1f);
        await _accelerator.SynchronizeAsync().ConfigureAwait(false);
        return await _probsBuf.CopyToHostAsync<float>(0, outputSize).ConfigureAwait(false);
    }

    /// <summary>
    /// Forward + Softmax. Returns calibrated probabilities (rows sum to 1).
    /// Copies only the tiny output vector (batchSize * OutputSize) to the host.
    /// </summary>
    public async Task<float[]> PredictProbsAsync(float[] inputData, int batchSize)
    {
        if (!_built) throw new InvalidOperationException("Call Build() first.");
        if (batchSize < 1 || batchSize > _maxBatch)
            throw new ArgumentOutOfRangeException(nameof(batchSize));

        UploadInputOnly(inputData, batchSize);
        var current = _inputBuf!.View.SubView(0, batchSize * InputSize);
        foreach (var layer in _layers)
            current = layer.Forward(_kernels, _accelerator, current, batchSize);

        int outputSize = batchSize * OutputSize;
        var probs = _probsBuf!.View.SubView(0, outputSize);
        _elementWise.Scale(current.SubView(0, outputSize), probs, outputSize, 1f);
        _softmax.Forward(probs, batchSize, OutputSize);
        await _accelerator.SynchronizeAsync().ConfigureAwait(false);
        return await _probsBuf.CopyToHostAsync<float>(0, outputSize).ConfigureAwait(false);
    }

    private void UploadInputOnly(float[] inputData, int batchSize)
    {
        int inElems = batchSize * InputSize;
        if (inputData.Length != inElems)
            throw new ArgumentException($"Expected input length {inElems}, got {inputData.Length}.", nameof(inputData));
        _inputBuf!.View.SubView(0, inElems).CopyFromCPU(inputData);
    }

    /// <summary>
    /// Serialize architecture + FP32 parameters. Tiny terminal sink (~ParameterCount floats);
    /// intended for localStorage / OPFS cache of System One heads, not bulk activations.
    /// </summary>
    public async Task<byte[]> ExportWeightsAsync()
    {
        if (!_built) throw new InvalidOperationException("Call Build() first.");

        var floats = new List<float>(ParameterCount);
        foreach (var layer in _layers)
            await layer.AppendWeightsAsync(floats).ConfigureAwait(false);

        using var ms = new MemoryStream(64 + floats.Count * sizeof(float));
        using (var bw = new BinaryWriter(ms, System.Text.Encoding.UTF8, leaveOpen: true))
        {
            bw.Write(WeightsMagic); // "TMPL"
            bw.Write(WeightsVersion);
            bw.Write(InputSize);
            bw.Write(OutputSize);
            bw.Write(_layers.Count);
            foreach (var layer in _layers)
                layer.WriteArch(bw);
            bw.Write(floats.Count);
            for (int i = 0; i < floats.Count; i++)
                bw.Write(floats[i]);
        }
        return ms.ToArray();
    }

    /// <summary>
    /// Load FP32 parameters from a blob produced by <see cref="ExportWeightsAsync"/>.
    /// Architecture must already match (same AddLinear/AddReLU sequence and sizes).
    /// </summary>
    public void ImportWeights(ReadOnlySpan<byte> blob)
    {
        if (!_built) throw new InvalidOperationException("Call Build() first.");

        using var ms = new MemoryStream(blob.ToArray());
        using var br = new BinaryReader(ms);
        int magic = br.ReadInt32();
        if (magic != WeightsMagic)
            throw new InvalidDataException($"Bad TrainableModel magic 0x{magic:X8}.");
        int ver = br.ReadInt32();
        if (ver != WeightsVersion)
            throw new InvalidDataException($"Unsupported TrainableModel weights version {ver}.");
        int inSize = br.ReadInt32();
        int outSize = br.ReadInt32();
        if (inSize != InputSize || outSize != OutputSize)
            throw new InvalidDataException(
                $"Arch mismatch: blob {inSize}→{outSize}, model {InputSize}→{OutputSize}.");
        int layerCount = br.ReadInt32();
        if (layerCount != _layers.Count)
            throw new InvalidDataException($"Layer count {layerCount} != {_layers.Count}.");
        for (int i = 0; i < layerCount; i++)
            _layers[i].ValidateArch(br);

        int n = br.ReadInt32();
        if (n != ParameterCount)
            throw new InvalidDataException($"Param count {n} != {ParameterCount}.");
        var floats = new float[n];
        for (int i = 0; i < n; i++)
            floats[i] = br.ReadSingle();

        int offset = 0;
        foreach (var layer in _layers)
            layer.LoadWeights(floats, ref offset);
        if (offset != n)
            throw new InvalidDataException($"Weight consume {offset} != {n}.");
    }

    public Task ImportWeightsAsync(byte[] blob)
    {
        ImportWeights(blob);
        return Task.CompletedTask;
    }

    private const int WeightsMagic = 0x4C504D54; // 'TMPL' LE
    private const int WeightsVersion = 1;

    public void Dispose()
    {
        foreach (var layer in _layers)
            layer.Dispose();
        _inputBuf?.Dispose();
        _targetBuf?.Dispose();
        _probsBuf?.Dispose();
        _lossBuf?.Dispose();
        _gradLogitsBuf?.Dispose();
    }

    private abstract class Layer : IDisposable
    {
        public abstract int ParameterCount { get; }
        public abstract void Allocate(Accelerator accelerator, int maxBatch, Random rng, ElementWiseKernels ew);
        public abstract ArrayView1D<float, Stride1D.Dense> Forward(TrainingKernels k, Accelerator acc, ArrayView1D<float, Stride1D.Dense> input, int batch);
        public abstract ArrayView1D<float, Stride1D.Dense> Backward(TrainingKernels k, Accelerator acc, ArrayView1D<float, Stride1D.Dense> gradOutput, int batch);
        public abstract void UpdateWeights(TrainingKernels k, float lr);
        public abstract void Dispose();
        public virtual Task AppendWeightsAsync(List<float> dst) => Task.CompletedTask;
        public virtual void LoadWeights(ReadOnlySpan<float> src, ref int offset) { }
        public abstract void WriteArch(BinaryWriter bw);
        public abstract void ValidateArch(BinaryReader br);
    }

    private class LinearLayer : Layer
    {
        private readonly int _inF, _outF;
        private MemoryBuffer1D<float, Stride1D.Dense>? _weight;
        private MemoryBuffer1D<float, Stride1D.Dense>? _bias;
        private MemoryBuffer1D<float, Stride1D.Dense>? _gradWeight;
        private MemoryBuffer1D<float, Stride1D.Dense>? _gradBias;
        private MemoryBuffer1D<float, Stride1D.Dense>? _output;
        private MemoryBuffer1D<float, Stride1D.Dense>? _gradInput;
        private ArrayView1D<float, Stride1D.Dense> _savedInput;

        public LinearLayer(int inF, int outF) { _inF = inF; _outF = outF; }
        public override int ParameterCount => _inF * _outF + _outF;

        public override void Allocate(Accelerator acc, int maxBatch, Random rng, ElementWiseKernels ew)
        {
            float std = MathF.Sqrt(2f / _inF);
            var wData = new float[_inF * _outF];
            for (int i = 0; i < wData.Length; i++)
                wData[i] = (float)(rng.NextDouble() * 2 - 1) * std;
            _weight = acc.Allocate1D(wData);
            _bias = acc.Allocate1D<float>(_outF);
            _gradWeight = acc.Allocate1D<float>(_inF * _outF);
            _gradBias = acc.Allocate1D<float>(_outF);
            _output = acc.Allocate1D<float>(maxBatch * _outF);
            _gradInput = acc.Allocate1D<float>(maxBatch * _inF);
        }

        public override ArrayView1D<float, Stride1D.Dense> Forward(TrainingKernels k, Accelerator acc,
            ArrayView1D<float, Stride1D.Dense> input, int batch)
        {
            _savedInput = input;
            k.LinearForward(input, _weight!.View, _output!.View, batch, _inF, _outF);
            k.AddBias(_output!.View, _bias!.View, batch, _outF);
            return _output!.View.SubView(0, batch * _outF);
        }

        public override ArrayView1D<float, Stride1D.Dense> Backward(TrainingKernels k, Accelerator acc,
            ArrayView1D<float, Stride1D.Dense> gradOutput, int batch)
        {
            k.BiasGradient(gradOutput, _gradBias!.View, batch, _outF);
            k.LinearBackwardWeight(gradOutput, _savedInput, _gradWeight!.View, batch, _outF, _inF);
            k.LinearBackwardData(gradOutput, _weight!.View, _gradInput!.View, batch, _outF, _inF);
            return _gradInput!.View.SubView(0, batch * _inF);
        }

        public override void UpdateWeights(TrainingKernels k, float lr)
        {
            k.SGDUpdate(_weight!.View, _gradWeight!.View, _inF * _outF, lr);
            k.SGDUpdate(_bias!.View, _gradBias!.View, _outF, lr);
        }

        public override void Dispose()
        {
            _weight?.Dispose();
            _bias?.Dispose();
            _gradWeight?.Dispose();
            _gradBias?.Dispose();
            _output?.Dispose();
            _gradInput?.Dispose();
        }

        public override async Task AppendWeightsAsync(List<float> dst)
        {
            int wn = _inF * _outF;
            var w = await _weight!.CopyToHostAsync<float>(0, wn).ConfigureAwait(false);
            var b = await _bias!.CopyToHostAsync<float>(0, _outF).ConfigureAwait(false);
            dst.AddRange(w);
            dst.AddRange(b);
        }

        public override void LoadWeights(ReadOnlySpan<float> src, ref int offset)
        {
            int wn = _inF * _outF;
            if (offset + wn + _outF > src.Length)
                throw new ArgumentException("Weight blob too short for linear layer.");
            var w = src.Slice(offset, wn).ToArray();
            offset += wn;
            var b = src.Slice(offset, _outF).ToArray();
            offset += _outF;
            _weight!.View.SubView(0, wn).CopyFromCPU(w);
            _bias!.View.SubView(0, _outF).CopyFromCPU(b);
        }

        public override void WriteArch(BinaryWriter bw)
        {
            bw.Write(1); // linear
            bw.Write(_inF);
            bw.Write(_outF);
        }

        public override void ValidateArch(BinaryReader br)
        {
            int kind = br.ReadInt32();
            if (kind != 1) throw new InvalidDataException($"Expected linear layer, got kind {kind}.");
            int inF = br.ReadInt32();
            int outF = br.ReadInt32();
            if (inF != _inF || outF != _outF)
                throw new InvalidDataException($"Linear size {inF}x{outF} != {_inF}x{_outF}.");
        }
    }

    private class ReLULayer : Layer
    {
        private MemoryBuffer1D<float, Stride1D.Dense>? _savedInput;
        private MemoryBuffer1D<float, Stride1D.Dense>? _output;
        private MemoryBuffer1D<float, Stride1D.Dense>? _gradInput;
        private ElementWiseKernels? _ew;
        private int _capacity;

        public override int ParameterCount => 0;

        public override void Allocate(Accelerator acc, int maxBatch, Random rng, ElementWiseKernels ew)
        {
            _ew = ew;
        }

        public override ArrayView1D<float, Stride1D.Dense> Forward(TrainingKernels k, Accelerator acc,
            ArrayView1D<float, Stride1D.Dense> input, int batch)
        {
            int size = (int)input.Length;
            if (_output == null || _capacity < size)
            {
                _output?.Dispose();
                _savedInput?.Dispose();
                _gradInput?.Dispose();
                _output = acc.Allocate1D<float>(size);
                _savedInput = acc.Allocate1D<float>(size);
                _gradInput = acc.Allocate1D<float>(size);
                _capacity = size;
            }

            _ew ??= new ElementWiseKernels(acc);
            _ew.Scale(input.SubView(0, size), _savedInput!.View.SubView(0, size), size, 1f);
            _ew.ReLU(input.SubView(0, size), _output.View.SubView(0, size), size);
            return _output.View.SubView(0, size);
        }

        public override ArrayView1D<float, Stride1D.Dense> Backward(TrainingKernels k, Accelerator acc,
            ArrayView1D<float, Stride1D.Dense> gradOutput, int batch)
        {
            int size = (int)gradOutput.Length;
            k.ReLUBackward(gradOutput, _savedInput!.View, _gradInput!.View, size);
            return _gradInput!.View.SubView(0, size);
        }

        public override void UpdateWeights(TrainingKernels k, float lr) { }

        public override void Dispose()
        {
            _output?.Dispose();
            _savedInput?.Dispose();
            _gradInput?.Dispose();
        }

        public override void WriteArch(BinaryWriter bw) => bw.Write(2); // relu

        public override void ValidateArch(BinaryReader br)
        {
            int kind = br.ReadInt32();
            if (kind != 2) throw new InvalidDataException($"Expected ReLU layer, got kind {kind}.");
        }
    }
}
