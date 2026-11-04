using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Tensors;

namespace SpawnDev.ILGPU.ML.Training;

/// <summary>Options for <see cref="GgufLoraTrainer"/>. Defaults follow PEFT / torch.optim.AdamW.</summary>
public sealed class LoraOptions
{
    /// <summary>GGUF projection names to adapt: attn_q, attn_k, attn_v, attn_output, ffn_gate, ffn_up, ffn_down.</summary>
    public string[] Targets { get; set; } = { "attn_q", "attn_v" };
    public int Rank { get; set; } = 8;
    public float Alpha { get; set; } = 16f;
    public float LearningRate { get; set; } = 1e-4f;
    public float Beta1 { get; set; } = 0.9f;
    public float Beta2 { get; set; } = 0.999f;
    public float Epsilon { get; set; } = 1e-8f;
    public float WeightDecay { get; set; } = 0.01f;
    /// <summary>Label value whose position contributes no loss (the prompt, padding). Hugging Face's convention.</summary>
    public int IgnoreIndex { get; set; } = -100;
    /// <summary>Seed for PEFT-default initialisation (A Kaiming-uniform, B zero) when <see cref="Initial"/> lacks an adapter.</summary>
    public int Seed { get; set; }
    /// <summary>
    /// Recompute each decoder layer's activations in the backward instead of keeping them from the forward (GraphAutograd's checkpointLayers):
    /// needed for long sequences on large models (llama3.1:8b at 2,048 tokens keeps ~20 GB of activations without it), one extra forward of compute.
    /// </summary>
    public bool CheckpointLayers { get; set; }
    /// <summary>
    /// Run the LM head and the loss only at the positions that carry a label (GraphAutograd's labeledPositionsOnly): the full-vocabulary logits
    /// of every position are never formed. Same loss and gradients; <see cref="GgufLoraTrainer.GetLogitsAsync"/> is then unavailable.
    /// </summary>
    public bool LossOnLabeledPositionsOnly { get; set; }
    /// <summary>Starting adapters in PEFT layout (<c>{module}.lora_A.weight</c> [rank, in], <c>{module}.lora_B.weight</c> [out, rank]).</summary>
    public IReadOnlyDictionary<string, float[]>? Initial { get; set; }
}

/// <summary>
/// LoRA fine-tuning of a GGUF language model on the device (Plans/training-autograd-lora-2026-11-04.md). The model's graph gets LoRA
/// adapters and a backward pass (<see cref="GraphAutograd"/>) before it is compiled, so one <see cref="InferenceSession.RunAsync"/> computes
/// the loss and every adapter gradient; AdamW then updates the adapters in place. The base weights are frozen.
/// </summary>
public sealed class GgufLoraTrainer : IDisposable
{
    private readonly Accelerator _acc;
    private readonly InferenceSession _session;
    private readonly LoraOptions _opt;
    private readonly BackwardKernels _kernels;
    private readonly List<(LoraAdapter Adapter, string Param, Tensor Weight, MemoryBuffer1D<float, Stride1D.Dense> M, MemoryBuffer1D<float, Stride1D.Dense> V)> _params = new();
    private readonly TrainingGraphInfo _info;
    private Dictionary<string, Tensor>? _lastOutputs;
    private int _step;

    /// <summary>The adapters this trainer optimises.</summary>
    public IReadOnlyList<LoraAdapter> Adapters { get; }

    /// <summary>AdamW steps taken.</summary>
    public int Step => _step;

    /// <summary>The model's vocabulary size (the logits' last dimension): token ids and labels must lie in [0, VocabSize).</summary>
    public int VocabSize { get; }

    private GgufLoraTrainer(Accelerator acc, InferenceSession session, LoraOptions opt, List<LoraAdapter> adapters, TrainingGraphInfo info, int vocab)
    {
        _acc = acc; _session = session; _opt = opt; _info = info; Adapters = adapters; VocabSize = vocab;
        _kernels = new BackwardKernels(acc);
        _session.OutputsLiveUntilNextRun = true;   // the gradient outputs are read by AdamW after RunAsync returns
        foreach (var a in adapters)
            foreach (var p in new[] { a.AName, a.BName })
            {
                var w = session.TryGetWeight(p) ?? throw new InvalidOperationException($"adapter weight {p} was not uploaded");
                // Initializers of <= 64 elements become host constants at load (CreateFromGGUF), which an in-place update would not reach.
                if (w.ElementCount <= 64) throw new NotSupportedException($"adapter {p} has {w.ElementCount} elements; LoRA tensors must exceed 64 (raise the rank)");
                _params.Add((a, p, w, acc.Allocate1D<float>(w.ElementCount), acc.Allocate1D<float>(w.ElementCount)));
            }
        foreach (var (_, _, _, m, v) in _params) { m.MemSetToZero(); v.MemSetToZero(); }
    }

    /// <summary>Load <paramref name="gguf"/>, add the adapters and the backward pass, and compile.</summary>
    public static GgufLoraTrainer Create(Accelerator accelerator, byte[] gguf, LoraOptions options)
    {
        var t = new Transform(options);
        var session = InferenceSession.CreateFromGGUF(accelerator, gguf, null, false, t.Apply);
        return new GgufLoraTrainer(accelerator, session, options, t.Adapters!, t.Info!, t.Vocab);
    }

    /// <summary>
    /// <see cref="Create"/> for a .gguf FILE, streamed to the device - the only way to load a model past ~2 GB (a byte[] caps there;
    /// llama3.1:8b Q4_K_M is 4.9 GB).
    /// </summary>
    public static async Task<GgufLoraTrainer> CreateFromFileAsync(Accelerator accelerator, string ggufPath, LoraOptions options, CancellationToken ct = default)
    {
        var t = new Transform(options);
        var session = await InferenceSession.CreateFromGGUFFileAsync(accelerator, ggufPath, null, ct, false, t.Apply).ConfigureAwait(false);
        return new GgufLoraTrainer(accelerator, session, options, t.Adapters!, t.Info!, t.Vocab);
    }

    // The graph rewrite both loaders apply: adapters, then loss + backward.
    private sealed class Transform(LoraOptions options)
    {
        public List<LoraAdapter>? Adapters;
        public TrainingGraphInfo? Info;
        public int Vocab;
        public void Apply(Graph.ModelGraph graph, Dictionary<string, float[]> weights)
        {
            var adapters = GraphAutograd.AddLora(graph, weights, options.Targets, options.Rank, options.Alpha, options.Initial, options.Seed);
            var logits = graph.Outputs.FirstOrDefault(o => o.Name == "logits") ?? throw new InvalidOperationException("the graph has no 'logits' output");
            Vocab = logits.Shape[^1];
            Info = GraphAutograd.AddBackward(graph, "logits", Vocab, adapters.SelectMany(a => new[] { a.AName, a.BName }).ToList(), options.IgnoreIndex,
                options.CheckpointLayers, options.LossOnLabeledPositionsOnly);
            Adapters = adapters;
        }
    }

    private Dictionary<string, MemoryBuffer1D<float, Stride1D.Dense>>? _accum;
    private int _accumCount;

    /// <summary>Sequences accumulated since the last <see cref="ApplyAsync"/>.</summary>
    public int AccumulatedSequences => _accumCount;

    /// <summary>
    /// Forward + loss + backward on one sequence and ADD its gradients to the running sum (gradient accumulation: a batch of sequences of any
    /// lengths, one at a time). <see cref="ApplyAsync"/> then takes one AdamW step on their mean. Returns this sequence's loss.
    /// </summary>
    public async Task<float> AccumulateAsync(int[] inputIds, int[] labels)
    {
        float loss = await StepAsync(inputIds, labels, update: false).ConfigureAwait(false);
        _accum ??= _params.ToDictionary(p => p.Param, p => _acc.Allocate1D<float>(p.Weight.ElementCount));
        foreach (var (_, p, w, _, _) in _params)
            _kernels.Axpby(_accum[p].View, _lastOutputs![_info.GradientOutputs[p]].Data, w.ElementCount, _accumCount == 0 ? 0f : 1f, 1f);
        _accumCount++;
        await _acc.SynchronizeAsync().ConfigureAwait(false);
        return loss;
    }

    /// <summary>One AdamW step on the MEAN of the gradients accumulated by <see cref="AccumulateAsync"/>, then reset the sum.</summary>
    public async Task ApplyAsync()
    {
        if (_accumCount == 0 || _accum == null) throw new InvalidOperationException("nothing accumulated");
        _step++;
        float inv = 1f / _accumCount;
        foreach (var (_, p, w, m, v) in _params)
        {
            var g = _accum[p].View;
            if (_accumCount > 1) _kernels.Scale(g, w.ElementCount, inv);   // in place (one binding: WebGPU rejects a buffer bound twice)
            _kernels.AdamW(w.Data, g, m.View, v.View, w.ElementCount, _opt.LearningRate, _opt.Beta1, _opt.Beta2, _opt.Epsilon, _opt.WeightDecay, _step);
        }
        _accumCount = 0;
        await _acc.SynchronizeAsync().ConfigureAwait(false);
    }

    /// <summary>
    /// One training step on one sequence: forward + loss + backward, then (when <paramref name="update"/>) one AdamW step. <paramref name="labels"/>
    /// has one entry per input position: the token that should FOLLOW that position, or <see cref="LoraOptions.IgnoreIndex"/>. Returns the loss.
    /// </summary>
    public async Task<float> StepAsync(int[] inputIds, int[] labels, bool update = true)
    {
        if (inputIds.Length != labels.Length) throw new ArgumentException("one label per input position");
        // Out-of-range ids read past the embedding table and out-of-range labels past a logits row: the GPU would return garbage, not fail.
        for (int t = 0; t < inputIds.Length; t++)
        {
            if ((uint)inputIds[t] >= (uint)VocabSize) throw new ArgumentOutOfRangeException(nameof(inputIds), $"token {inputIds[t]} at position {t} is outside the vocabulary [0, {VocabSize})");
            if (labels[t] != _opt.IgnoreIndex && (uint)labels[t] >= (uint)VocabSize)
                throw new ArgumentOutOfRangeException(nameof(labels), $"label {labels[t]} at position {t} is outside the vocabulary [0, {VocabSize}) and is not IgnoreIndex");
        }
        if (!labels.Any(l => l != _opt.IgnoreIndex)) throw new ArgumentException("every label is IgnoreIndex: the loss would be 0/0");
        using var ids = _acc.Allocate1D(inputIds.Select(i => (float)i).ToArray());
        var inputs = new Dictionary<string, Tensor> { ["input_ids"] = new Tensor(ids.View, new[] { 1, inputIds.Length }, "input_ids") };
        MemoryBuffer1D<float, Stride1D.Dense> lab, pos = null!;
        if (_info.PositionsInput != null)
        {
            // the labelled positions and their labels only
            var at = Enumerable.Range(0, labels.Length).Where(t => labels[t] != _opt.IgnoreIndex).ToArray();
            pos = _acc.Allocate1D(at.Select(t => (float)t).ToArray());
            lab = _acc.Allocate1D(at.Select(t => (float)labels[t]).ToArray());
            inputs[_info.PositionsInput] = new Tensor(pos.View, new[] { at.Length }, _info.PositionsInput);
            inputs[_info.LabelsInput] = new Tensor(lab.View, new[] { 1, at.Length }, _info.LabelsInput);
        }
        else
        {
            lab = _acc.Allocate1D(labels.Select(i => (float)i).ToArray());
            inputs[_info.LabelsInput] = new Tensor(lab.View, new[] { 1, labels.Length }, _info.LabelsInput);
        }
        using var _lab = lab;
        using var _pos = pos;
        var outs = await _session.RunAsync(inputs);
        _lastOutputs = outs;
        float loss = (await outs[_info.LossOutput].Data.SubView(0, 1).CopyToHostAsync())[0];
        if (update)
        {
            _step++;
            foreach (var (_, p, w, m, v) in _params)
                _kernels.AdamW(w.Data, outs[_info.GradientOutputs[p]].Data, m.View, v.View, w.ElementCount,
                    _opt.LearningRate, _opt.Beta1, _opt.Beta2, _opt.Epsilon, _opt.WeightDecay, _step);
        }
        await _acc.SynchronizeAsync();
        return loss;
    }

    /// <summary>The logits of the last <see cref="StepAsync"/> at <paramref name="position"/> (one row of <see cref="VocabSize"/>), computed with the adapters as they were for that step.</summary>
    public async Task<float[]> GetLogitsAsync(int position)
    {
        if (_lastOutputs == null) throw new InvalidOperationException("no step has run");
        if (_info.PositionsInput != null) throw new InvalidOperationException("LossOnLabeledPositionsOnly: the full logits are never formed");
        return await _lastOutputs["logits"].Data.SubView((long)position * VocabSize, VocabSize).CopyToHostAsync();
    }

    /// <summary>The gradients of the last <see cref="StepAsync"/> in PEFT layout (<c>{module}.lora_A.weight</c> [rank, in], <c>lora_B</c> [out, rank]).</summary>
    public async Task<Dictionary<string, float[]>> GetGradientsAsync()
    {
        if (_lastOutputs == null) throw new InvalidOperationException("no step has run");
        var r = new Dictionary<string, float[]>();
        foreach (var a in Adapters)
        {
            var gA = await _lastOutputs[_info.GradientOutputs[a.AName]].Data.SubView(0, a.In * a.Rank).CopyToHostAsync();
            var gB = await _lastOutputs[_info.GradientOutputs[a.BName]].Data.SubView(0, a.Rank * a.Out).CopyToHostAsync();
            r[$"{a.PeftName}.lora_A.weight"] = Transpose(gA, a.In, a.Rank);
            r[$"{a.PeftName}.lora_B.weight"] = Transpose(gB, a.Rank, a.Out);
        }
        return r;
    }

    /// <summary>The current adapters in PEFT layout (the same names as <see cref="LoraOptions.Initial"/>).</summary>
    public async Task<Dictionary<string, float[]>> GetAdaptersAsync()
    {
        var r = new Dictionary<string, float[]>();
        foreach (var a in Adapters)
        {
            var aT = await _session.TryGetWeight(a.AName)!.Data.SubView(0, a.In * a.Rank).CopyToHostAsync();
            var bT = await _session.TryGetWeight(a.BName)!.Data.SubView(0, a.Rank * a.Out).CopyToHostAsync();
            r[$"{a.PeftName}.lora_A.weight"] = Transpose(aT, a.In, a.Rank);
            r[$"{a.PeftName}.lora_B.weight"] = Transpose(bT, a.Rank, a.Out);
        }
        return r;
    }

    /// <summary>
    /// Overwrite the adapters with <paramref name="adapters"/> (PEFT layout and names, as <see cref="GetAdaptersAsync"/> returns them); names not
    /// given keep their values. The AdamW moments are not reset.
    /// </summary>
    public async Task SetAdaptersAsync(IReadOnlyDictionary<string, float[]> adapters)
    {
        foreach (var a in Adapters)
        {
            if (adapters.TryGetValue($"{a.PeftName}.lora_A.weight", out var A))
            {
                if (A.Length != a.Rank * a.In) throw new ArgumentException($"{a.PeftName}.lora_A.weight: {A.Length} values, expected {a.Rank * a.In}");
                _session.TryGetWeight(a.AName)!.Data.SubView(0, A.Length).CopyFromCPU(Transpose(A, a.Rank, a.In));
            }
            if (adapters.TryGetValue($"{a.PeftName}.lora_B.weight", out var B))
            {
                if (B.Length != a.Out * a.Rank) throw new ArgumentException($"{a.PeftName}.lora_B.weight: {B.Length} values, expected {a.Out * a.Rank}");
                _session.TryGetWeight(a.BName)!.Data.SubView(0, B.Length).CopyFromCPU(Transpose(B, a.Out, a.Rank));
            }
        }
        await _acc.SynchronizeAsync();
    }

    /// <summary>
    /// The current adapters as PEFT's two files: <c>adapter_model.safetensors</c> bytes and <c>adapter_config.json</c> text
    /// (<see cref="LoraAdapterFile"/>). Load them back with <see cref="LoraAdapterFile.ReadSafeTensors"/> into <see cref="LoraOptions.Initial"/>.
    /// </summary>
    public async Task<(byte[] SafeTensors, string ConfigJson)> SaveAdaptersAsync(string? baseModel = null)
    {
        var current = await GetAdaptersAsync();
        var tensors = new Dictionary<string, (float[] Data, int[] Shape)>();
        foreach (var a in Adapters)
        {
            tensors[$"{a.PeftName}.lora_A.weight"] = (current[$"{a.PeftName}.lora_A.weight"], new[] { a.Rank, a.In });
            tensors[$"{a.PeftName}.lora_B.weight"] = (current[$"{a.PeftName}.lora_B.weight"], new[] { a.Out, a.Rank });
        }
        var meta = new Dictionary<string, string> { ["format"] = "pt", ["trainer"] = "SpawnDev.ILGPU.ML GgufLoraTrainer", ["steps"] = _step.ToString() };
        return (LoraAdapterFile.WriteSafeTensors(tensors, meta), LoraAdapterFile.PeftConfigJson(Adapters, _opt.Rank, _opt.Alpha, baseModel));
    }

    private static float[] Transpose(float[] src, int rows, int cols)
    {
        var t = new float[src.Length];
        for (int r = 0; r < rows; r++) for (int c = 0; c < cols; c++) t[c * rows + r] = src[r * cols + c];
        return t;
    }

    private bool _disposed;

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (var (_, _, _, m, v) in _params) { m.Dispose(); v.Dispose(); }
        if (_accum != null) foreach (var a in _accum.Values) a.Dispose();
        _kernels.Dispose();
        _session.Dispose();
    }
}
