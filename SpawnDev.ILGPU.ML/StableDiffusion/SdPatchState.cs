using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.StableDiffusion;

/// <summary>
/// The weight edits of one component's store (UNet or text encoders) and how to undo them. Edits apply in call order, like ComfyUI's graph:
/// <list type="bullet">
/// <item>LoRAs (<see cref="SdLoraMerger"/>) are replaceable: a new set first takes the previous one off, back to the weights as they were before it (the file's, or the merged ones).</item>
/// <item>A merge (<see cref="SdMerge"/>) applies to the current weights, LoRAs included, and bakes those LoRAs in: ComfyUI's LoraLoader -> ModelMergeSimple computes exactly that.</item>
/// <item><see cref="ResetAsync"/> returns every edited tensor to the file's weights.</item>
/// </list>
/// The file's weights of an edited tensor are copied on the device before its first edit (a merge with <c>restorable: false</c> skips that copy and makes the reset impossible).
/// </summary>
internal sealed class SdPatchState : IDisposable
{
    private readonly Accelerator _acc;
    private readonly SdWeights _w;
    private readonly Dictionary<string, MemoryBuffer1D<global::ILGPU.Half, Stride1D.Dense>> _file = new(StringComparer.Ordinal), _loraBase = new(StringComparer.Ordinal);
    private readonly HashSet<string> _loraTouched = new(StringComparer.Ordinal);
    private bool _merged, _lossy;

    public SdPatchState(Accelerator acc, SdWeights w) { _acc = acc; _w = w; }

    public SdWeights Weights => _w;

    private void Keep(Dictionary<string, MemoryBuffer1D<global::ILGPU.Half, Stride1D.Dense>> into, string name)
    {
        if (into.ContainsKey(name)) return;
        var src = _w[name];
        var copy = _acc.Allocate1D<global::ILGPU.Half>(src.Length);
        copy.View.CopyFrom(src);
        into[name] = copy;
    }

    /// <summary>Call before a LoRA writes tensor <paramref name="name"/>: keeps what the LoRA set will be taken off to.</summary>
    public void BeforeLora(string name)
    {
        if (_loraTouched.Add(name)) Keep(_merged ? _loraBase : _file, name);
    }

    /// <summary>Take the current LoRA set off.</summary>
    public async Task RestoreLorasAsync()
    {
        if (_loraTouched.Count == 0) return;
        var from = _merged ? _loraBase : _file;
        foreach (var n in _loraTouched) _w[n].CopyFrom(from[n].View);
        _loraTouched.Clear();
        await _acc.SynchronizeAsync();      // the copies must have run before the pre-LoRA buffers go (browser backends batch them)
        FreeLoraBase();
    }

    /// <summary>Call before a merge writes tensor <paramref name="name"/>.</summary>
    public void BeforeMerge(string name, bool restorable)
    {
        if (restorable) Keep(_file, name);
        else if (!_file.ContainsKey(name)) _lossy = true;
    }

    /// <summary>Call after a merge ran (synchronized): the LoRAs applied before it are part of the weights now.</summary>
    public void AfterMerge()
    {
        _loraTouched.Clear();
        FreeLoraBase();
        _merged = true;
    }

    /// <summary>Every edited tensor back to the file's weights.</summary>
    public async Task ResetAsync()
    {
        if (_lossy) throw new InvalidOperationException("a merge made with restorable: false changed weights whose originals were not kept: reload the model to get the file's weights back");
        foreach (var (n, orig) in _file) _w[n].CopyFrom(orig.View);
        await _acc.SynchronizeAsync();
        foreach (var b in _file.Values) b.Dispose();
        _file.Clear();
        FreeLoraBase();
        _loraTouched.Clear();
        _merged = false;
    }

    private void FreeLoraBase()
    {
        foreach (var b in _loraBase.Values) b.Dispose();
        _loraBase.Clear();
    }

    public void Dispose()
    {
        foreach (var b in _file.Values) b.Dispose();
        _file.Clear();
        FreeLoraBase();
    }
}
