namespace SpawnDev.ILGPU.ML.Tensors;

/// <summary>
/// Host-side ONNX value that is not (only) a float GPU <see cref="Tensor"/>.
/// Sequence / Optional / String / Bytes live here; float activations stay on GPU as <see cref="Tensor"/>.
/// </summary>
public enum OnnxValueKind
{
    Tensor,
    Sequence,
    Optional,
    String,
    Bytes,
}

/// <summary>
/// Discriminated ONNX value for types the float GPU path cannot represent.
/// </summary>
public sealed class OnnxValue
{
    public OnnxValueKind Kind { get; }

    /// <summary>GPU float tensor when <see cref="Kind"/> is <see cref="OnnxValueKind.Tensor"/>.</summary>
    public Tensor? Tensor { get; }

    /// <summary>Sequence elements when <see cref="Kind"/> is <see cref="OnnxValueKind.Sequence"/>.</summary>
    public IReadOnlyList<OnnxValue>? Sequence { get; }

    /// <summary>Wrapped value when <see cref="Kind"/> is <see cref="OnnxValueKind.Optional"/> and present.</summary>
    public OnnxValue? OptionalValue { get; }

    /// <summary>True when Optional has a value (including an empty sequence / empty string).</summary>
    public bool OptionalHasValue { get; }

    public string? String { get; }

    public byte[]? Bytes { get; }

    private OnnxValue(OnnxValueKind kind, Tensor? tensor = null, IReadOnlyList<OnnxValue>? sequence = null,
        OnnxValue? optionalValue = null, bool optionalHasValue = false, string? str = null, byte[]? bytes = null)
    {
        Kind = kind;
        Tensor = tensor;
        Sequence = sequence;
        OptionalValue = optionalValue;
        OptionalHasValue = optionalHasValue;
        String = str;
        Bytes = bytes;
    }

    public static OnnxValue FromTensor(Tensor tensor)
    {
        ArgumentNullException.ThrowIfNull(tensor);
        return new OnnxValue(OnnxValueKind.Tensor, tensor: tensor);
    }

    public static OnnxValue FromSequence(IReadOnlyList<OnnxValue> items)
    {
        ArgumentNullException.ThrowIfNull(items);
        return new OnnxValue(OnnxValueKind.Sequence, sequence: items.ToList());
    }

    public static OnnxValue EmptySequence() => new(OnnxValueKind.Sequence, sequence: Array.Empty<OnnxValue>());

    public static OnnxValue FromOptional(OnnxValue? value) =>
        value == null
            ? new OnnxValue(OnnxValueKind.Optional, optionalHasValue: false)
            : new OnnxValue(OnnxValueKind.Optional, optionalValue: value, optionalHasValue: true);

    public static OnnxValue EmptyOptional() => new(OnnxValueKind.Optional, optionalHasValue: false);

    public static OnnxValue FromString(string value)
    {
        ArgumentNullException.ThrowIfNull(value);
        return new OnnxValue(OnnxValueKind.String, str: value);
    }

    public static OnnxValue FromStrings(IReadOnlyList<string> values)
    {
        // ONNX string tensors are rank-1 sequences of strings for our host path.
        ArgumentNullException.ThrowIfNull(values);
        var items = new OnnxValue[values.Count];
        for (int i = 0; i < values.Count; i++) items[i] = FromString(values[i]);
        return FromSequence(items);
    }

    public static OnnxValue FromBytes(byte[] value)
    {
        ArgumentNullException.ThrowIfNull(value);
        return new OnnxValue(OnnxValueKind.Bytes, bytes: value);
    }

    public Tensor AsTensor() =>
        Kind == OnnxValueKind.Tensor && Tensor != null
            ? Tensor
            : throw new InvalidOperationException($"OnnxValue Kind={Kind} is not a Tensor.");

    public IReadOnlyList<OnnxValue> AsSequence() =>
        Kind == OnnxValueKind.Sequence && Sequence != null
            ? Sequence
            : throw new InvalidOperationException($"OnnxValue Kind={Kind} is not a Sequence.");

    public string AsString() =>
        Kind == OnnxValueKind.String && String != null
            ? String
            : throw new InvalidOperationException($"OnnxValue Kind={Kind} is not a String.");

    public byte[] AsBytes() =>
        Kind == OnnxValueKind.Bytes && Bytes != null
            ? Bytes
            : throw new InvalidOperationException($"OnnxValue Kind={Kind} is not Bytes.");
}
