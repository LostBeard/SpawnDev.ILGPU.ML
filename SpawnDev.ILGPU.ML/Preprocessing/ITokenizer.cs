namespace SpawnDev.ILGPU.ML.Preprocessing;

/// <summary>
/// Common interface for all tokenizer implementations.
/// Consumed by LoadedTokenizer, InferenceSession, and pipeline code.
/// </summary>
public interface ITokenizer
{
    /// <summary>Size of the vocabulary.</summary>
    int VocabSize { get; }

    /// <summary>Encode text to token IDs.</summary>
    int[] Encode(string text);

    /// <summary>Decode token IDs back to text.</summary>
    string Decode(int[] tokenIds);
}
