namespace SpawnDev.ILGPU.ML.SystemOne;

public abstract record SystemOneAnswer(SystemOneQuestionType Type);

public sealed record ChoiceAnswer(
    string Choice,
    float Confidence,
    IReadOnlyDictionary<string, float> Probabilities) : SystemOneAnswer(SystemOneQuestionType.Choice);

public sealed record ScoreAnswer(
    float Score,
    float Confidence,
    IReadOnlyDictionary<string, float> Probabilities) : SystemOneAnswer(SystemOneQuestionType.Score);

public sealed record NoulAnswer(float Noul) : SystemOneAnswer(SystemOneQuestionType.Noul);

/// <summary>Result of one <see cref="SystemOneDecisionHead.DecideAsync"/> call.</summary>
public sealed class SystemOneResponse
{
    public required IReadOnlyDictionary<string, SystemOneAnswer> Answers { get; init; }
    public double DecisionLatencyMs { get; init; }
}
