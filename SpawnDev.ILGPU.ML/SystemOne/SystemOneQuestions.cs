namespace SpawnDev.ILGPU.ML.SystemOne;

/// <summary>Typed question kinds aligned with the Jev/Laya System One category.</summary>
public enum SystemOneQuestionType
{
    Choice,
    Score,
    Noul,
}

/// <summary>Base typed question about a float state vector.</summary>
public abstract record SystemOneQuestion(string Instructions);

/// <summary>
/// Pick one labelled option. Criteria keys are the option ids returned in <see cref="ChoiceAnswer"/>.
/// </summary>
public sealed record ChoiceQuestion(
    string Instructions,
    IReadOnlyDictionary<string, string?> Criteria) : SystemOneQuestion(Instructions)
{
    public ChoiceQuestion(string instructions, params string[] optionKeys)
        : this(instructions, optionKeys.ToDictionary(k => k, _ => (string?)null))
    {
    }
}

/// <summary>Rate state on an ordered rubric (low → high). Levels become Softmax classes.</summary>
public sealed record ScoreQuestion(
    string Instructions,
    IReadOnlyList<string> Criteria) : SystemOneQuestion(Instructions);

/// <summary>Calibrated yes/no. Answer is P(true).</summary>
public sealed record NoulQuestion(
    string Instructions,
    string? TrueCriteria = null,
    string? FalseCriteria = null) : SystemOneQuestion(Instructions);
