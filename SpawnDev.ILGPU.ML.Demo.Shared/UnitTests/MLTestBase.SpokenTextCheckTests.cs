using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Comparing what a synthesiser was asked to say with what a recogniser heard.
/// </summary>
/// <remarks>
/// This is what lets speech be checked before it is played. It exists because ZipVoice produces garbage
/// on some noise draws - measured, and it happens to the reference implementation too - and the only way
/// to see that from inside the stack is to listen to the output with the recogniser already present.
///
/// The strings below are REAL transcripts from real renders, including the failure that started this:
/// "The lawyer tried to lose his case" came back as "Loner's call, Nanawa, Nenfer. The Loyo Tritu's his
/// case" at one seed and cleanly at three others.
/// </remarks>
public abstract partial class MLTestBase
{
    [TestMethod]
    public async Task SpokenTextCheck_ScoresAGoodRenderAsClean() => await RunTest(_ =>
    {
        // A real transcript of a good render. The leading "Others call me Mother Nature" is the model
        // regenerating its own reference clip ahead of the line, which happens whatever frontend is used
        // and must not be charged - hence the free skip at the head.
        var error = SpokenTextCheck.WordErrorRate(
            "The lawyer tried to lose his case.",
            "Others call me Mother Nature. The lawyer tried to lose his case.");
        if (error > 0.001) throw new Exception($"a clean render scored {error:P0}, expected zero");
        return Task.CompletedTask;
    });

    [TestMethod]
    public async Task SpokenTextCheck_CatchesTheGarbledRender() => await RunTest(_ =>
    {
        // The actual failure this whole mechanism exists for, at seed 1234.
        var error = SpokenTextCheck.WordErrorRate(
            "The lawyer tried to lose his case.",
            "Loner's call, Nanawa, Nenfer. The Loyo Tritu's his case.");
        if (error < 0.3) throw new Exception($"a garbled render scored {error:P0} - too clean to catch");
        return Task.CompletedTask;
    });

    [TestMethod]
    public async Task SpokenTextCheck_ChargesForWordsInsideTheSentence() => await RunTest(_ =>
    {
        // The free skips are at the ENDS only. A word lost in the middle is the thing being measured and
        // must cost, or the check would pass renders that dropped half the line.
        var dropped = SpokenTextCheck.WordErrorRate("feed the white mouse some flower seeds",
                                                    "feed the white mouse and flower seeds");
        if (dropped < 0.1) throw new Exception($"a substituted word scored {dropped:P0}, expected a cost");

        var truncated = SpokenTextCheck.WordErrorRate("feed the white mouse some flower seeds",
                                                      "feed the white mouse");
        if (truncated < 0.4) throw new Exception($"half a sentence scored {truncated:P0}, expected far more");
        return Task.CompletedTask;
    });

    [TestMethod]
    public async Task SpokenTextCheck_IgnoresPunctuationAndCase() => await RunTest(_ =>
    {
        // A recogniser punctuates however it likes; that is not a pronunciation error.
        var error = SpokenTextCheck.WordErrorRate("Take two shares as a fair profit.",
                                                  "take two shares, as a fair profit");
        if (error > 0.001) throw new Exception($"punctuation was charged: {error:P0}");
        return Task.CompletedTask;
    });

    [TestMethod]
    public async Task SpokenTextCheck_HandlesSilence() => await RunTest(_ =>
    {
        // A render that produced nothing must score as total failure, not as a clean pass - an empty
        // transcript is the shape of the worst possible outcome.
        var error = SpokenTextCheck.WordErrorRate("the slush lay deep along the street", "");
        if (error < 0.99) throw new Exception($"an empty transcript scored {error:P0}");
        return Task.CompletedTask;
    });
}
