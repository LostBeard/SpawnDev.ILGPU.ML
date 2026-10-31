using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// <see cref="SelfConsistency"/>: the last <c>\boxed{}</c> with nested braces, the spellings that must vote together (and the ones that must not), and the vote's order
    /// (most votes, a tie to the first seen, unboxed chains abstain). Host-only: runs on the CPU lane.
    /// </summary>
    [TestMethod]
    public async Task SelfConsistency_ExtractNormalizeVote() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType != AcceleratorType.CPU) throw new UnsupportedTestException("host-only check: runs on the CPU lane");
        void Eq(string? want, string? got, string what) { if (want != got) throw new Exception($"{what}: expected '{want}', got '{got}'"); }
        Eq(@"\frac{\sqrt{3}}{2}", SelfConsistency.ExtractBoxed(@"first \boxed{1}, then so the answer is $\boxed{\frac{\sqrt{3}}{2}}$."), "last box, nested braces");
        Eq("7", SelfConsistency.ExtractBoxed(@"so \boxed 7."), "unbraced box");
        Eq(null, SelfConsistency.ExtractBoxed(@"no answer \boxed{\frac{1}{2}"), "unclosed box");
        Eq(null, SelfConsistency.ExtractBoxed("no box at all"), "no box");
        // the same value spelled differently
        foreach (var (a, b) in new[]
        {
            (@"\dfrac{1}{2}", @"\frac12"), (@"\left( 3, 4 \right)", "(3,4)"), ("x = 5", "5"), (@"10\%", "10"), (@"90^\circ", "90"), (@"5\text{ cm}", "5"),
            ("1,000", "1000"), ("1{,}000", "1000"), ("0.50", ".5"), ("3.0", "3"), (@"\$12", "12"), (@"\sqrt2", @"\sqrt{2}"), ("12.", "12"),
            (@"\begin{pmatrix} -7 \\ 16 \end{pmatrix}", @"\begin{pmatrix}-7\\16\end{pmatrix}"),
        })
            Eq(SelfConsistency.Normalize(b), SelfConsistency.Normalize(a), $"'{a}' and '{b}' must vote together");
        // different answers stay apart
        foreach (var (a, b) in new[] { ("2x", "2"), ("x+y=5", "5"), ("1,2", "12"), (@"\frac{1}{2}", @"\frac{1}{3}"), ("-5", "5"),
            (@"\begin{pmatrix}1\\2\end{pmatrix}", @"\begin{pmatrix}12\end{pmatrix}"), (@"\begin{pmatrix}1\\-2\end{pmatrix}", @"\begin{pmatrix}1\-2\end{pmatrix}") })
            if (SelfConsistency.Normalize(a) == SelfConsistency.Normalize(b)) throw new Exception($"'{a}' and '{b}' must not vote together ({SelfConsistency.Normalize(a)})");
        var vote = SelfConsistency.Vote(new[] { @"\boxed{4}", "no answer", @"\boxed{\dfrac12}", @"\boxed{x=4}", @"\boxed{\frac{1}{2}}", @"\boxed{9}" });
        Eq("4", vote[0].Answer, "the first of two tied groups wins");
        if (vote[0].Votes != 2 || !vote[0].Members.SequenceEqual(new[] { 0, 3 })) throw new Exception($"group 4: {vote[0].Votes} votes, members {string.Join(",", vote[0].Members)}");
        Eq(@"\frac{1}{2}", vote[1].Answer, "second group");
        if (vote.Count != 3) throw new Exception($"{vote.Count} groups (the unboxed chain must abstain)");
        await Task.CompletedTask;
        return "extract, normalise and vote as specified";
    });
}
