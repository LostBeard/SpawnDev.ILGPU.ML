using System.Text;
using System.Text.RegularExpressions;

namespace SpawnDev.ILGPU.ML.Pipelines;

/// <summary>
/// Self-consistency voting (Wang et al. 2022): sample several reasoning chains for one question, take each chain's final answer, and answer with the most common one.
/// The final answer is the last <c>\boxed{...}</c> (the convention math models are trained on); answers are compared after <see cref="Normalize"/>, which folds the
/// spellings LaTeX allows for one value (<c>\dfrac</c> / <c>\frac</c>, <c>\left( \right)</c>, spacing, units in <c>\text{}</c>, degree signs, thousands separators,
/// <c>x = 5</c> / <c>5</c>, trailing zeros). It does not prove two different forms equal (<c>1/2</c> and <c>0.5</c> stay apart): a symbolic checker would.
/// Plans/test-time-search-2026-10-31.md: on 50 MATH-500 problems, the majority of 8 samples took Qwen2.5-Math-1.5B from 66% to 80%.
/// </summary>
public static class SelfConsistency
{
    /// <summary>The contents of the last <c>\boxed{...}</c> (or <c>\fbox{...}</c>) in <paramref name="text"/>, braces balanced; null when there is none or it is unclosed.</summary>
    public static string? ExtractBoxed(string text)
    {
        int at = Math.Max(text.LastIndexOf(@"\boxed", StringComparison.Ordinal), text.LastIndexOf(@"\fbox", StringComparison.Ordinal));
        if (at < 0) return null;
        int start = at + (text.AsSpan(at).StartsWith(@"\boxed") ? 6 : 5), open = start;
        while (open < text.Length && char.IsWhiteSpace(text[open])) open++;
        if (open >= text.Length) return null;
        if (text[open] != '{')
        {
            // "\boxed 5" (no braces) is legal LaTeX: one token (a sentence's full stop is not part of it)
            var m = Regex.Match(text[open..], @"^[^\s$]+");
            return m.Success ? m.Value.TrimEnd('.', ',', ';') : null;
        }
        int depth = 0;
        for (int i = open; i < text.Length; i++)
        {
            if (text[i] == '\\' && i + 1 < text.Length && (text[i + 1] == '{' || text[i + 1] == '}')) { i++; continue; }
            if (text[i] == '{') depth++;
            else if (text[i] == '}' && --depth == 0) return text[(open + 1)..i];
        }
        return null;
    }

    /// <summary>A canonical spelling of an answer for voting (see the class summary for what it folds).</summary>
    public static string Normalize(string answer)
    {
        string s = answer.Trim();
        // \text{...} / \textbf{...} / \mbox{...}: keep the words (a unit is dropped below when a number precedes it)
        s = Regex.Replace(s, @"\\(?:text|textbf|textrm|mathrm|mbox)\s*\{([^{}]*)\}", "$1");
        s = s.Replace(@"\dfrac", @"\frac").Replace(@"\tfrac", @"\frac");
        // a row break "\\" (matrices, cases) is kept whole: the spacing commands below are a backslash and one character, and "\\ " is not "\" + "\ "
        const char RowBreak = '\u0001';
        s = s.Replace(@"\\", RowBreak.ToString());
        s = Regex.Replace(s, @"\\(?:left|right)(?![a-zA-Z])", "");
        foreach (var t in new[] { @"\!", @"\,", @"\;", @"\:", @"\ ", @"\$", "$", @"\displaystyle" }) s = s.Replace(t, "");
        s = s.Replace(@"^\circ", "").Replace(@"^{\circ}", "").Replace("°", "").Replace(@"\%", "").Replace("%", "");
        s = Regex.Replace(s, @"\s+", "").Replace(RowBreak.ToString(), @"\\");
        s = s.TrimEnd('.');
        // "x=5", "n=12": a single-letter assignment is its value
        var assign = Regex.Match(s, @"^[a-zA-Z]=(.+)$");
        if (assign.Success) s = assign.Groups[1].Value;
        // \frac12 -> \frac{1}{2}, \sqrt2 -> \sqrt{2}
        s = Regex.Replace(s, @"\\frac(\d)(\d)", @"\frac{$1}{$2}");
        s = Regex.Replace(s, @"\\frac\{([^{}]+)\}(\d)", @"\frac{$1}{$2}");
        s = Regex.Replace(s, @"\\sqrt(\d+)", @"\sqrt{$1}");
        // thousands separators: 1,000 / 1{,}000 / 1\,000 (the last already gone)
        s = s.Replace("{,}", ",");
        s = Regex.Replace(s, @"^(\d{1,3})(,\d{3})+(\.\d+)?$", m => m.Value.Replace(",", ""));
        // a number followed by a unit word: the number ("5 cm", "12 dollars"; "\$" went with "$"); one letter stays ("2x" is an expression)
        var unit = Regex.Match(s, @"^(-?\d+(?:\.\d+)?)[a-zA-Z]{2,}$");
        if (unit.Success) s = unit.Groups[1].Value;
        // plain decimals: 0.50 -> 0.5, 3.0 -> 3, .5 -> 0.5
        if (Regex.IsMatch(s, @"^-?\d*\.\d+$"))
        {
            if (s.StartsWith(".")) s = "0" + s; else if (s.StartsWith("-.")) s = "-0" + s[1..];
            s = s.TrimEnd('0').TrimEnd('.');
        }
        return s;
    }

    /// <summary>One answer group of a vote: the normalised answer, how many chains gave it, and which (indices into the voted texts).</summary>
    public sealed record Group(string Answer, int Votes, int[] Members);

    /// <summary>
    /// Vote over the chains' final answers: groups by <see cref="Normalize"/>d boxed answer, most votes first (a tie goes to the answer seen first). Chains without a boxed
    /// answer do not vote. <see cref="Group.Answer"/> of the first group is the self-consistency answer (null when no chain boxed one).
    /// </summary>
    public static IReadOnlyList<Group> Vote(IReadOnlyList<string> texts)
    {
        var order = new List<string>();
        var members = new Dictionary<string, List<int>>();
        for (int i = 0; i < texts.Count; i++)
        {
            var box = ExtractBoxed(texts[i]);
            if (box == null) continue;
            var key = Normalize(box);
            if (!members.TryGetValue(key, out var list)) { members[key] = list = new List<int>(); order.Add(key); }
            list.Add(i);
        }
        return order.Select((k, first) => (k, first)).OrderByDescending(x => members[x.k].Count).ThenBy(x => x.first)
            .Select(x => new Group(x.k, members[x.k].Count, members[x.k].ToArray())).ToList();
    }
}
