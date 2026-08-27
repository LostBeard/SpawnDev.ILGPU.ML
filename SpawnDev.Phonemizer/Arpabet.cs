namespace SpawnDev.Phonemizer;

/// <summary>
/// Turns ARPAbet phones (what CMUdict stores) into the IPA symbols TTS models are trained on.
/// </summary>
/// <remarks>
/// <para>
/// The correspondences here were read off REAL espeak-ng output for real words - about, better, water,
/// roses, understand, garden, served, canoe - captured through sherpa-onnx, not recalled from memory.
/// espeak itself is GPL-3 and none of its code, data or rules are used; it is a measuring instrument
/// here, the same way a compiler is.
/// </para>
/// <para>
/// Two conventions are not obvious and both were verified against captured output:
/// the stress mark goes immediately before the stressed VOWEL rather than before the syllable onset
/// (better is b-STRESS-E-flap-schwar), and several vowels carry an explicit length mark.
/// </para>
/// </remarks>
public static class Arpabet
{
    /// <summary>Primary stress. Placed immediately before the vowel it belongs to.</summary>
    public const string PrimaryStress = "ˈ";

    /// <summary>Secondary stress.</summary>
    public const string SecondaryStress = "ˌ";

    /// <summary>Length mark, appended to the long vowels.</summary>
    public const string Length = "ː";

    /// <summary>The alveolar tap, which American English uses for a T or D between vowels.</summary>
    public const string Flap = "ɾ";

    private static readonly Dictionary<string, string[]> Consonants = new(StringComparer.Ordinal)
    {
        ["B"] = ["b"], ["CH"] = ["t", "ʃ"], ["D"] = ["d"], ["DH"] = ["ð"], ["F"] = ["f"],
        ["G"] = ["ɡ"], ["HH"] = ["h"], ["JH"] = ["d", "ʒ"], ["K"] = ["k"], ["L"] = ["l"],
        ["M"] = ["m"], ["N"] = ["n"], ["NG"] = ["ŋ"], ["P"] = ["p"], ["R"] = ["ɹ"],
        ["S"] = ["s"], ["SH"] = ["ʃ"], ["T"] = ["t"], ["TH"] = ["θ"], ["V"] = ["v"],
        ["W"] = ["w"], ["Y"] = ["j"], ["Z"] = ["z"], ["ZH"] = ["ʒ"],
    };

    /// <summary>Every vowel symbol this mapping can emit. Used by the rules that need to find vowels.</summary>
    public static readonly HashSet<string> Vowels = new(StringComparer.Ordinal)
    {
        "a", "e", "i", "o", "u", "æ", "ɐ", "ɑ", "ɒ", "ɔ", "ə", "ɚ", "ɛ", "ɜ", "ɪ", "ʊ", "ʌ", "ᵻ",
    };

    /// <summary>Map one ARPAbet vowel to IPA, given the stress digit CMUdict attached to it.</summary>
    /// <remarks>
    /// AH and ER genuinely differ by stress rather than merely carrying a mark: unstressed AH is a schwa
    /// while stressed AH is an open back vowel, and unstressed ER is the r-coloured vowel while stressed
    /// ER is a long one. Collapsing either pair mispronounces ordinary words.
    /// </remarks>
    public static string[] Vowel(string phone, int stress) => (phone, stress) switch
    {
        ("AA", _) => ["ɑ", Length],
        ("AE", _) => ["æ"],
        ("AH", 0) => ["ə"],
        ("AH", _) => ["ʌ"],
        ("AO", _) => ["ɔ", Length],
        ("AW", _) => ["a", "ʊ"],
        ("AY", _) => ["a", "ɪ"],
        ("EH", _) => ["ɛ"],
        ("ER", 0) => ["ɚ"],
        ("ER", _) => ["ɜ", Length],
        ("EY", _) => ["e", "ɪ"],
        ("IH", _) => ["ɪ"],
        ("IY", 0) => ["i"],
        ("IY", _) => ["i", Length],
        ("OW", _) => ["o", "ʊ"],
        ("OY", _) => ["ɔ", "ɪ"],
        ("UH", _) => ["ʊ"],
        ("UW", _) => ["u", Length],
        _ => [],
    };

    /// <summary>Map one ARPAbet consonant to IPA. Returns false for anything unrecognised.</summary>
    public static bool TryConsonant(string phone, out string[] symbols) => Consonants.TryGetValue(phone, out symbols!);

    /// <summary>Split an ARPAbet phone into its base and its stress digit, or -1 when it carries none.</summary>
    public static (string Phone, int Stress) SplitStress(string phone)
    {
        if (phone.Length > 0 && char.IsDigit(phone[^1])) return (phone[..^1], phone[^1] - '0');
        return (phone, -1);
    }
}
