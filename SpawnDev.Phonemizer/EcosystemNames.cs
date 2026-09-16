namespace SpawnDev.Phonemizer;

/// <summary>
/// Proper nouns this ecosystem says out loud that CMUdict has never heard of.
/// </summary>
/// <remarks>
/// <para>
/// 🔴 A NAME THE LETTER-TO-SOUND MODEL GUESSES IS WRONG IN A WAY NOTHING CATCHES. It is not dropped, it is
/// not unmapped, and it does not fail any check: every phoneme it produces is a real phoneme with a real
/// token, so the frontend reports 100% coverage while the voice says something else entirely. The only
/// instrument is a person listening.
/// </para>
/// <para>
/// MEASURED 2026-09-16: <c>"Reachy"</c> came out as <c>ɹiˈeɪki</c> - "ree-AY-kee", three syllables, with
/// the stress on a vowel the word does not contain. The rules are behaving correctly; English spells
/// <c>ch</c> as /k/ often enough (<i>ache</i>, <i>stomach</i>, <i>chrome</i>) that a model learned from a
/// dictionary will reach for it. TJ heard it in the robot's own voice: <i>"the voice pronounces Reachy as
/// ree-kee but it should be ree-chee"</i>.
/// </para>
/// <para>
/// ⚠️ THESE ARE NAMES, NOT POLICY. A pronunciation for a word that exists in the world belongs in the
/// dictionary, the same as any other entry - it is not app configuration and should not be duplicated per
/// host. Applied in <see cref="EmbeddedData.CreatePhonemizer"/>, so every consumer gets it: Kokoro and
/// ZipVoice, browser and desktop, this repo and the next one.
/// </para>
/// <para>
/// ⚠️ A user-supplied definition must still win. <see cref="Apply"/> runs at construction, so a caller's
/// later <c>Define</c> overrides anything here, which is the right order for a name someone pronounces
/// differently to us.
/// </para>
/// </remarks>
public static class EcosystemNames
{
    /// <summary>
    /// Word to ARPAbet, for names the bundled dictionary does not carry.
    /// </summary>
    /// <remarks>
    /// Stress digits matter as much as the phonemes: <c>1</c> is primary, <c>0</c> unstressed. Getting
    /// them wrong moves the emphasis without changing a single sound, which is how a name ends up
    /// recognisable but wrong.
    /// </remarks>
    public static IReadOnlyDictionary<string, string> Pronunciations { get; } =
        new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
        {
            // Reachy Mini, the Pollen Robotics robot. REE-chee.
            ["reachy"] = "R IY1 CH IY0",
            // Kept beside it because they are said in the same breath and are equally absent from CMUdict.
            ["pollen"] = "P AA1 L AH0 N",
            ["spawndev"] = "S P AO1 N D EH2 V",
            ["blazor"] = "B L EY1 Z ER0",
            ["gemineachy"] = "JH EH1 M IH0 N IY2 CH IY0",
        };

    /// <summary>Teach <paramref name="phonemizer"/> every name above.</summary>
    public static void Apply(EnglishPhonemizer phonemizer)
    {
        ArgumentNullException.ThrowIfNull(phonemizer);
        foreach (var (word, arpabet) in Pronunciations) phonemizer.Define(word, arpabet);
    }
}
