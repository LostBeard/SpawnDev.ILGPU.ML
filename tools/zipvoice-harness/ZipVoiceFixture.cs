using System.Text.Json;

namespace ZipVoiceHarness;

/// <summary>
/// A cloning test case: what to say, the reference clip and its exact transcript, and the token ids
/// the real espeak-ng front end produced for both.
/// </summary>
/// <remarks>
/// The token ids are recorded rather than computed on purpose. Until our own phonemizer exists there
/// is nothing in this repo that can turn English into these ids, and once it does exist these are what
/// it has to match - so the fixture is both the input to the synthesis gate today and the expected
/// output of the phonemizer gate later.
/// </remarks>
public sealed class ZipVoiceFixture
{
    public string Text { get; init; } = "";
    public string PromptText { get; init; } = "";
    public string PromptWav { get; init; } = "prompt.wav";
    public long[] Tokens { get; init; } = Array.Empty<long>();
    public long[] PromptTokens { get; init; } = Array.Empty<long>();

    public static ZipVoiceFixture Load(string path)
    {
        using var document = JsonDocument.Parse(File.ReadAllText(path));
        var root = document.RootElement;

        return new ZipVoiceFixture
        {
            Text = root.GetProperty("text").GetString() ?? "",
            PromptText = root.GetProperty("promptText").GetString() ?? "",
            PromptWav = root.TryGetProperty("promptWav", out var wav) ? wav.GetString() ?? "prompt.wav" : "prompt.wav",
            Tokens = ReadIds(root, "tokens"),
            PromptTokens = ReadIds(root, "promptTokens"),
        };
    }

    private static long[] ReadIds(JsonElement root, string name)
    {
        var array = root.GetProperty(name);
        var ids = new long[array.GetArrayLength()];
        int i = 0;
        foreach (var element in array.EnumerateArray()) ids[i++] = element.GetInt64();
        return ids;
    }
}
