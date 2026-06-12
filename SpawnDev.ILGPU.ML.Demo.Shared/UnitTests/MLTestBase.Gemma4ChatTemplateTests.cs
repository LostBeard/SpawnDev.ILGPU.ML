using System.Linq;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// gemma4 chat-template structure regression. gemma4's turn markers are CONTROL TOKENS
/// (&lt;|turn&gt; / &lt;turn|&gt; / &lt;|think|&gt;), emitted as single ids — NOT gemma2/3's
/// &lt;start_of_turn&gt; text. This locks the structure ChatTemplates.BuildGemma4PromptTokens emits
/// (bos, system turn with the think toggle, user turn, open model turn) against a synthetic vocab —
/// no GPU, pure CPU logic. The real-model token ids are validated end-to-end by Examples/04.
/// </summary>
public partial class MLTestBase
{
    [TestMethod]
    public async Task Gemma4ChatTemplate_EmitsControlTokenStructure() => await RunTest(async accelerator =>
    {
        // Synthetic gemma4-ish vocab: the control tokens the template needs + minimal text pieces so
        // Encode() of "system\n"/"user\n"/"model\n"/the prompt produces something (exact text ids don't
        // matter — the test asserts CONTROL-token placement).
        string[] toks = { "<unk>", "<bos>", "<eos>", "<|turn>", "<turn|>", "<|think|>",
                          "▁system", "▁user", "▁model", "▁Hi", "\n" };
        var scores = new float[toks.Length];
        var tok = new SentencePieceTokenizer(toks, scores);
        int bos = Idx("<bos>"), turnO = Idx("<|turn>"), turnC = Idx("<turn|>"), think = Idx("<|think|>");
        int Idx(string s) => System.Array.IndexOf(toks, s);

        var ids = ChatTemplates.BuildGemma4PromptTokens(tok, systemPrompt: null, userMessage: "Hi", thinking: true);

        if (ids.Length < 6) throw new Exception($"prompt too short ({ids.Length})");
        if (ids[0] != bos) throw new Exception($"prompt must start with <bos> ({bos}), got {ids[0]}");
        if (ids[1] != turnO) throw new Exception($"<bos> must be followed by <|turn> ({turnO}), got {ids[1]}");
        if (!ids.Contains(think)) throw new Exception("thinking:true must emit the <|think|> toggle");
        int nOpen = ids.Count(i => i == turnO), nClose = ids.Count(i => i == turnC);
        if (nOpen != 3) throw new Exception($"expected 3 <|turn> opens (system/user/model), got {nOpen}");
        if (nClose != 2) throw new Exception($"expected 2 <turn|> closes (system/user; model turn stays OPEN for generation), got {nClose}");
        // The LAST <|turn> (model turn) must NOT be followed by a <turn|> — it's left open to generate.
        int lastOpen = System.Array.LastIndexOf(ids, turnO);
        if (ids.Skip(lastOpen).Contains(turnC))
            throw new Exception("the final model turn must stay OPEN (no <turn|> after the last <|turn>) so generation can continue");

        // thinking:false omits the toggle.
        var noThink = ChatTemplates.BuildGemma4PromptTokens(tok, systemPrompt: null, userMessage: "Hi", thinking: false);
        if (noThink.Contains(think)) throw new Exception("thinking:false must NOT emit <|think|>");

        // The stop-token helper resolves the turn-close id.
        if (ChatTemplates.Gemma4TurnCloseId(tok) != turnC)
            throw new Exception("Gemma4TurnCloseId must return the <turn|> id");

        Console.WriteLine($"[Gemma4ChatTemplate] structure OK: bos→<|turn>system+<|think|>→<turn|>→<|turn>user→<turn|>→<|turn>model(open); " +
            $"{nOpen} opens / {nClose} closes; thinking toggle gated.");
        await Task.CompletedTask;
    });
}
