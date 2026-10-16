using SpawnDev.ILGPU.ML.Preprocessing;

namespace SpawnDev.ILGPU.ML.Tests;

/// <summary>ChatTemplates.ParseToolCalls: the shapes models really emit tool calls in, and prose it must leave alone.</summary>
public class ChatTemplatesToolCallTests
{
    [Fact]
    public void TaggedJson_IsParsed()
    {
        var calls = ChatTemplates.ParseToolCalls("<tool_call>\n{\"name\": \"come_here\", \"arguments\": {\"player\": \"Aubri\"}}\n</tool_call>");
        Assert.Single(calls);
        Assert.Equal("come_here", calls[0].Name);
        Assert.Equal("{\"player\": \"Aubri\"}", calls[0].ArgumentsJson);
    }

    /// <summary>ministral-3:8b, measured 2026-10-16 (Vertex): the closing brace of the call was left out before the closing tag, and the whole call was thrown away.</summary>
    [Fact]
    public void TaggedJson_WithAMissingClosingBrace_IsParsed()
    {
        var calls = ChatTemplates.ParseToolCalls("<tool_call> {\"name\": \"build_shape\", \"arguments\": {\"shape\": \"castle\"} </tool_call>");
        Assert.Single(calls);
        Assert.Equal("build_shape", calls[0].Name);
        Assert.Equal("{\"shape\": \"castle\"}", calls[0].ArgumentsJson);
    }

    /// <summary>llama3.1:8b, measured 2026-10-16 (Vertex): an optional argument written as an empty string, then a comma, then the closing braces.</summary>
    [Fact]
    public void TaggedJson_WithATrailingComma_IsParsed()
    {
        var calls = ChatTemplates.ParseToolCalls("<tool_call>" + (char)10 + "{\"name\": \"build_shape\", \"arguments\": {\"shape\": \"house\", \"size\": 5, \"material\": \"\", }}" + (char)10 + "</tool_call>");
        Assert.Single(calls);
        Assert.Equal("build_shape", calls[0].Name);
    }

    [Theory]
    [InlineData("{\"a\": [1, 2", "{\"a\": [1, 2]}")]
    [InlineData("{\"a\": \"x", "{\"a\": \"x\"}")]
    [InlineData("{\"a\": \"brace } inside\"", "{\"a\": \"brace } inside\"}")]
    [InlineData("{\"a\": 1}", "{\"a\": 1}")]
    public void CloseOpenJson_ClosesWhatIsOpen(string json, string expected) => Assert.Equal(expected, ChatTemplates.CloseOpenJson(json));

    /// <summary>llama3.1:8b, measured 2026-10-11 (Vertex): the whole reply was this line, and nothing ran.</summary>
    [Fact]
    public void LlamaPythonicCall_IsParsed()
    {
        var calls = ChatTemplates.ParseToolCalls("come_here(player='Aubri')");
        Assert.Single(calls);
        Assert.Equal("come_here", calls[0].Name);
        Assert.Equal("{\"player\":\"Aubri\"}", calls[0].ArgumentsJson);
    }

    [Fact]
    public void LlamaPythonicList_WithTypes_IsParsed()
    {
        var calls = ChatTemplates.ParseToolCalls("[collect(block=\"dirt\", count=3), stop(), eat(now=True, why=None, speed=1.5)]");
        Assert.Equal(3, calls.Count);
        Assert.Equal(("collect", "{\"block\":\"dirt\",\"count\":3}"), (calls[0].Name, calls[0].ArgumentsJson));
        Assert.Equal(("stop", "{}"), (calls[1].Name, calls[1].ArgumentsJson));
        Assert.Equal(("eat", "{\"now\":true,\"why\":null,\"speed\":1.5}"), (calls[2].Name, calls[2].ArgumentsJson));
    }

    [Fact]
    public void OnlyPythonicCalls_IsTheWholeReplyOrNothing()
    {
        Assert.True(ChatTemplates.IsOnlyPythonicCalls("come_here(player='Aubri')"));
        Assert.True(ChatTemplates.IsOnlyPythonicCalls("[stop(), eat()]"));
        Assert.False(ChatTemplates.IsOnlyPythonicCalls("On my way! come_here(player='Aubri')"));
        Assert.False(ChatTemplates.IsOnlyPythonicCalls("<tool_call>{\"name\": \"stop\"}</tool_call>"));
    }

    [Theory]
    [InlineData("I'm on my way (almost there)!")]
    [InlineData("Sure! come_here(player='Aubri') is what I would do.")]
    [InlineData("That costs f(x) = 2x.")]
    [InlineData("print(hello world)")]
    [InlineData("Hi Aubri, how are you?")]
    public void Prose_IsNeverACall(string text) => Assert.Empty(ChatTemplates.ParseToolCalls(text));
}
