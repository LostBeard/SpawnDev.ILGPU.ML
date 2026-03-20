using Microsoft.AspNetCore.Components;
using SpawnDev.BlazorJS;
using SpawnDev.BlazorJS.JSObjects;
using System.Net.Http.Json;
using System.Text.Json;

namespace SpawnDev.ILGPU.ML.Demo.Pages;

public partial class VoiceCollabPage : IDisposable
{
    [Inject] BlazorJSRuntime JS { get; set; } = default!;
    [Inject] HttpClient Http { get; set; } = default!;

    // State
    private bool _isListening;
    private bool _isProcessing;
    private bool _speechSupported;
    private bool _speakResponses = true;
    private string _sttEngine = "webspeech";
    private string _llmBackend = "claude";
    private string? _selectedVoice;
    private List<string>? _availableVoices;
    private ElementReference _transcriptEl;

    // Claude API (via CORS proxy)
    private string _proxyUrl = "";
    private string _apiKey = "";
    private string _claudeModel = "claude-sonnet-4-20250514";

    // OpenAI-compatible (Ollama, LM Studio, vLLM, OpenAI, etc.)
    private string _openAiEndpoint = "http://localhost:11434/v1/chat/completions";
    private string _openAiKey = "";
    private string _openAiModel = "llama3";

    // Local GGUF on WebGPU
    private string _localModel = "";
    private List<string> _localModels = new();
    private bool _localModelReady;

    // Speech API objects
    private SpeechRecognition? _recognition;
    private SpeechSynthesis? _synthesis;

    // Transcript
    private List<TranscriptEntry> _transcript = new();

    // Agents
    private List<AgentPersona> _agents = new()
    {
        new AgentPersona
        {
            Name = "Engineer",
            SystemPrompt = "You are a senior software engineer. You write and review code. Be concise and technical. When asked about code, provide specific implementations.",
            VoicePitch = 1.0f,
            VoiceRate = 1.0f,
        },
        new AgentPersona
        {
            Name = "Analyst",
            SystemPrompt = "You are a research analyst and debugger. You investigate issues, read documentation, and provide detailed analysis. Be thorough but focused.",
            VoicePitch = 1.1f,
            VoiceRate = 1.1f,
        },
        new AgentPersona
        {
            Name = "Architect",
            SystemPrompt = "You are a software architect. You focus on system design, trade-offs, and high-level planning. Think about scalability, maintainability, and performance.",
            VoicePitch = 0.9f,
            VoiceRate = 0.95f,
        },
    };

    private AgentPersona _activeAgent = default!;

    protected override void OnInitialized()
    {
        _activeAgent = _agents[0];
    }

    protected override void OnAfterRender(bool firstRender)
    {
        if (firstRender)
        {
            CheckSpeechSupport();
            LoadAvailableVoices();
            StateHasChanged();
        }
    }

    private void CheckSpeechSupport()
    {
        try
        {
            // Check for SpeechRecognition support
            var srType = JS.Get<object?>("SpeechRecognition");
            if (srType == null)
                srType = JS.Get<object?>("webkitSpeechRecognition");
            _speechSupported = srType != null;
        }
        catch
        {
            _speechSupported = false;
        }
    }

    private void LoadAvailableVoices()
    {
        try
        {
            _synthesis = JS.Get<SpeechSynthesis>("speechSynthesis");
            var voices = _synthesis.GetVoices();
            _availableVoices = new List<string>();
            foreach (var voice in voices)
            {
                _availableVoices.Add(voice.Name);
                voice.Dispose();
            }
            if (_availableVoices.Count > 0 && _selectedVoice == null)
                _selectedVoice = _availableVoices[0];
        }
        catch
        {
            _availableVoices = new List<string> { "(default)" };
            _selectedVoice = "(default)";
        }
    }

    private void StartListening()
    {
        if (!_speechSupported) return;
        try
        {
            _recognition = new SpeechRecognition();
            _recognition.Continuous = true;
            _recognition.InterimResults = false;
            _recognition.Lang = "en-US";
            _recognition.OnResult += OnSpeechResult;
            _recognition.OnEnd += OnSpeechEnd;
            _recognition.OnError += OnSpeechError;
            _recognition.Start();
            _isListening = true;
            AddSystemMessage("Listening started. Speak naturally.");
        }
        catch (Exception ex)
        {
            AddSystemMessage($"Failed to start: {ex.Message}");
        }
        StateHasChanged();
    }

    private void StopListening()
    {
        _recognition?.Stop();
        _isListening = false;
        AddSystemMessage("Listening stopped.");
        StateHasChanged();
    }

    private void OnSpeechResult(SpeechRecognitionEvent e)
    {
        try
        {
            using var results = e.Results;
            var length = results.Length;
            for (int i = 0; i < length; i++)
            {
                using var result = results[i];
                if (result.IsFinal)
                {
                    using var alt = result[0];
                    var text = alt.Transcript?.Trim();
                    if (!string.IsNullOrEmpty(text))
                    {
                        AddUserMessage(text);
                        _ = SendToAgentAsync(text);
                    }
                }
            }
        }
        catch { }
    }

    private async Task SendToAgentAsync(string userText)
    {
        _isProcessing = true;
        StateHasChanged();

        try
        {
            var responseText = _llmBackend switch
            {
                "claude" => await SendClaudeAsync(userText),
                "openai" => await SendOpenAiAsync(userText),
                "local" => await SendLocalAsync(userText),
                _ => "(Unknown backend)",
            };

            if (!string.IsNullOrEmpty(responseText))
                AddAgentMessage(_activeAgent.Name, responseText);
            else
                AddAgentMessage(_activeAgent.Name, "(Empty response)");
        }
        catch (Exception ex)
        {
            AddSystemMessage($"Request failed: {ex.Message}");
        }
        finally
        {
            _isProcessing = false;
            StateHasChanged();
        }
    }

    private List<object> BuildConversationMessages()
    {
        var messages = new List<object>();
        var recentEntries = _transcript.TakeLast(20);
        foreach (var entry in recentEntries)
        {
            if (entry.Role == "user")
                messages.Add(new { role = "user", content = entry.Text });
            else if (entry.Role == "agent")
                messages.Add(new { role = "assistant", content = entry.Text });
        }
        return messages;
    }

    private async Task<string> SendClaudeAsync(string userText)
    {
        if (string.IsNullOrEmpty(_proxyUrl))
            return $"(Heard: \"{userText}\" — set a Proxy URL to enable Claude responses.)";

        var messages = BuildConversationMessages();
        var requestBody = new
        {
            model = _claudeModel,
            max_tokens = 1024,
            system = _activeAgent.SystemPrompt,
            messages,
        };

        using var request = new HttpRequestMessage(HttpMethod.Post, _proxyUrl);
        request.Content = JsonContent.Create(requestBody);
        request.Headers.Add("Accept", "application/json");
        if (!string.IsNullOrEmpty(_apiKey))
            request.Headers.Add("x-api-key", _apiKey);

        using var response = await Http.SendAsync(request);
        if (!response.IsSuccessStatusCode)
        {
            var err = await response.Content.ReadAsStringAsync();
            throw new Exception($"Claude API {(int)response.StatusCode}: {err[..Math.Min(200, err.Length)]}");
        }

        var json = await response.Content.ReadFromJsonAsync<JsonElement>();
        var content = json.GetProperty("content");
        var text = "";
        foreach (var block in content.EnumerateArray())
        {
            if (block.GetProperty("type").GetString() == "text")
                text += block.GetProperty("text").GetString();
        }
        return text;
    }

    private async Task<string> SendOpenAiAsync(string userText)
    {
        if (string.IsNullOrEmpty(_openAiEndpoint))
            return $"(Heard: \"{userText}\" — set an endpoint URL.)";

        var messages = new List<object>
        {
            new { role = "system", content = _activeAgent.SystemPrompt },
        };
        messages.AddRange(BuildConversationMessages());

        var requestBody = new
        {
            model = _openAiModel,
            messages,
            max_tokens = 1024,
        };

        using var request = new HttpRequestMessage(HttpMethod.Post, _openAiEndpoint);
        request.Content = JsonContent.Create(requestBody);
        request.Headers.Add("Accept", "application/json");
        if (!string.IsNullOrEmpty(_openAiKey))
            request.Headers.TryAddWithoutValidation("Authorization", $"Bearer {_openAiKey}");

        using var response = await Http.SendAsync(request);
        if (!response.IsSuccessStatusCode)
        {
            var err = await response.Content.ReadAsStringAsync();
            throw new Exception($"OpenAI API {(int)response.StatusCode}: {err[..Math.Min(200, err.Length)]}");
        }

        var json = await response.Content.ReadFromJsonAsync<JsonElement>();
        var choices = json.GetProperty("choices");
        if (choices.GetArrayLength() > 0)
        {
            var message = choices[0].GetProperty("message");
            return message.GetProperty("content").GetString() ?? "";
        }
        return "";
    }

    private Task<string> SendLocalAsync(string userText)
    {
        if (!_localModelReady || string.IsNullOrEmpty(_localModel))
            return Task.FromResult($"(Heard: \"{userText}\" — load a GGUF model on the Models page first.)");

        // TODO: Wire into SpawnDev.ILGPU.ML text generation pipeline
        // When the inference engine supports conversational-speed text gen,
        // this will call the local model directly — no network, no API key.
        return Task.FromResult("(Local inference not yet connected. Load a model and check back when text generation is ready.)");
    }

    private void OnSpeechEnd(Event e)
    {
        // Restart if still in listening mode (continuous recognition can stop)
        if (_isListening && _recognition != null)
        {
            try { _recognition.Start(); } catch { }
        }
    }

    private void OnSpeechError(SpeechRecognitionErrorEvent e)
    {
        var error = e.Error;
        if (error != "no-speech" && error != "aborted")
        {
            AddSystemMessage($"Speech error: {error}");
            StateHasChanged();
        }
    }

    private void SetActiveAgent(AgentPersona agent)
    {
        _activeAgent = agent;
        AddSystemMessage($"Active agent: {agent.Name}");
        StateHasChanged();
    }

    private void AddSystemMessage(string text)
    {
        _transcript.Add(new TranscriptEntry
        {
            Speaker = "System",
            Role = "system",
            Text = text,
            Timestamp = DateTime.Now,
        });
    }

    private void AddUserMessage(string text)
    {
        _transcript.Add(new TranscriptEntry
        {
            Speaker = "You",
            Role = "user",
            Text = text,
            Timestamp = DateTime.Now,
        });
    }

    private void AddAgentMessage(string speaker, string text)
    {
        _transcript.Add(new TranscriptEntry
        {
            Speaker = speaker,
            Role = "agent",
            Text = text,
            Timestamp = DateTime.Now,
        });

        if (_speakResponses)
            SpeakText(text);
    }

    private void ClearTranscript()
    {
        _transcript.Clear();
        StateHasChanged();
    }

    private void SpeakText(string text)
    {
        if (_synthesis == null) return;
        try
        {
            using var utterance = new SpeechSynthesisUtterance(text);
            utterance.Pitch = _activeAgent.VoicePitch;
            utterance.Rate = _activeAgent.VoiceRate;
            if (_selectedVoice != null && _selectedVoice != "(default)")
            {
                var voices = _synthesis.GetVoices();
                foreach (var voice in voices)
                {
                    if (voice.Name == _selectedVoice)
                    {
                        utterance.Voice = voice;
                        // Don't dispose this voice — utterance holds a reference
                        break;
                    }
                    voice.Dispose();
                }
            }
            _synthesis.Speak(utterance);
        }
        catch { }
    }

    public void Dispose()
    {
        if (_recognition != null)
        {
            _recognition.OnResult -= OnSpeechResult;
            _recognition.OnEnd -= OnSpeechEnd;
            _recognition.OnError -= OnSpeechError;
            try { _recognition.Abort(); } catch { }
            _recognition.Dispose();
            _recognition = null;
        }
        _synthesis?.Dispose();
        _synthesis = null;
    }

    // Data classes

    private class TranscriptEntry
    {
        public string Speaker { get; set; } = "";
        public string Role { get; set; } = "";
        public string Text { get; set; } = "";
        public DateTime Timestamp { get; set; }
    }

    private class AgentPersona
    {
        public string Name { get; set; } = "";
        public string SystemPrompt { get; set; } = "";
        public float VoicePitch { get; set; } = 1.0f;
        public float VoiceRate { get; set; } = 1.0f;
    }
}
