using System.Text.Json;

namespace OllamaServer.Console;

/// <summary>
/// A model resolved from Ollama's on-disk cache — points DIRECTLY at the content-addressed blobs
/// (zero-copy; we never duplicate the GGUF). <see cref="GgufPath"/> is loadable straight by
/// <c>InferenceSession.CreateFromGGUFFileAsync</c>.
/// </summary>
public sealed record OllamaModel(
    string Name,            // display name as Ollama shows it, e.g. "gemma4:12b" or "ZimaBlueAI/X:latest"
    string GgufPath,        // blobs/sha256-<hex> — the model layer
    long GgufSize,
    string? MmprojPath,     // projector (vision mmproj) blob, or null for text-only
    string? ParamsJson,     // params blob (sampling defaults / stop arrays) JSON, or null
    bool HasOllamaTemplate, // an Ollama "template" layer is present (Go template; secondary to GGUF chat_template)
    bool HasSystem);        // an Ollama "system" layer is present

/// <summary>
/// Reads Ollama's model cache (<c>~/.ollama/models</c> or <c>$OLLAMA_MODELS</c>) — an OCI/Docker
/// content-addressed store — and resolves model names to their GGUF (and projector/params) blobs
/// WITHOUT copying anything. Verified against a real cache (17/17 models resolved: gemma4, gpt-oss,
/// qwen2.5-coder, deepseek-r1, llama3.1, custom spawndev-coder, …).
/// </summary>
public sealed class OllamaModelStore
{
    private const string ModelMedia     = "application/vnd.ollama.image.model";
    private const string ProjectorMedia = "application/vnd.ollama.image.projector";
    private const string ParamsMedia    = "application/vnd.ollama.image.params";
    private const string TemplateMedia  = "application/vnd.ollama.image.template";
    private const string SystemMedia    = "application/vnd.ollama.image.system";

    private readonly string _root;          // …/models
    private readonly string _manifestsRoot; // …/models/manifests

    public OllamaModelStore(string? modelsRoot = null)
    {
        _root = modelsRoot ?? DefaultRoot();
        _manifestsRoot = Path.Combine(_root, "manifests");
    }

    /// <summary>Default cache root: <c>$OLLAMA_MODELS</c> else <c>%USERPROFILE%/.ollama/models</c>.</summary>
    public static string DefaultRoot()
    {
        var env = Environment.GetEnvironmentVariable("OLLAMA_MODELS");
        if (!string.IsNullOrWhiteSpace(env)) return env;
        var home = Environment.GetEnvironmentVariable("USERPROFILE")
                   ?? Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        return Path.Combine(home, ".ollama", "models");
    }

    public bool CacheExists => Directory.Exists(_manifestsRoot);

    /// <summary>Enumerate every cached model whose GGUF blob is present on disk.</summary>
    public IReadOnlyList<OllamaModel> List()
    {
        var result = new List<OllamaModel>();
        if (!CacheExists) return result;

        foreach (var manifestFile in Directory.EnumerateFiles(_manifestsRoot, "*", SearchOption.AllDirectories))
        {
            var m = TryParseManifest(manifestFile);
            if (m != null) result.Add(m);
        }
        result.Sort((a, b) => string.CompareOrdinal(a.Name, b.Name));
        return result;
    }

    /// <summary>
    /// Resolve a requested model name to its blobs. Accepts the forms Ollama clients send:
    /// "gemma4:12b", "gemma4" (→ "gemma4:latest"), "ZimaBlueAI/Model:latest". Returns null if not cached.
    /// </summary>
    public OllamaModel? Resolve(string name)
    {
        if (string.IsNullOrWhiteSpace(name)) return null;
        string wanted = name.Contains(':') ? name : name + ":latest";
        // Match by display name (handles the library/ namespace + tag normalization uniformly).
        foreach (var m in List())
            if (string.Equals(m.Name, wanted, StringComparison.OrdinalIgnoreCase))
                return m;
        return null;
    }

    private OllamaModel? TryParseManifest(string manifestFile)
    {
        try
        {
            using var doc = JsonDocument.Parse(File.ReadAllBytes(manifestFile));
            if (!doc.RootElement.TryGetProperty("layers", out var layers)) return null;

            string? modelDigest = null, projDigest = null, paramsDigest = null;
            long modelSize = 0;
            bool hasTemplate = false, hasSystem = false;

            foreach (var l in layers.EnumerateArray())
            {
                var mt = l.GetProperty("mediaType").GetString();
                var dg = l.TryGetProperty("digest", out var d) ? d.GetString() : null;
                var sz = l.TryGetProperty("size", out var s) ? s.GetInt64() : 0;
                switch (mt)
                {
                    case ModelMedia:     modelDigest = dg; modelSize = sz; break;
                    case ProjectorMedia: projDigest = dg; break;
                    case ParamsMedia:    paramsDigest = dg; break;
                    case TemplateMedia:  hasTemplate = true; break;
                    case SystemMedia:    hasSystem = true; break;
                }
            }
            if (modelDigest == null) return null;

            string ggufPath = BlobPath(modelDigest);
            if (!File.Exists(ggufPath)) return null; // manifest references a blob we don't have

            string? projPath = projDigest != null ? BlobPath(projDigest) : null;
            if (projPath != null && !File.Exists(projPath)) projPath = null;

            string? paramsJson = null;
            if (paramsDigest != null)
            {
                var pp = BlobPath(paramsDigest);
                if (File.Exists(pp)) paramsJson = File.ReadAllText(pp);
            }

            return new OllamaModel(DisplayName(manifestFile), ggufPath, modelSize, projPath, paramsJson,
                hasTemplate, hasSystem);
        }
        catch
        {
            return null; // a malformed/partial manifest must not break listing the rest
        }
    }

    // Manifest digests are "sha256:<hex>"; on disk the blob file is "sha256-<hex>".
    private string BlobPath(string digest) => Path.Combine(_root, "blobs", digest.Replace(':', '-'));

    // …/manifests/registry.ollama.ai/<ns>/<model>/<tag> → "<model>:<tag>" (library ns hidden, like Ollama).
    private string DisplayName(string manifestFile)
    {
        var rel = Path.GetRelativePath(_manifestsRoot, manifestFile).Replace('\\', '/');
        var parts = rel.Split('/');
        if (parts.Length < 4) return rel; // unexpected shape — surface raw rather than guess
        var ns = parts[1];
        var model = string.Join('/', parts[2..^1]);
        var tag = parts[^1];
        var prefix = ns == "library" ? model : $"{ns}/{model}";
        return $"{prefix}:{tag}";
    }
}
