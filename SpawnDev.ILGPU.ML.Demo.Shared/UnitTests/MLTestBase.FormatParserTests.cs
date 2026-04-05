using System.IO.Compression;
using System.Text;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.GGUF;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.ILGPU.ML.SafeTensors;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Unit tests for ALL model format parsers — synthetic in-memory data, no external files.
/// Each test constructs minimal valid binary data and verifies the parser extracts correct metadata.
/// </summary>
public abstract partial class MLTestBase
{
    // ═══════════════════════════════════════════════════════════
    //  GGUF Parser
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task GGUFParser_MinimalValid_ParsesMetadata() => await RunTest(async accelerator =>
    {
        // Build minimal GGUF: version 2, 0 metadata, 1 tensor (4 floats)
        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms);
        bw.Write((byte)'G'); bw.Write((byte)'G'); bw.Write((byte)'U'); bw.Write((byte)'F'); // magic
        bw.Write((uint)2); // version
        bw.Write((ulong)1); // tensor count
        bw.Write((ulong)0); // metadata count
        // Tensor info: name "w", 1D, 4 elements, F32, offset 0
        bw.Write((ulong)1); bw.Write((byte)'w'); // name
        bw.Write((uint)1); // nDims
        bw.Write((ulong)4); // dim[0]
        bw.Write((uint)0); // GGMLType.F32
        bw.Write((ulong)0); // data offset
        // Pad to 32-byte alignment
        while (ms.Position % 32 != 0) bw.Write((byte)0);
        // Tensor data: 4 floats
        for (int i = 0; i < 4; i++) bw.Write(1.0f * (i + 1));
        bw.Flush();

        var data = ms.ToArray();
        var model = GGUFParser.Parse(data);
        if (model.Version != 2) throw new Exception($"Version: expected 2, got {model.Version}");
        if (model.Tensors.Length != 1) throw new Exception($"Tensors: expected 1, got {model.Tensors.Length}");
        if (model.Tensors[0].Name != "w") throw new Exception($"Name: expected 'w', got '{model.Tensors[0].Name}'");
        if (model.Tensors[0].Dimensions[0] != 4) throw new Exception($"Dim: expected 4, got {model.Tensors[0].Dimensions[0]}");
        Console.WriteLine("[GGUF] Minimal parse — PASS");
        await Task.CompletedTask;
    });

    [TestMethod]
    public async Task GGUFParser_WithMetadata_ReadsStringsAndInts() => await RunTest(async accelerator =>
    {
        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms);
        bw.Write((byte)'G'); bw.Write((byte)'G'); bw.Write((byte)'U'); bw.Write((byte)'F');
        bw.Write((uint)3); // version 3
        bw.Write((ulong)0); // 0 tensors
        bw.Write((ulong)2); // 2 metadata KV pairs
        // KV 1: "general.architecture" = "llama" (string)
        WriteGGUFString(bw, "general.architecture");
        bw.Write((uint)8); // GGUFValueType.String
        WriteGGUFString(bw, "llama");
        // KV 2: "llama.context_length" = 4096 (uint32)
        WriteGGUFString(bw, "llama.context_length");
        bw.Write((uint)4); // GGUFValueType.UInt32
        bw.Write((uint)4096);
        bw.Flush();

        var data = ms.ToArray();
        var model = GGUFParser.Parse(data);
        if (model.Architecture != "llama") throw new Exception($"Architecture: expected 'llama', got '{model.Architecture}'");
        long ctxLen = model.GetMetadataInt("llama.context_length");
        if (ctxLen != 4096) throw new Exception($"Context length: expected 4096, got {ctxLen}");
        Console.WriteLine("[GGUF] Metadata parse — PASS");
        await Task.CompletedTask;
    });

    [TestMethod]
    public async Task GGUFParser_WithTokenizer_ReadsVocab() => await RunTest(async accelerator =>
    {
        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms);
        bw.Write((byte)'G'); bw.Write((byte)'G'); bw.Write((byte)'U'); bw.Write((byte)'F');
        bw.Write((uint)3);
        bw.Write((ulong)0); // 0 tensors
        bw.Write((ulong)2); // 2 metadata: architecture + tokens
        // KV 1: architecture
        WriteGGUFString(bw, "general.architecture");
        bw.Write((uint)8); WriteGGUFString(bw, "llama");
        // KV 2: tokenizer.ggml.tokens = ["<unk>", "<s>", "</s>", "hello"]
        WriteGGUFString(bw, "tokenizer.ggml.tokens");
        bw.Write((uint)9); // GGUFValueType.Array
        bw.Write((uint)8); // element type = String
        bw.Write((ulong)4); // count
        WriteGGUFString(bw, "<unk>");
        WriteGGUFString(bw, "<s>");
        WriteGGUFString(bw, "</s>");
        WriteGGUFString(bw, "hello");
        bw.Flush();

        var data = ms.ToArray();
        var model = GGUFParser.Parse(data);
        var tokens = model.GetMetadataStringArray("tokenizer.ggml.tokens");
        if (tokens == null) throw new Exception("Tokens array is null");
        if (tokens.Length != 4) throw new Exception($"Tokens: expected 4, got {tokens.Length}");
        if (tokens[3] != "hello") throw new Exception($"Token[3]: expected 'hello', got '{tokens[3]}'");
        Console.WriteLine("[GGUF] Tokenizer vocab parse — PASS");
        await Task.CompletedTask;
    });

    [TestMethod]
    public async Task GGUFParser_Detection_WorksCorrectly() => await RunTest(async accelerator =>
    {
        if (!GGUFParser.IsGGUF(new byte[] { 0x47, 0x47, 0x55, 0x46, 0x02, 0x00, 0x00, 0x00 }))
            throw new Exception("Failed to detect valid GGUF");
        if (GGUFParser.IsGGUF(new byte[] { 0x00, 0x00, 0x00, 0x00 }))
            throw new Exception("False positive on non-GGUF");
        if (GGUFParser.IsGGUF(new byte[] { 0x47, 0x47 })) // too short
            throw new Exception("False positive on truncated data");
        Console.WriteLine("[GGUF] Detection — PASS");
        await Task.CompletedTask;
    });

    private static void WriteGGUFString(BinaryWriter bw, string s)
    {
        var bytes = Encoding.UTF8.GetBytes(s);
        bw.Write((ulong)bytes.Length);
        bw.Write(bytes);
    }

    // ═══════════════════════════════════════════════════════════
    //  SafeTensors Parser
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task SafeTensors_MinimalValid_ParsesTensor() => await RunTest(async accelerator =>
    {
        string json = """{"weight":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}""";
        byte[] jsonBytes = Encoding.UTF8.GetBytes(json);
        using var ms = new MemoryStream();
        ms.Write(BitConverter.GetBytes((long)jsonBytes.Length));
        ms.Write(jsonBytes);
        ms.Write(new byte[16]); // 4 × float32 zeros

        var data = ms.ToArray();
        var parsed = SafeTensorsParser.Parse(data);
        if (parsed.Tensors.Length != 1) throw new Exception($"Expected 1 tensor, got {parsed.Tensors.Length}");
        var t = parsed.Tensors[0];
        if (t.Name != "weight") throw new Exception($"Name: expected 'weight', got '{t.Name}'");
        if (t.Shape[0] != 4) throw new Exception($"Shape: expected [4], got [{t.Shape[0]}]");
        if (t.DType != "F32") throw new Exception($"DType: expected F32, got {t.DType}");
        Console.WriteLine("[SafeTensors] Minimal parse — PASS");
        await Task.CompletedTask;
    });

    [TestMethod]
    public async Task SafeTensors_MultipleTensors_AllParsed() => await RunTest(async accelerator =>
    {
        string json = """{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"F32","shape":[3],"data_offsets":[8,20]}}""";
        byte[] jsonBytes = Encoding.UTF8.GetBytes(json);
        using var ms = new MemoryStream();
        ms.Write(BitConverter.GetBytes((long)jsonBytes.Length));
        ms.Write(jsonBytes);
        ms.Write(new byte[20]); // 8 + 12 bytes data

        var parsed = SafeTensorsParser.Parse(ms.ToArray());
        if (parsed.Tensors.Length != 2) throw new Exception($"Expected 2 tensors, got {parsed.Tensors.Length}");
        var ta = parsed.Tensors.FirstOrDefault(t => t.Name == "a");
        var tb = parsed.Tensors.FirstOrDefault(t => t.Name == "b");
        if (ta == null || ta.Shape[0] != 2) throw new Exception("Tensor 'a' shape wrong");
        if (tb == null || tb.Shape[0] != 3) throw new Exception("Tensor 'b' shape wrong");
        Console.WriteLine("[SafeTensors] Multi-tensor — PASS");
        await Task.CompletedTask;
    });

    [TestMethod]
    public async Task SafeTensors_Detection_WorksCorrectly() => await RunTest(async accelerator =>
    {
        // Valid: uint64 header size, then '{'
        var valid = new byte[16];
        BitConverter.GetBytes((long)2).CopyTo(valid, 0); // header size = 2
        valid[8] = (byte)'{'; valid[9] = (byte)'}';
        if (!SafeTensorsParser.IsSafeTensors(valid))
            throw new Exception("Failed to detect valid SafeTensors");
        if (SafeTensorsParser.IsSafeTensors(new byte[] { 0, 0, 0, 0, 0, 0, 0, 0, 0 }))
            throw new Exception("False positive on non-SafeTensors");
        Console.WriteLine("[SafeTensors] Detection — PASS");
        await Task.CompletedTask;
    });

    // ═══════════════════════════════════════════════════════════
    //  ONNX Parser
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task OnnxParser_MinimalValid_Parses() => await RunTest(async accelerator =>
    {
        // Minimal: ir_version=7, one opset version=17
        byte[] minimal = { 0x08, 0x07, 0x42, 0x02, 0x10, 0x11 };
        var model = Onnx.OnnxParser.Parse(minimal);
        if (model.IrVersion != 7) throw new Exception($"ir_version: expected 7, got {model.IrVersion}");
        if (model.OpsetImports.Count != 1) throw new Exception($"Opsets: expected 1, got {model.OpsetImports.Count}");
        if (model.OpsetImports[0].Version != 17) throw new Exception($"Opset version: expected 17, got {model.OpsetImports[0].Version}");
        Console.WriteLine("[ONNX] Minimal parse — PASS");
        await Task.CompletedTask;
    });

    [TestMethod]
    public async Task OnnxParser_EmptyData_DoesNotThrow() => await RunTest(async accelerator =>
    {
        // Empty protobuf is valid — all fields default
        var model = Onnx.OnnxParser.Parse(Array.Empty<byte>());
        if (model.IrVersion != 0) throw new Exception($"Empty should have ir_version=0, got {model.IrVersion}");
        Console.WriteLine("[ONNX] Empty parse — PASS");
        await Task.CompletedTask;
    });

    // ═══════════════════════════════════════════════════════════
    //  CoreML Parser
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task CoreMLParser_MinimalValid_Parses() => await RunTest(async accelerator =>
    {
        // Minimal CoreML: specificationVersion = 4 (field 1, varint)
        byte[] minimal = { 0x08, 0x04 };
        var model = CoreML.CoreMLParser.Parse(minimal);
        if (model.SpecVersion != 4)
            throw new Exception($"specVersion: expected 4, got {model.SpecVersion}");
        Console.WriteLine("[CoreML] Minimal parse — PASS");
        await Task.CompletedTask;
    });

    [TestMethod]
    public async Task CoreMLParser_Detection_WorksCorrectly() => await RunTest(async accelerator =>
    {
        if (!CoreML.CoreMLParser.IsCoreML(new byte[] { 0x08, 0x04 }))
            throw new Exception("Failed to detect valid CoreML");
        if (!CoreML.CoreMLParser.IsCoreML(new byte[] { 0x08, 0x01 }))
            throw new Exception("Failed to detect CoreML v1");
        if (CoreML.CoreMLParser.IsCoreML(new byte[] { 0x08, 0x00 }))
            throw new Exception("False positive: version 0 is not valid");
        if (CoreML.CoreMLParser.IsCoreML(new byte[] { 0x08, 0x20 }))
            throw new Exception("False positive: version 32 is too high");
        if (CoreML.CoreMLParser.IsCoreML(new byte[] { 0x00, 0x04 }))
            throw new Exception("False positive: wrong field tag");
        Console.WriteLine("[CoreML] Detection — PASS");
        await Task.CompletedTask;
    });

    // ═══════════════════════════════════════════════════════════
    //  TensorFlow GraphDef Parser
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task TFGraphDef_EmptyValid_Parses() => await RunTest(async accelerator =>
    {
        var graphDef = TensorFlow.TFGraphDefParser.Parse(Array.Empty<byte>());
        if (graphDef.Nodes.Count != 0)
            throw new Exception($"Empty graph should have 0 nodes, got {graphDef.Nodes.Count}");
        Console.WriteLine("[TFGraphDef] Empty parse — PASS");
        await Task.CompletedTask;
    });

    [TestMethod]
    public async Task TFGraphDef_Detection_WorksCorrectly() => await RunTest(async accelerator =>
    {
        // Valid: starts with 0x0A (field 1, wire type 2 = node)
        if (!TensorFlow.TFGraphDefParser.IsGraphDef(new byte[] { 0x0A, 0x05, 0x00, 0x00, 0x00 }))
            throw new Exception("Failed to detect valid GraphDef");
        if (TensorFlow.TFGraphDefParser.IsGraphDef(new byte[] { 0x47, 0x47 }))
            throw new Exception("False positive on non-GraphDef");
        Console.WriteLine("[TFGraphDef] Detection — PASS");
        await Task.CompletedTask;
    });

    // ═══════════════════════════════════════════════════════════
    //  PyTorch Loader
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task PyTorchLoader_ValidZip_ParsesDataFiles() => await RunTest(async accelerator =>
    {
        using var ms = new MemoryStream();
        using (var zip = new ZipArchive(ms, ZipArchiveMode.Create, leaveOpen: true))
        {
            var entry = zip.CreateEntry("archive/data/0");
            using var s = entry.Open();
            var floats = new float[] { 1f, 2f, 3f, 4f };
            var bytes = new byte[16];
            Buffer.BlockCopy(floats, 0, bytes, 0, 16);
            s.Write(bytes);
        }
        var data = ms.ToArray();
        var checkpoint = PyTorch.PyTorchLoader.Parse(data);
        if (checkpoint.DataFiles.Count == 0) throw new Exception("No data files found in ZIP");
        Console.WriteLine($"[PyTorch] DataFiles: {checkpoint.DataFiles.Count} — PASS");
        await Task.CompletedTask;
    });

    [TestMethod]
    public async Task PyTorchLoader_Detection_WorksCorrectly() => await RunTest(async accelerator =>
    {
        // ZIP magic: PK\x03\x04
        if (!PyTorch.PyTorchLoader.IsPyTorchCheckpoint(new byte[] { 0x50, 0x4B, 0x03, 0x04, 0x00, 0x00 }))
            throw new Exception("Failed to detect valid PyTorch ZIP");
        if (PyTorch.PyTorchLoader.IsPyTorchCheckpoint(new byte[] { 0x47, 0x47, 0x55, 0x46 }))
            throw new Exception("False positive: GGUF detected as PyTorch");
        Console.WriteLine("[PyTorch] Detection — PASS");
        await Task.CompletedTask;
    });

    // ═══════════════════════════════════════════════════════════
    //  SentencePiece Tokenizer
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task SentencePiece_EncodeAndDecode_RoundTrip() => await RunTest(async accelerator =>
    {
        // Minimal SentencePiece vocab: special tokens + a few words
        var tokens = new[] { "<unk>", "<s>", "</s>", "\u2581", "\u2581hello", "\u2581world", "h", "e", "l", "o" };
        var scores = new float[] { 0, 0, 0, -1, -2, -2, -3, -3, -3, -3 };
        var types = new int[] { 1, 2, 2, 0, 0, 0, 0, 0, 0, 0 }; // 1=unk, 2=control, 0=normal

        var sp = new SentencePieceTokenizer(tokens, scores, types);

        if (sp.VocabSize != 10) throw new Exception($"VocabSize: expected 10, got {sp.VocabSize}");
        if (sp.BosId != 1) throw new Exception($"BosId: expected 1, got {sp.BosId}");
        if (sp.EosId != 2) throw new Exception($"EosId: expected 2, got {sp.EosId}");

        // Encode "hello world"
        var encoded = sp.Encode("hello world");
        if (encoded.Length == 0) throw new Exception("Encode returned empty");

        // Decode back
        var decoded = sp.Decode(encoded);
        if (!decoded.Contains("hello")) throw new Exception($"Decode missing 'hello': got '{decoded}'");
        if (!decoded.Contains("world")) throw new Exception($"Decode missing 'world': got '{decoded}'");

        Console.WriteLine($"[SentencePiece] Encode: [{string.Join(",", encoded)}], Decode: '{decoded}' — PASS");
        await Task.CompletedTask;
    });

    [TestMethod]
    public async Task SentencePiece_ByteFallback_HandlesUnknownChars() => await RunTest(async accelerator =>
    {
        // Vocab with byte fallback tokens
        var tokens = new List<string> { "<unk>", "<s>", "</s>", "\u2581" };
        var scores = new List<float> { 0, 0, 0, -1 };
        var types = new List<int> { 1, 2, 2, 0 };
        // Add byte tokens <0x00> through <0xFF>
        for (int b = 0; b < 256; b++)
        {
            tokens.Add($"<0x{b:X2}>");
            scores.Add(-10);
            types.Add(5); // byte type
        }

        var sp = new SentencePieceTokenizer(tokens.ToArray(), scores.ToArray(), types.ToArray());
        // "A" should encode via byte fallback since no "A" token exists
        var encoded = sp.Encode("A");
        if (encoded.Length == 0) throw new Exception("Encode returned empty for 'A'");
        var decoded = sp.Decode(encoded);
        if (!decoded.Contains("A")) throw new Exception($"Byte fallback decode failed: got '{decoded}'");
        Console.WriteLine($"[SentencePiece] Byte fallback for 'A': ids=[{string.Join(",", encoded)}] — PASS");
        await Task.CompletedTask;
    });

    [TestMethod]
    public async Task SentencePiece_ToLoadedTokenizer_Works() => await RunTest(async accelerator =>
    {
        var tokens = new[] { "<unk>", "<s>", "</s>", "\u2581hello" };
        var scores = new float[] { 0, 0, 0, -1 };
        var types = new int[] { 1, 2, 2, 0 };
        var sp = new SentencePieceTokenizer(tokens, scores, types);
        var loaded = sp.ToLoadedTokenizer();

        if (loaded.Tokenizer is not SentencePieceTokenizer)
            throw new Exception($"Expected SentencePieceTokenizer, got {loaded.Tokenizer.GetType().Name}");
        if (loaded.BosTokenId != 1) throw new Exception($"BOS: expected 1, got {loaded.BosTokenId}");
        if (loaded.EosTokenId != 2) throw new Exception($"EOS: expected 2, got {loaded.EosTokenId}");

        var ids = loaded.Encode("hello");
        if (ids.Length == 0) throw new Exception("LoadedTokenizer.Encode returned empty");
        Console.WriteLine($"[SentencePiece] LoadedTokenizer integration — PASS");
        await Task.CompletedTask;
    });

    // ═══════════════════════════════════════════════════════════
    //  Format auto-detection (InferenceSession.DetectModelFormat)
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task FormatDetection_AllFormats_Correct() => await RunTest(async accelerator =>
    {
        // GGUF
        var gguf = new byte[] { 0x47, 0x47, 0x55, 0x46, 0x02, 0x00, 0x00, 0x00, 0x00 };
        if (InferenceSession.DetectModelFormat(gguf) != ModelFormat.GGUF)
            throw new Exception("GGUF detection failed");

        // SafeTensors
        var st = new byte[16];
        BitConverter.GetBytes((long)2).CopyTo(st, 0);
        st[8] = (byte)'{'; st[9] = (byte)'}';
        if (InferenceSession.DetectModelFormat(st) != ModelFormat.SafeTensors)
            throw new Exception("SafeTensors detection failed");

        // PyTorch ZIP
        var pt = new byte[] { 0x50, 0x4B, 0x03, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00 };
        if (InferenceSession.DetectModelFormat(pt) != ModelFormat.PyTorch)
            throw new Exception("PyTorch detection failed");

        // CoreML
        var cml = new byte[] { 0x08, 0x04, 0x12, 0x00 };
        if (InferenceSession.DetectModelFormat(cml) != ModelFormat.CoreML)
            throw new Exception("CoreML detection failed");

        Console.WriteLine("[FormatDetection] All formats detected correctly — PASS");
        await Task.CompletedTask;
    });
}
