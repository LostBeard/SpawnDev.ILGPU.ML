namespace SpawnDev.ILGPU.ML.PyTorch;

/// <summary>
/// Minimal pickle protocol 2-5 reader for PyTorch tensor metadata.
/// Extracts tensor storage information: name, dtype, shape, storage key.
/// Does NOT execute arbitrary Python code — only reads data structures.
///
/// PyTorch pickle format for tensors:
/// - GLOBAL "torch._utils" "_rebuild_tensor_v2"
/// - BINPERSID (storage_key, dtype, size)
/// - tuple(storage, offset, shape, stride)
///
/// We extract: storage_key (maps to data/N file), dtype, shape.
/// </summary>
public static class PickleReader
{
    public record TensorMeta(string Name, string StorageKey, string DType, long[] Shape, long Offset);

    /// <summary>Parse pickle bytes and extract tensor metadata.</summary>
    public static List<TensorMeta> ReadTensors(byte[] pkl)
    {
        var tensors = new List<TensorMeta>();
        var stack = new List<object?>();
        var memo = new Dictionary<int, object?>();
        int pos = 0;
        string? currentKey = null;

        while (pos < pkl.Length)
        {
            byte op = pkl[pos++];
            switch (op)
            {
                case 0x80: // PROTO
                    pos++; // protocol version
                    break;
                case 0x28: // MARK
                    stack.Add("__MARK__");
                    break;
                case 0x29: // EMPTY_TUPLE
                    stack.Add(Array.Empty<object>());
                    break;
                case 0x85: // TUPLE1
                    if (stack.Count >= 1) { var a = stack[^1]; stack.RemoveAt(stack.Count - 1); stack.Add(new object?[] { a }); }
                    break;
                case 0x86: // TUPLE2
                    if (stack.Count >= 2) { var b = stack[^1]; var a2 = stack[^2]; stack.RemoveRange(stack.Count - 2, 2); stack.Add(new object?[] { a2, b }); }
                    break;
                case 0x87: // TUPLE3
                    if (stack.Count >= 3) { var c = stack[^1]; var b2 = stack[^2]; var a3 = stack[^3]; stack.RemoveRange(stack.Count - 3, 3); stack.Add(new object?[] { a3, b2, c }); }
                    break;
                case 0x5D: // EMPTY_LIST
                    stack.Add(new List<object?>());
                    break;
                case 0x7D: // EMPTY_DICT
                    stack.Add(new Dictionary<string, object?>());
                    break;
                case 0x4E: // NONE
                    stack.Add(null);
                    break;
                case 0x88: // NEWTRUE
                    stack.Add(true);
                    break;
                case 0x89: // NEWFALSE
                    stack.Add(false);
                    break;
                case 0x8C: // SHORT_BINUNICODE
                    { int len = pkl[pos++]; string s = System.Text.Encoding.UTF8.GetString(pkl, pos, len); pos += len; stack.Add(s); }
                    break;
                case 0x58: // BINUNICODE
                    { int len = BitConverter.ToInt32(pkl, pos); pos += 4; string s = System.Text.Encoding.UTF8.GetString(pkl, pos, len); pos += len; stack.Add(s); }
                    break;
                case 0x8A: // LONG1
                    { int len = pkl[pos++]; long val = 0; for (int i = 0; i < len; i++) val |= (long)pkl[pos + i] << (i * 8); pos += len; stack.Add(val); }
                    break;
                case 0x8B: // LONG4
                    { int len = BitConverter.ToInt32(pkl, pos); pos += 4; long val = 0; for (int i = 0; i < Math.Min(len, 8); i++) val |= (long)pkl[pos + i] << (i * 8); pos += len; stack.Add(val); }
                    break;
                case 0x4A: // BININT
                    { int val = BitConverter.ToInt32(pkl, pos); pos += 4; stack.Add((long)val); }
                    break;
                case 0x4B: // BININT1
                    stack.Add((long)pkl[pos++]);
                    break;
                case 0x4D: // BININT2
                    { int val = pkl[pos] | (pkl[pos + 1] << 8); pos += 2; stack.Add((long)val); }
                    break;
                case 0x47: // BINFLOAT
                    { double val = BitConverter.ToDouble(pkl, pos); pos += 8; stack.Add(val); }
                    break;
                case 0x71: // BINPUT
                    { int idx = pkl[pos++]; if (stack.Count > 0) memo[idx] = stack[^1]; }
                    break;
                case 0x72: // LONG_BINPUT
                    { int idx = BitConverter.ToInt32(pkl, pos); pos += 4; if (stack.Count > 0) memo[idx] = stack[^1]; }
                    break;
                case 0x68: // BINGET
                    { int idx = pkl[pos++]; stack.Add(memo.GetValueOrDefault(idx)); }
                    break;
                case 0x6A: // LONG_BINGET
                    { int idx = BitConverter.ToInt32(pkl, pos); pos += 4; stack.Add(memo.GetValueOrDefault(idx)); }
                    break;
                case 0x73: // SETITEM
                    if (stack.Count >= 3 && stack[^3] is Dictionary<string, object?> d)
                    {
                        var val = stack[^1]; var key = stack[^2];
                        stack.RemoveRange(stack.Count - 2, 2);
                        if (key is string ks) d[ks] = val;
                    }
                    break;
                case 0x75: // SETITEMS
                    {
                        int mark = stack.LastIndexOf("__MARK__");
                        if (mark >= 0 && mark > 0 && stack[mark - 1] is Dictionary<string, object?> dict)
                        {
                            for (int i = mark + 1; i < stack.Count - 1; i += 2)
                                if (stack[i] is string k) dict[k] = stack[i + 1];
                            stack.RemoveRange(mark, stack.Count - mark);
                        }
                    }
                    break;
                case 0x74: // TUPLE (from mark)
                    {
                        int mark = stack.LastIndexOf("__MARK__");
                        if (mark >= 0)
                        {
                            var items = stack.GetRange(mark + 1, stack.Count - mark - 1).ToArray();
                            stack.RemoveRange(mark, stack.Count - mark);
                            stack.Add(items);
                        }
                    }
                    break;
                case 0x5B: // APPENDS
                    {
                        int mark = stack.LastIndexOf("__MARK__");
                        if (mark >= 0 && mark > 0 && stack[mark - 1] is List<object?> list)
                        {
                            for (int i = mark + 1; i < stack.Count; i++) list.Add(stack[i]);
                            stack.RemoveRange(mark, stack.Count - mark);
                        }
                    }
                    break;
                case 0x63: // GLOBAL
                    {
                        int nl1 = Array.IndexOf(pkl, (byte)'\n', pos); string mod = System.Text.Encoding.ASCII.GetString(pkl, pos, nl1 - pos); pos = nl1 + 1;
                        int nl2 = Array.IndexOf(pkl, (byte)'\n', pos); string name = System.Text.Encoding.ASCII.GetString(pkl, pos, nl2 - pos); pos = nl2 + 1;
                        stack.Add($"__GLOBAL__{mod}.{name}");
                    }
                    break;
                case 0x93: // STACK_GLOBAL
                    if (stack.Count >= 2)
                    {
                        var n = stack[^1]; var m = stack[^2]; stack.RemoveRange(stack.Count - 2, 2);
                        stack.Add($"__GLOBAL__{m}.{n}");
                    }
                    break;
                case 0x52: // REDUCE
                    if (stack.Count >= 2) { stack.RemoveAt(stack.Count - 2); } // callable(args) → result (keep args as result)
                    break;
                case 0x81: // NEWOBJ
                    if (stack.Count >= 2) { stack.RemoveAt(stack.Count - 2); }
                    break;
                case 0x62: // BUILD
                    if (stack.Count >= 2)
                    {
                        var state = stack[^1]; stack.RemoveAt(stack.Count - 1);
                        // If building an OrderedDict, extract key-value pairs as tensor names
                        if (state is Dictionary<string, object?> sd)
                            foreach (var kv in sd)
                                currentKey = kv.Key; // Track most recent key for tensor association
                    }
                    break;
                case 0x51: // BINPERSID
                    // Persistent ID — used by PyTorch for storage references
                    // The top of stack should be (storage_type, storage_key, device, num_elements)
                    if (stack.Count > 0 && stack[^1] is object?[] pid && pid.Length >= 4)
                    {
                        string storageKey = pid[1]?.ToString() ?? "";
                        string dtype = pid[2]?.ToString() ?? "float32";
                        stack[^1] = $"__STORAGE__{storageKey}__{dtype}";
                    }
                    break;
                case 0x2E: // STOP
                    goto done;
                default:
                    // Unknown opcode — skip (may cause issues for complex pickles)
                    break;
            }
        }
        done:

        // Extract tensor metadata from the final state
        // PyTorch OrderedDict maps tensor names → (storage, offset, shape, stride) tuples
        if (stack.Count > 0 && stack[^1] is Dictionary<string, object?> stateDict)
        {
            foreach (var (name, value) in stateDict)
            {
                if (value is object?[] tuple && tuple.Length >= 3)
                {
                    string storage = tuple[0]?.ToString() ?? "";
                    long offset = tuple[1] is long lo ? lo : 0;
                    long[] shape = Array.Empty<long>();
                    if (tuple[2] is object?[] shapeArr)
                        shape = shapeArr.Select(s => s is long l ? l : 0).ToArray();

                    string storageKey = "";
                    string dtype = "float32";
                    if (storage.StartsWith("__STORAGE__"))
                    {
                        var parts = storage.Split("__", StringSplitOptions.RemoveEmptyEntries);
                        if (parts.Length >= 2) storageKey = parts[1];
                        if (parts.Length >= 3) dtype = parts[2];
                    }

                    tensors.Add(new TensorMeta(name, storageKey, dtype, shape, offset));
                }
            }
        }

        return tensors;
    }
}
