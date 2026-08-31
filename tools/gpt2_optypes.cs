// Extract the distinct ONNX op_type set from a model file by targeted protobuf walking.
// ModelProto.graph = field 7 (message); GraphProto.node = field 1 (repeated NodeProto);
// NodeProto.op_type = field 4 (string). Everything else (initializers/weights) is skipped by length.
// Usage: dotnet run gpt2_optypes.cs -- <path-to-model.onnx> [manifest.txt]
using System.Buffers.Binary;

string path = args.Length > 0 ? args[0] : throw new ArgumentException("need model path");
using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20);

ulong ReadVarint(Stream s)
{
    ulong result = 0; int shift = 0;
    while (true)
    {
        int b = s.ReadByte();
        if (b < 0) throw new EndOfStreamException();
        result |= (ulong)(b & 0x7F) << shift;
        if ((b & 0x80) == 0) break;
        shift += 7;
    }
    return result;
}

// Read a single field header; returns (fieldNumber, wireType) or null at EOF/limit.
(int field, int wire)? ReadTag(Stream s, long limit)
{
    if (s.Position >= limit) return null;
    ulong tag = ReadVarint(s);
    return ((int)(tag >> 3), (int)(tag & 7));
}

void SkipField(Stream s, int wire)
{
    switch (wire)
    {
        case 0: ReadVarint(s); break;                                  // varint
        case 1: s.Seek(8, SeekOrigin.Current); break;                 // 64-bit
        case 2: { long len = (long)ReadVarint(s); s.Seek(len, SeekOrigin.Current); break; } // length-delimited
        case 5: s.Seek(4, SeekOrigin.Current); break;                 // 32-bit
        default: throw new InvalidDataException($"bad wire type {wire}");
    }
}

string ReadString(Stream s)
{
    long len = (long)ReadVarint(s);
    var buf = new byte[len];
    int read = 0;
    while (read < len) { int n = s.Read(buf, read, (int)(len - read)); if (n <= 0) throw new EndOfStreamException(); read += n; }
    return System.Text.Encoding.UTF8.GetString(buf);
}

var opTypes = new SortedSet<string>(StringComparer.Ordinal);

// Top level: ModelProto. Find graph (field 7).
long fileLen = fs.Length;
while (true)
{
    var tag = ReadTag(fs, fileLen);
    if (tag is null) break;
    var (field, wire) = tag.Value;
    if (field == 7 && wire == 2) // graph
    {
        long glen = (long)ReadVarint(fs);
        long gend = fs.Position + glen;
        // GraphProto: walk nodes (field 1).
        while (fs.Position < gend)
        {
            var gt = ReadTag(fs, gend);
            if (gt is null) break;
            var (gf, gw) = gt.Value;
            if (gf == 1 && gw == 2) // node (repeated NodeProto)
            {
                long nlen = (long)ReadVarint(fs);
                long nend = fs.Position + nlen;
                string? op = null;
                while (fs.Position < nend)
                {
                    var nt = ReadTag(fs, nend);
                    if (nt is null) break;
                    var (nf, nw) = nt.Value;
                    if (nf == 4 && nw == 2) op = ReadString(fs); // op_type
                    else SkipField(fs, nw);
                }
                if (op is not null) opTypes.Add(op);
            }
            else SkipField(fs, gw);
        }
        break; // done with graph
    }
    else SkipField(fs, wire);
}

Console.WriteLine($"DISTINCT_OPS={opTypes.Count}");
foreach (var op in opTypes) Console.WriteLine(op);

if (args.Length > 1 && File.Exists(args[1]))
{
    var manifest = new HashSet<string>(File.ReadAllLines(args[1]).Select(l => l.Trim()).Where(l => l.Length > 0), StringComparer.OrdinalIgnoreCase);
    var missing = opTypes.Where(o => !manifest.Contains(o)).ToArray();
    Console.WriteLine($"---\nMISSING_FROM_MANIFEST={missing.Length}");
    foreach (var m in missing) Console.WriteLine("MISSING: " + m);
}
