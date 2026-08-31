// Dump ONNX node sequence: idx | op_type | inputs -> outputs
// ModelProto.graph=7(msg); GraphProto.node=1(repeated NodeProto)
// NodeProto: input=1(str,repeated), output=2(str,repeated), name=3(str), op_type=4(str)
// Usage: dotnet run onnx_nodewalk.cs -- <model.onnx> [focusOpType]
using System.Text;

string path = args.Length > 0 ? args[0] : throw new ArgumentException("need model path");
string focus = args.Length > 1 ? args[1] : "Where";

byte[] bytes = File.ReadAllBytes(path);
int pos = 0;
long ReadVarint(byte[] b, ref int p)
{
    long result = 0; int shift = 0;
    while (true) { byte x = b[p++]; result |= (long)(x & 0x7F) << shift; if ((x & 0x80) == 0) break; shift += 7; }
    return result;
}
(int field, int wire) ReadTag(byte[] b, ref int p) { long t = ReadVarint(b, ref p); return ((int)(t >> 3), (int)(t & 7)); }
void Skip(byte[] b, ref int p, int wire)
{
    if (wire == 0) ReadVarint(b, ref p);
    else if (wire == 2) { long len = ReadVarint(b, ref p); p += (int)len; }
    else if (wire == 5) p += 4;
    else if (wire == 1) p += 8;
    else throw new Exception($"wire {wire}");
}

// Find graph (field 7) in ModelProto
byte[] graphBytes = null;
while (pos < bytes.Length)
{
    var (f, w) = ReadTag(bytes, ref pos);
    if (f == 7 && w == 2) { long len = ReadVarint(bytes, ref pos); graphBytes = new byte[len]; Array.Copy(bytes, pos, graphBytes, 0, (int)len); pos += (int)len; break; }
    else Skip(bytes, ref pos, w);
}
if (graphBytes == null) { Console.WriteLine("no graph"); return; }

// Walk GraphProto.node (field 1)
int gp = 0; int idx = 0;
var lines = new List<string>();
var focusIdx = new List<int>();
while (gp < graphBytes.Length)
{
    var (f, w) = ReadTag(graphBytes, ref gp);
    if (f == 1 && w == 2)
    {
        long len = ReadVarint(graphBytes, ref gp);
        int nodeEnd = gp + (int)len;
        var inputs = new List<string>(); var outputs = new List<string>(); string opType = "?"; string name = "";
        while (gp < nodeEnd)
        {
            var (nf, nw) = ReadTag(graphBytes, ref gp);
            if (nf == 1 && nw == 2) { long l = ReadVarint(graphBytes, ref gp); inputs.Add(Encoding.UTF8.GetString(graphBytes, gp, (int)l)); gp += (int)l; }
            else if (nf == 2 && nw == 2) { long l = ReadVarint(graphBytes, ref gp); outputs.Add(Encoding.UTF8.GetString(graphBytes, gp, (int)l)); gp += (int)l; }
            else if (nf == 3 && nw == 2) { long l = ReadVarint(graphBytes, ref gp); name = Encoding.UTF8.GetString(graphBytes, gp, (int)l); gp += (int)l; }
            else if (nf == 4 && nw == 2) { long l = ReadVarint(graphBytes, ref gp); opType = Encoding.UTF8.GetString(graphBytes, gp, (int)l); gp += (int)l; }
            else Skip(graphBytes, ref gp, nw);
        }
        gp = nodeEnd;
        string inStr = string.Join(",", inputs);
        string outStr = string.Join(",", outputs);
        lines.Add($"{idx,4} | {opType,-18} | in:[{inStr}] -> out:[{outStr}]");
        if (opType == focus) focusIdx.Add(idx);
        idx++;
    }
    else Skip(graphBytes, ref gp, w);
}

Console.WriteLine($"TOTAL_NODES={idx}");
Console.WriteLine($"=== {focus} nodes: {focusIdx.Count} at indices [{string.Join(",", focusIdx)}] ===");
// Print each focus node + 3 nodes of context before/after
foreach (var fi in focusIdx.Take(4))
{
    Console.WriteLine($"--- context around {focus} #{fi} ---");
    for (int i = Math.Max(0, fi - 4); i <= Math.Min(idx - 1, fi + 2); i++)
        Console.WriteLine(lines[i]);
}
// Also dump first 60 nodes (embedding + first attention block)
Console.WriteLine("=== first 60 nodes ===");
for (int i = 0; i < Math.Min(60, idx); i++) Console.WriteLine(lines[i]);
