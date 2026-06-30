// Pixel diff between two 24-bit BMPs (the f64-vs-f32 conv A/B image quality check).
//   dotnet run bmpdiff.cs <a.bmp> <b.bmp>
// Reports max / mean absolute per-channel-byte difference and the % of bytes that differ at all.
if (args.Length < 2) { Console.Error.WriteLine("usage: bmpdiff <a.bmp> <b.bmp>"); return 2; }
var a = File.ReadAllBytes(args[0]);
var b = File.ReadAllBytes(args[1]);
int w = BitConverter.ToInt32(a, 18), h = BitConverter.ToInt32(a, 22);
int off = BitConverter.ToInt32(a, 10);          // pixel-data offset (54)
int n = w * 3 * h;
if (a.Length < off + n || b.Length < off + n) { Console.Error.WriteLine("size mismatch / truncated"); return 1; }
long sumAbs = 0; int maxAbs = 0, nDiff = 0;
for (int i = 0; i < n; i++)
{
    int d = Math.Abs(a[off + i] - b[off + i]);
    sumAbs += d; if (d > maxAbs) maxAbs = d; if (d > 0) nDiff++;
}
double mean = (double)sumAbs / n, pct = 100.0 * nDiff / n;
Console.WriteLine($"{w}x{h}  maxAbsDiff={maxAbs}/255  meanAbsDiff={mean:F4}  bytesDiffering={pct:F2}%");
// Verdict: an 8-bit image is perceptually identical if maxAbs is tiny; flag if a channel shifts a lot.
Console.WriteLine(maxAbs <= 4 ? "VERDICT: perceptually identical (f32 conv preserves SD quality)"
                : maxAbs <= 16 ? "VERDICT: negligible drift (a few LSBs)"
                : "VERDICT: VISIBLE difference — investigate");
return 0;
