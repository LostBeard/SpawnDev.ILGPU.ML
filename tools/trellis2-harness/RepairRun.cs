using System.Diagnostics;
using System.Globalization;
using SpawnDev.ILGPU.ML.Trellis2;

namespace Trellis2Harness;

/// <summary>
/// MeshRepair on a real mesh: <c>repair in.obj out.obj [simplify N]</c>. The input is a stage dump from <c>generate</c> with
/// TRELLIS2_MESHSTAGES=1; the output is audited by reference/mesh_audit.py, which is an independent implementation of the same counts.
/// With "simplify N" the repaired mesh is also edge-collapsed to N faces and repaired again (the order the pipeline will use).
/// </summary>
internal static class RepairRun
{
    public static int Run(string inObj, string outObj, int simplifyTo)
    {
        var pos = new List<float>(); var idx = new List<int>();
        foreach (var line in File.ReadLines(inObj))
        {
            if (line.StartsWith("v ")) { var t = line.Split(' '); pos.Add(float.Parse(t[1], CultureInfo.InvariantCulture)); pos.Add(float.Parse(t[2], CultureInfo.InvariantCulture)); pos.Add(float.Parse(t[3], CultureInfo.InvariantCulture)); }
            else if (line.StartsWith("f ")) { var t = line.Split(' '); idx.Add(int.Parse(t[1].Split('/')[0]) - 1); idx.Add(int.Parse(t[2].Split('/')[0]) - 1); idx.Add(int.Parse(t[3].Split('/')[0]) - 1); }
        }
        var p = pos.ToArray(); var f = idx.ToArray();
        Console.WriteLine($"read {p.Length / 3} verts, {f.Length / 3} faces");
        var sw = Stopwatch.StartNew();
        var (rp, rf, rep) = MeshRepair.Repair(p, f);
        Console.WriteLine($"repair {sw.Elapsed.TotalSeconds:F2}s");
        Console.WriteLine($"  before {rep.Before}");
        Console.WriteLine($"  after  {rep.After}");
        Console.WriteLine($"  degenerate {rep.DegenerateFaces}, duplicate {rep.DuplicateFaces}, vertices split {rep.VerticesSplit}, fragments {rep.FragmentComponents} ({rep.FragmentFaces} faces), holes filled {rep.HolesFilled} (+{rep.HoleFacesAdded} faces), faces flipped {rep.FacesFlipped}, pieces turned outward {rep.PiecesTurnedOutward}, excess faces dropped {rep.ExcessFacesDropped}");
        if (simplifyTo > 0)
        {
            sw.Restart();
            (rp, rf) = MeshSimplifier.Simplify(rp, rf, simplifyTo);
            Console.WriteLine($"simplify -> {rf.Length / 3} faces {sw.Elapsed.TotalSeconds:F2}s");
            Console.WriteLine($"  simplified (before 2nd repair) {MeshRepair.Measure(rp, rf)}");
            sw.Restart();
            (rp, rf, rep) = MeshRepair.Repair(rp, rf);
            Console.WriteLine($"repair #2 {sw.Elapsed.TotalSeconds:F2}s: {rep.After}  holes filled {rep.HolesFilled}, split {rep.VerticesSplit}, flipped {rep.FacesFlipped}");
        }
        using var w = new StreamWriter(outObj);
        for (int i = 0; i < rp.Length; i += 3) w.WriteLine(string.Create(CultureInfo.InvariantCulture, $"v {rp[i]:R} {rp[i + 1]:R} {rp[i + 2]:R}"));
        for (int i = 0; i < rf.Length; i += 3) w.WriteLine($"f {rf[i] + 1} {rf[i + 1] + 1} {rf[i + 2] + 1}");
        Console.WriteLine($"wrote {outObj}");
        return 0;
    }
}
