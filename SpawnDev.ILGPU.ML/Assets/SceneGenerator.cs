using SpawnDev.ILGPU.ML.Trellis2;

namespace SpawnDev.ILGPU.ML.Assets;

/// <summary>What <see cref="SceneGenerator.BuildAsync"/> made: the layout, the scene GLB, and the GLB of each distinct object.</summary>
public sealed record GeneratedScene(SceneLayout Layout, byte[] Glb, IReadOnlyDictionary<string, byte[]> Assets);

/// <summary>
/// A scene from a layout: every distinct object of the <see cref="SceneLayout"/> is generated once with <see cref="AssetGenerator"/> (FLUX.2 or SD-Turbo picture, RMBG cut-out, TRELLIS.2 3D) and the
/// results are put in one glTF scene by <see cref="GltfSceneComposer"/>. The layout comes from <see cref="SceneLayoutPlanner"/> (a local LLM) or from the caller. The LLM is not loaded here:
/// run the planner first and free its model, the 3D stages want the memory.
/// </summary>
public sealed class SceneGenerator
{
    private readonly AssetGenerator _assets;
    public SceneGenerator(AssetGenerator assets) => _assets = assets;

    /// <summary>Called when an object starts and finishes: (what happened).</summary>
    public Action<string>? Progress { get; set; }

    /// <param name="layout">what stands where</param>
    /// <param name="seed">first picture seed; every object gets its own</param>
    /// <param name="pictures">pictures drawn per object; the one whose cut-out touches the frame least is built</param>
    /// <param name="trellis">3D options per object (default: 80k triangles and a 1024 px texture: a scene holds many objects)</param>
    /// <param name="onAsset">called with each object's GLB as soon as it is built (to save it, to show it)</param>
    public async Task<GeneratedScene> BuildAsync(SceneLayout layout, int seed = 1, int pictures = 2, ImageModel model = ImageModel.Flux2Klein, int size = 512,
        Trellis2Options? trellis = null, Func<string, byte[], Task>? onAsset = null)
    {
        var built = new Dictionary<string, byte[]>();
        int k = 0;
        foreach (var obj in layout.Objects)
        {
            k++;
            Progress?.Invoke($"object {k}/{layout.Objects.Count}: {obj.Name} ({obj.Instances.Count} instance{(obj.Instances.Count == 1 ? "" : "s")})");
            int objSeed = seed + k * 100;
            var o = trellis ?? new Trellis2Options { MaxFaces = 80_000, TextureSize = 1024 };
            var asset = await _assets.GenerateAsync(obj.Prompt, objSeed, pictures, model, size, new Trellis2Options { Seed = objSeed, MaxFaces = o.MaxFaces, TextureSize = o.TextureSize, Resolution = o.Resolution, RepairMesh = o.RepairMesh, Remesh = o.Remesh }, rig: false);
            if (asset.PromptMatch is float m) Progress?.Invoke($"  {obj.Name}: picture matches the prompt {m:F2}{(m < _assets.MinPromptMatch ? " (LOW: check this object)" : "")}");
            built[obj.Name] = asset.Glb;
            if (onAsset != null) await onAsset(obj.Name, asset.Glb);
        }
        Progress?.Invoke("composing the scene");
        return new GeneratedScene(layout, GltfSceneComposer.Compose(layout, built), built);
    }
}
