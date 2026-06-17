namespace SpawnDev.ILGPU.ML.Tiling;

/// <summary>
/// A [C, H, W] feature map split into a rows×cols grid of spatial tiles, each carried as a CPU float[] with a
/// persistent <see cref="Halo"/>-pixel margin on every side (filled from neighbor tiles' cores, zero at the image
/// boundary). The exact-stat tiled VAE decoder keeps only ONE tile on the GPU at a time (this CPU backing is the
/// offload that bounds GPU peak); 3×3 SAME convs read the padded tile (core+halo) and write the core, so the halo
/// must be refreshed from neighbors (<see cref="RefreshHalos"/>) before each conv (the conv consumes the halo and
/// leaves it stale). Layout per tile: channel-major, [C, coreH+2H, coreW+2H], row-major within a channel.
///
/// Plan: Plans/exact-tiled-vae-decode-2026-06-16.md (step 2).
/// </summary>
public sealed class TiledFeatureMap
{
    public int Channels { get; private set; }
    public int Height { get; private set; }      // full (recombined) core height
    public int Width { get; private set; }
    public int Halo { get; }
    public int Rows { get; }
    public int Cols { get; }

    // Per-tile core extent (edge tiles may be larger/smaller). [r,c] → core rows/cols + the start row/col in full.
    private readonly int[] _coreH, _coreW, _y0, _x0;
    private readonly float[][] _tiles;            // [r*Cols+c] → padded tile data, length C*(coreH+2H)*(coreW+2H)

    private TiledFeatureMap(int c, int h, int w, int halo, int rows, int cols)
    {
        Channels = c; Height = h; Width = w; Halo = halo; Rows = rows; Cols = cols;
        _coreH = new int[rows]; _coreW = new int[cols]; _y0 = new int[rows]; _x0 = new int[cols];
        // Even-ish split (remainder spread over the first tiles), like the latent-tile grid.
        int y = 0;
        for (int r = 0; r < rows; r++) { _coreH[r] = h / rows + (r < h % rows ? 1 : 0); _y0[r] = y; y += _coreH[r]; }
        int x = 0;
        for (int cc = 0; cc < cols; cc++) { _coreW[cc] = w / cols + (cc < w % cols ? 1 : 0); _x0[cc] = x; x += _coreW[cc]; }
        _tiles = new float[rows * cols][];
        for (int r = 0; r < rows; r++)
            for (int cc = 0; cc < cols; cc++)
                _tiles[r * cols + cc] = new float[(long)c * (_coreH[r] + 2 * halo) * (_coreW[cc] + 2 * halo) is var n && n <= int.MaxValue ? (int)n : throw new OverflowException()];
    }

    // Explicit per-row/col core sizes (used after a 2× upsample, where tile bands must be EXACTLY 2× the source
    // bands so the grid stays aligned — an even re-split of 2H/rows would shift the boundaries).
    private TiledFeatureMap(int channels, int halo, int[] coreH, int[] coreW)
    {
        Channels = channels; Halo = halo; Rows = coreH.Length; Cols = coreW.Length;
        _coreH = coreH; _coreW = coreW; _y0 = new int[Rows]; _x0 = new int[Cols];
        int y = 0; for (int r = 0; r < Rows; r++) { _y0[r] = y; y += coreH[r]; } Height = y;
        int x = 0; for (int c = 0; c < Cols; c++) { _x0[c] = x; x += coreW[c]; } Width = x;
        _tiles = new float[Rows * Cols][];
        for (int r = 0; r < Rows; r++)
            for (int c = 0; c < Cols; c++)
                _tiles[r * Cols + c] = new float[(long)channels * (coreH[r] + 2 * halo) * (coreW[c] + 2 * halo) is var n && n <= int.MaxValue ? (int)n : throw new OverflowException()];
    }

    /// <summary>Allocate an empty grid with EXPLICIT per-row/col core sizes (cores+halos zeroed).</summary>
    public static TiledFeatureMap AllocateExplicit(int channels, int[] coreH, int[] coreW, int halo)
        => new TiledFeatureMap(channels, halo, coreH, coreW);

    public int CoreH(int r) => _coreH[r];
    public int CoreW(int c) => _coreW[c];
    /// <summary>The padded (core + 2*Halo) data buffer for tile (r,c). Indexed [ch, y, x] with
    /// y,x in [0, core+2*Halo); the core occupies [Halo, Halo+core).</summary>
    public float[] Tile(int r, int c) => _tiles[r * Cols + c];
    public int PaddedH(int r) => _coreH[r] + 2 * Halo;
    public int PaddedW(int c) => _coreW[c] + 2 * Halo;

    /// <summary>Allocate an empty grid (cores zeroed, halos zeroed) for the given dims — used to receive the
    /// per-tile CORE outputs of an op via <see cref="WriteCore"/>, after which <see cref="RefreshHalos"/> fills
    /// the halos from neighbors. The channel count is the OP's output channels (may differ from the input map).</summary>
    public static TiledFeatureMap Allocate(int channels, int height, int width, int rows, int cols, int halo)
        => new TiledFeatureMap(channels, height, width, halo, rows, cols);

    /// <summary>Write a tile's CORE from a packed [channels, coreH, coreW] buffer (the op's output for that
    /// tile). Leaves the halo margin untouched (refresh it afterward). Mirrors <see cref="Allocate"/>'s channels.</summary>
    public void WriteCore(int r, int c, float[] core)
    {
        var tile = _tiles[r * Cols + c];
        int ph = PaddedH(r), pw = PaddedW(c), ch0 = _coreH[r], cw0 = _coreW[c];
        for (int ch = 0; ch < Channels; ch++)
            for (int yy = 0; yy < ch0; yy++)
                for (int xx = 0; xx < cw0; xx++)
                    tile[(long)ch * ph * pw + (yy + Halo) * pw + (xx + Halo)] = core[(long)ch * ch0 * cw0 + yy * cw0 + xx];
    }

    /// <summary>Build from a full [C,H,W] tensor: each tile's core is copied from its region; halos are filled
    /// directly from the neighbor regions in the FULL tensor (zero past the image boundary).</summary>
    public static TiledFeatureMap FromFull(float[] full, int channels, int height, int width, int rows, int cols, int halo)
    {
        var t = new TiledFeatureMap(channels, height, width, halo, rows, cols);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
            {
                var tile = t._tiles[r * cols + c];
                int ph = t.PaddedH(r), pw = t.PaddedW(c);
                int gy0 = t._y0[r] - halo, gx0 = t._x0[c] - halo;   // top-left of the padded tile in full coords
                for (int ch = 0; ch < channels; ch++)
                {
                    long cBaseTile = (long)ch * ph * pw, cBaseFull = (long)ch * height * width;
                    for (int yy = 0; yy < ph; yy++)
                    {
                        int gy = gy0 + yy;
                        for (int xx = 0; xx < pw; xx++)
                        {
                            int gx = gx0 + xx;
                            tile[cBaseTile + yy * pw + xx] =
                                (gy >= 0 && gy < height && gx >= 0 && gx < width) ? full[cBaseFull + gy * width + gx] : 0f;
                        }
                    }
                }
            }
        return t;
    }

    /// <summary>Refill every tile's halo margin from its neighbors' CORE pixels (zero past the image boundary).
    /// Call before each 3×3 conv: the conv read+consumed the previous halo, so it's now stale. Reads from the
    /// CORE regions only (the authoritative data), so it is exact regardless of how the cores were produced.</summary>
    public void RefreshHalos()
    {
        for (int r = 0; r < Rows; r++)
            for (int c = 0; c < Cols; c++)
            {
                var tile = _tiles[r * Cols + c];
                int ph = PaddedH(r), pw = PaddedW(c), ch0 = _coreH[r], cw0 = _coreW[c];
                int gy0 = _y0[r] - Halo, gx0 = _x0[c] - Halo;
                for (int ch = 0; ch < Channels; ch++)
                {
                    long cBase = (long)ch * ph * pw;
                    for (int yy = 0; yy < ph; yy++)
                    {
                        bool yCore = yy >= Halo && yy < Halo + ch0;
                        for (int xx = 0; xx < pw; xx++)
                        {
                            bool xCore = xx >= Halo && xx < Halo + cw0;
                            if (yCore && xCore) continue;                 // core stays as-is
                            int gy = gy0 + yy, gx = gx0 + xx;             // full-image coordinate of this halo px
                            tile[cBase + yy * pw + xx] = SampleCore(ch, gy, gx);
                        }
                    }
                }
            }
    }

    /// <summary>Read core pixel (ch) at full-image coordinate (gy,gx) from whichever tile owns it; 0 outside.</summary>
    private float SampleCore(int ch, int gy, int gx)
    {
        if (gy < 0 || gy >= Height || gx < 0 || gx >= Width) return 0f;
        int r = 0; while (r < Rows - 1 && gy >= _y0[r] + _coreH[r]) r++;
        int c = 0; while (c < Cols - 1 && gx >= _x0[c] + _coreW[c]) c++;
        var tile = _tiles[r * Cols + c];
        int pw = PaddedW(c);
        int ly = gy - _y0[r] + Halo, lx = gx - _x0[c] + Halo;
        return tile[(long)ch * PaddedH(r) * pw + ly * pw + lx];
    }

    /// <summary>Read tile (r,c)'s CORE into a packed [C, coreH, coreW] buffer (halo dropped).</summary>
    public float[] ReadCore(int r, int c)
    {
        var tile = _tiles[r * Cols + c];
        int ph = PaddedH(r), pw = PaddedW(c), ch0 = _coreH[r], cw0 = _coreW[c];
        var core = new float[(long)Channels * ch0 * cw0 is var n && n <= int.MaxValue ? (int)n : throw new OverflowException()];
        for (int ch = 0; ch < Channels; ch++)
            for (int yy = 0; yy < ch0; yy++)
                for (int xx = 0; xx < cw0; xx++)
                    core[(long)ch * ch0 * cw0 + yy * cw0 + xx] = tile[(long)ch * ph * pw + (yy + Halo) * pw + (xx + Halo)];
        return core;
    }

    /// <summary>Recombine the tile CORES into a full [C,H,W] tensor (halos discarded).</summary>
    public float[] ToFull()
    {
        var full = new float[(long)Channels * Height * Width is var n && n <= int.MaxValue ? (int)n : throw new OverflowException()];
        for (int r = 0; r < Rows; r++)
            for (int c = 0; c < Cols; c++)
            {
                var tile = _tiles[r * Cols + c];
                int ph = PaddedH(r), pw = PaddedW(c), ch0 = _coreH[r], cw0 = _coreW[c];
                for (int ch = 0; ch < Channels; ch++)
                {
                    long cBaseTile = (long)ch * ph * pw, cBaseFull = (long)ch * Height * Width;
                    for (int yy = 0; yy < ch0; yy++)
                        for (int xx = 0; xx < cw0; xx++)
                            full[cBaseFull + (_y0[r] + yy) * Width + (_x0[c] + xx)] =
                                tile[cBaseTile + (yy + Halo) * pw + (xx + Halo)];
                }
            }
        return full;
    }
}
