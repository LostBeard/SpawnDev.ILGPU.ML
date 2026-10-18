namespace SpawnDev.ILGPU.ML.Hub;

/// <summary>
/// What <c>navigator.storage.estimate()</c> reported: the origin's quota and current usage, in bytes.
/// </summary>
/// <remarks>
/// <para>
/// ⚠️ <b>The quota is not always a limit.</b> MEASURED 2026-10-18 on stock Chrome 151 (Windows, no flags or policy): an origin with no
/// usage reports a quota of exactly 10 GiB, and the reported quota rises by exactly one byte for every byte written
/// (usage 0 / quota 10, then 1 / 11, 2 / 12 ... 14 / 24 GiB). A fresh profile on a different drive behaved the same, and all
/// 14 GiB of OPFS writes succeeded. So <c>quota - usage</c> is a sliding 10 GiB WINDOW, not the room the origin has left, and a caller
/// that treats it as a budget stores at most 10 GiB while the browser would have taken more. Chrome's documentation says an origin may
/// use up to 60% of the disk; a browser that reports honestly (older Chrome, other engines) still gives an exact headroom.
/// </para>
/// <para>
/// <see cref="HeadroomIsOpenEnded"/> tells the two apart: headroom that reached the window means the real limit is unknown, so plan
/// optimistically and let a refused write fall back (see <see cref="RangeStreamModelSource"/>).
/// </para>
/// </remarks>
public readonly record struct BrowserStorageEstimate(long Quota, long Usage)
{
    /// <summary>The headroom Chrome reports above current usage when the origin has more room than it will say (10 GiB, measured).</summary>
    public const long ReportedWindowBytes = 10L << 30;

    /// <summary>Default safety margin <see cref="PlanningBudget"/> keeps free when the headroom is exact.</summary>
    public const long DefaultMarginBytes = 300_000_000L;

    /// <summary>Reported quota minus usage (never negative).</summary>
    public long Headroom => Math.Max(0, Quota - Usage);

    /// <summary>
    /// True when <see cref="Headroom"/> reached the 10 GiB window Chrome reports above usage: the browser is not saying how much room
    /// is really left, so the headroom is a floor and not a limit.
    /// </summary>
    public bool HeadroomIsOpenEnded => Headroom >= ReportedWindowBytes;

    /// <summary>
    /// Bytes a caller may plan to add to browser storage: <see cref="Headroom"/> minus <paramref name="marginBytes"/> when it is exact;
    /// effectively unbounded when it is <see cref="HeadroomIsOpenEnded"/> (the write itself is then the arbiter).
    /// </summary>
    public long PlanningBudget(long marginBytes = DefaultMarginBytes)
        => HeadroomIsOpenEnded ? long.MaxValue / 2 : Math.Max(0, Headroom - marginBytes);
}
