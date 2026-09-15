using ILGPU.Runtime;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Which backends the heavy Kokoro tests are worth running on at all.
/// </summary>
public abstract partial class MLTestBase
{
    /// <summary>
    /// Refuses the ILGPU CPU accelerator for a heavy TTS test.
    /// </summary>
    /// <remarks>
    /// <para>
    /// 🔴 THE CPU ACCELERATOR IS A DEBUGGING DEVICE, NOT A SLOW GPU. From one of the original ILGPU
    /// authors, quoted by TJ 2026-09-15: <i>"The cpu accelerator is for debugging. It will not be faster
    /// than even a simple Parallel.For."</i> It exists so a kernel can be stepped through in a managed
    /// debugger, and this runner's own scheduler already records that it "saturates every core inside ONE
    /// process".
    /// </para>
    /// <para>
    /// So running Kokoro-82M end to end on it costs ~260 s PER PASS - the streaming test alone is four
    /// passes - to produce a number that describes nothing anybody ships. TJ: <i>"even if CPU passes, it
    /// is useless in the current state so not sure why that test is enabled for it right now."</i> He is
    /// right: a green CPU row here is not evidence, it is an invoice.
    /// </para>
    /// <para>
    /// ⚠️ WHAT THIS GIVES UP, STATED RATHER THAN HIDDEN: the CPU accelerator was a sixth independent
    /// check that these kernels compute the right waveform. That check is NOT gone - CUDA, OpenCL,
    /// WebGPU, WebGL and Wasm all still compare against the SAME onnxruntime fixture
    /// (<c>Pipeline_Kokoro_MatchesOnnxRuntimeWaveform</c>), and five independent implementations agreeing
    /// with an external oracle is what the correctness claim rests on. Dropping the slowest of six is a
    /// cost decision, not a coverage decision - and if a kernel ever needs debugging, the CPU accelerator
    /// is still there to step through, which is the job it is actually for.
    /// </para>
    /// <para>
    /// ⚠️ This is deliberately NOT a blanket "skip slow backends". WebGL and Wasm still run these: they
    /// are real targets a user's browser may hand us, so a regression there is a product regression. The
    /// CPU accelerator is not a target at all.
    /// </para>
    /// </remarks>
    private static void RequireShippableTtsBackend(Accelerator accelerator)
    {
        if (accelerator.AcceleratorType == AcceleratorType.CPU)
            throw new UnsupportedTestException(
                "the ILGPU CPU accelerator is a DEBUGGING device (\"it will not be faster than even a "
              + "simple Parallel.For\"), so running a 326 MB TTS model on it at ~260 s per pass measures "
              + "nothing that ships. Correctness is covered by the five real backends against the same "
              + "onnxruntime fixture.");
    }
}
