using ILGPU;
using ILGPU.Runtime;

// Kernels for WGSLDUMP repros (offline WGSL, no GPU): the smallest shape of a codegen question, for the backend's owner.
internal static class WgslRepro
{
    // an atomic slot allocator and plain stores into the SAME buffer (the first FaceKernels.DecodeImpl): on WebGPU the pipeline was invalid
    // ("Invalid ComputePipeline \"Kernel_DecodeImpl\"", no error raised by the launch; 2026-11-01)
    public static void MixedAtomicAndPlainStore(Index1D i, ArrayView1D<int, Stride1D.Dense> buf)
    {
        int slot = Atomic.Add(ref buf[0], 1);
        buf[1 + slot] = i;
    }
}
