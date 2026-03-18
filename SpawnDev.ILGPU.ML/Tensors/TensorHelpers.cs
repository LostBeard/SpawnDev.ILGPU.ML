namespace SpawnDev.ILGPU.ML.Tensors;

/// <summary>
/// Static helpers for shape manipulation, broadcasting, and stride computation.
/// Follows ONNX/NumPy conventions.
/// </summary>
public static class TensorHelpers
{
    /// <summary>Product of all dimensions.</summary>
    public static int ElementCount(int[] shape)
    {
        int count = 1;
        for (int i = 0; i < shape.Length; i++)
            count *= shape[i];
        return count;
    }

    /// <summary>Compute row-major strides from shape. strides[i] = product of shape[i+1..].</summary>
    public static int[] ComputeStrides(int[] shape)
    {
        var strides = new int[shape.Length];
        if (shape.Length == 0) return strides;
        strides[shape.Length - 1] = 1;
        for (int i = shape.Length - 2; i >= 0; i--)
            strides[i] = strides[i + 1] * shape[i + 1];
        return strides;
    }

    /// <summary>
    /// Resolve a shape with one -1 dimension. The -1 dimension is inferred
    /// from totalElements and the other dimensions.
    /// </summary>
    public static int[] InferShape(int[] shape, int totalElements)
    {
        int inferIdx = -1;
        int known = 1;
        for (int i = 0; i < shape.Length; i++)
        {
            if (shape[i] == -1)
            {
                if (inferIdx >= 0) throw new ArgumentException("Only one dimension can be -1");
                inferIdx = i;
            }
            else
            {
                known *= shape[i];
            }
        }
        if (inferIdx < 0)
        {
            if (known != totalElements)
                throw new ArgumentException($"Shape product {known} != element count {totalElements}");
            return shape;
        }
        if (totalElements % known != 0)
            throw new ArgumentException($"Cannot infer dim: {totalElements} not divisible by {known}");
        var result = (int[])shape.Clone();
        result[inferIdx] = totalElements / known;
        return result;
    }

    /// <summary>Check if two shapes are identical.</summary>
    public static bool ShapesEqual(int[] a, int[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++)
            if (a[i] != b[i]) return false;
        return true;
    }

    /// <summary>
    /// Compute broadcast output shape following ONNX/NumPy rules.
    /// Shapes are right-aligned; each dim must be equal or one of them must be 1.
    /// </summary>
    public static int[] BroadcastShape(int[] a, int[] b)
    {
        int rank = Math.Max(a.Length, b.Length);
        var result = new int[rank];
        for (int i = 0; i < rank; i++)
        {
            int da = i < rank - a.Length ? 1 : a[i - (rank - a.Length)];
            int db = i < rank - b.Length ? 1 : b[i - (rank - b.Length)];
            if (da != db && da != 1 && db != 1)
                throw new ArgumentException($"Shapes [{string.Join(",", a)}] and [{string.Join(",", b)}] are not broadcastable at dim {i}");
            result[i] = Math.Max(da, db);
        }
        return result;
    }
}
