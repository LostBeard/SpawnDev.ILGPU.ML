namespace SpawnDev.ILGPU.ML.Animation;

/// <summary>A double-precision 3-vector (the retargeting maths is done in double, like the reference; the output is rounded to float once).</summary>
public readonly record struct Vec3(double X, double Y, double Z)
{
    public static Vec3 operator +(Vec3 a, Vec3 b) => new(a.X + b.X, a.Y + b.Y, a.Z + b.Z);
    public static Vec3 operator -(Vec3 a, Vec3 b) => new(a.X - b.X, a.Y - b.Y, a.Z - b.Z);
    public static Vec3 operator *(Vec3 a, double s) => new(a.X * s, a.Y * s, a.Z * s);
    public static Vec3 operator /(Vec3 a, double s) => new(a.X / s, a.Y / s, a.Z / s);
    public static double Dot(Vec3 a, Vec3 b) => a.X * b.X + a.Y * b.Y + a.Z * b.Z;
    public static Vec3 Cross(Vec3 a, Vec3 b) => new(a.Y * b.Z - a.Z * b.Y, a.Z * b.X - a.X * b.Z, a.X * b.Y - a.Y * b.X);
    public double Length => Math.Sqrt(Dot(this, this));
    public Vec3 Normalized() => this / Length;
}

/// <summary>A 3x3 rotation matrix, row-major (M[r, c]): <c>Mul(v)</c> is M v, <c>A * B</c> applies B first.</summary>
public readonly struct Mat3
{
    private readonly double _m00, _m01, _m02, _m10, _m11, _m12, _m20, _m21, _m22;

    public Mat3(double m00, double m01, double m02, double m10, double m11, double m12, double m20, double m21, double m22)
    { _m00 = m00; _m01 = m01; _m02 = m02; _m10 = m10; _m11 = m11; _m12 = m12; _m20 = m20; _m21 = m21; _m22 = m22; }

    public static Mat3 Identity => new(1, 0, 0, 0, 1, 0, 0, 0, 1);

    /// <summary>The matrix whose COLUMNS are a, b, c.</summary>
    public static Mat3 FromColumns(Vec3 a, Vec3 b, Vec3 c) => new(a.X, b.X, c.X, a.Y, b.Y, c.Y, a.Z, b.Z, c.Z);

    public double this[int r, int c] => (r, c) switch
    {
        (0, 0) => _m00, (0, 1) => _m01, (0, 2) => _m02, (1, 0) => _m10, (1, 1) => _m11, (1, 2) => _m12, (2, 0) => _m20, (2, 1) => _m21, (2, 2) => _m22,
        _ => throw new IndexOutOfRangeException()
    };

    public Vec3 Column(int c) => new(this[0, c], this[1, c], this[2, c]);
    public Mat3 Transpose() => new(_m00, _m10, _m20, _m01, _m11, _m21, _m02, _m12, _m22);
    public Vec3 Mul(Vec3 v) => new(_m00 * v.X + _m01 * v.Y + _m02 * v.Z, _m10 * v.X + _m11 * v.Y + _m12 * v.Z, _m20 * v.X + _m21 * v.Y + _m22 * v.Z);

    public static Mat3 operator *(Mat3 a, Mat3 b) => new(
        a._m00 * b._m00 + a._m01 * b._m10 + a._m02 * b._m20, a._m00 * b._m01 + a._m01 * b._m11 + a._m02 * b._m21, a._m00 * b._m02 + a._m01 * b._m12 + a._m02 * b._m22,
        a._m10 * b._m00 + a._m11 * b._m10 + a._m12 * b._m20, a._m10 * b._m01 + a._m11 * b._m11 + a._m12 * b._m21, a._m10 * b._m02 + a._m11 * b._m12 + a._m12 * b._m22,
        a._m20 * b._m00 + a._m21 * b._m10 + a._m22 * b._m20, a._m20 * b._m01 + a._m21 * b._m11 + a._m22 * b._m21, a._m20 * b._m02 + a._m21 * b._m12 + a._m22 * b._m22);

    /// <summary>The smallest rotation turning direction <paramref name="a"/> onto direction <paramref name="b"/>.</summary>
    public static Mat3 Between(Vec3 a, Vec3 b)
    {
        a = a.Normalized(); b = b.Normalized();
        var v = Vec3.Cross(a, b); double c = Vec3.Dot(a, b);
        if (c < -1 + 1e-9)                                                       // opposite: half a turn about any perpendicular axis
        {
            var perp = Vec3.Cross(a, Math.Abs(a.X) < 0.9 ? new Vec3(1, 0, 0) : new Vec3(0, 1, 0)).Normalized();
            return new Mat3(2 * perp.X * perp.X - 1, 2 * perp.X * perp.Y, 2 * perp.X * perp.Z, 2 * perp.Y * perp.X, 2 * perp.Y * perp.Y - 1, 2 * perp.Y * perp.Z, 2 * perp.Z * perp.X, 2 * perp.Z * perp.Y, 2 * perp.Z * perp.Z - 1);
        }
        var vx = new Mat3(0, -v.Z, v.Y, v.Z, 0, -v.X, -v.Y, v.X, 0);
        var vx2 = vx * vx; double k = 1 / (1 + c);
        return new Mat3(1 + vx._m00 + vx2._m00 * k, vx._m01 + vx2._m01 * k, vx._m02 + vx2._m02 * k,
                        vx._m10 + vx2._m10 * k, 1 + vx._m11 + vx2._m11 * k, vx._m12 + vx2._m12 * k,
                        vx._m20 + vx2._m20 * k, vx._m21 + vx2._m21 * k, 1 + vx._m22 + vx2._m22 * k);
    }

    /// <summary>Unit quaternion (x, y, z, w), w >= 0, of this rotation.</summary>
    public (double X, double Y, double Z, double W) ToQuaternion()
    {
        double w = Math.Sqrt(Math.Max(0, 1 + _m00 + _m11 + _m22)) / 2, x = Math.Sqrt(Math.Max(0, 1 + _m00 - _m11 - _m22)) / 2,
               y = Math.Sqrt(Math.Max(0, 1 - _m00 + _m11 - _m22)) / 2, z = Math.Sqrt(Math.Max(0, 1 - _m00 - _m11 + _m22)) / 2;
        x = Math.CopySign(x, _m21 - _m12); y = Math.CopySign(y, _m02 - _m20); z = Math.CopySign(z, _m10 - _m01);
        double n = Math.Sqrt(x * x + y * y + z * z + w * w);
        return (x / n, y / n, z / n, w / n);
    }

    /// <summary>The rotation of a quaternion (x, y, z, w), normalised first.</summary>
    public static Mat3 FromQuaternion(double x, double y, double z, double w)
    {
        double n = Math.Sqrt(x * x + y * y + z * z + w * w); x /= n; y /= n; z /= n; w /= n;
        return new Mat3(1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y));
    }
}
