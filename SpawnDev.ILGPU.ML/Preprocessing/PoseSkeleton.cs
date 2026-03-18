namespace SpawnDev.ILGPU.ML.Preprocessing;

/// <summary>
/// COCO pose keypoint definitions and skeleton connections for MoveNet.
/// </summary>
public static class PoseSkeleton
{
    public static readonly string[] KeypointNames =
    {
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    };

    /// <summary>
    /// Skeleton bone connections as (from_index, to_index) pairs.
    /// </summary>
    public static readonly (int From, int To)[] Bones =
    {
        // Head
        (0, 1), (0, 2), (1, 3), (2, 4),
        // Shoulders
        (5, 6),
        // Arms
        (5, 7), (7, 9), (6, 8), (8, 10),
        // Torso
        (5, 11), (6, 12),
        // Hips
        (11, 12),
        // Legs
        (11, 13), (13, 15), (12, 14), (14, 16),
    };

    /// <summary>
    /// Decode MoveNet output [1, 1, 17, 3] to keypoint array.
    /// MoveNet outputs [y, x, confidence] (y first, normalized to [0,1]).
    /// </summary>
    public static Keypoint[] DecodeMoveNetOutput(float[] output, int imageWidth, int imageHeight)
    {
        var keypoints = new Keypoint[17];
        for (int i = 0; i < 17; i++)
        {
            // MoveNet output layout: [1, 1, 17, 3] flattened = offset i*3
            float y = output[i * 3 + 0]; // y comes first
            float x = output[i * 3 + 1];
            float conf = output[i * 3 + 2];

            keypoints[i] = new Keypoint
            {
                Name = KeypointNames[i],
                X = x * imageWidth,
                Y = y * imageHeight,
                Confidence = conf,
            };
        }
        return keypoints;
    }

    public class Keypoint
    {
        public string Name { get; set; } = "";
        public float X { get; set; }
        public float Y { get; set; }
        public float Confidence { get; set; }
    }
}
