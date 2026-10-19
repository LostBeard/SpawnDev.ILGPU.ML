namespace SpawnDev.ILGPU.ML.Animation;

/// <summary>
/// NVIDIA Kimodo's 77-joint SOMA skeleton, in the order Kimodo writes motion for it (kimodo.skeleton.definitions.SOMASkeleton77). SOMA is Y-up, X = the character's left,
/// Z = forward; a clip stores each joint's rotation AWAY from SOMA's standard T-pose, in world space.
/// </summary>
public static class SomaSkeleton
{
    public static readonly string[] Names =
    {
        "Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "HeadEnd", "Jaw", "LeftEye", "RightEye", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
        "LeftHandThumb1", "LeftHandThumb2", "LeftHandThumb3", "LeftHandThumbEnd",
        "LeftHandIndex1", "LeftHandIndex2", "LeftHandIndex3", "LeftHandIndex4", "LeftHandIndexEnd",
        "LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3", "LeftHandMiddle4", "LeftHandMiddleEnd",
        "LeftHandRing1", "LeftHandRing2", "LeftHandRing3", "LeftHandRing4", "LeftHandRingEnd",
        "LeftHandPinky1", "LeftHandPinky2", "LeftHandPinky3", "LeftHandPinky4", "LeftHandPinkyEnd",
        "RightShoulder", "RightArm", "RightForeArm", "RightHand",
        "RightHandThumb1", "RightHandThumb2", "RightHandThumb3", "RightHandThumbEnd",
        "RightHandIndex1", "RightHandIndex2", "RightHandIndex3", "RightHandIndex4", "RightHandIndexEnd",
        "RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3", "RightHandMiddle4", "RightHandMiddleEnd",
        "RightHandRing1", "RightHandRing2", "RightHandRing3", "RightHandRing4", "RightHandRingEnd",
        "RightHandPinky1", "RightHandPinky2", "RightHandPinky3", "RightHandPinky4", "RightHandPinkyEnd",
        "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase", "LeftToeEnd",
        "RightLeg", "RightShin", "RightFoot", "RightToeBase", "RightToeEnd",
    };

    public static readonly int[] Parents =
    {
        -1, 0, 1, 2, 3, 4, 5, 6, 6, 6, 6, 3, 11, 12, 13, 14, 15, 16, 17, 14, 19, 20, 21, 22, 14, 24, 25, 26, 27, 14, 29, 30, 31, 32, 14, 34, 35, 36, 37,
        3, 39, 40, 41, 42, 43, 44, 45, 42, 47, 48, 49, 50, 42, 52, 53, 54, 55, 42, 57, 58, 59, 60, 42, 62, 63, 64, 65, 0, 67, 68, 69, 70, 0, 72, 73, 74, 75,
    };

    public const int Count = 77;

    private static readonly Dictionary<string, int> Index = Names.Select((n, i) => (n, i)).ToDictionary(x => x.n, x => x.i);
    public static int IndexOf(string name) => Index[name];
}
