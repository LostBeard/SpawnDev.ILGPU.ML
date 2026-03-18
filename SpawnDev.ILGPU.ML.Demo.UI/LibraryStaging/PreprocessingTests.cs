namespace SpawnDev.ILGPU.ML.Preprocessing.Tests;

/// <summary>
/// Self-contained validation tests for preprocessing utilities.
/// No xUnit dependency — can be called from anywhere (demo app, console, test runner).
/// Each method returns (passed, message). Call RunAll() for a full sweep.
/// </summary>
public static class PreprocessingTests
{
    public static List<(string Name, bool Passed, string Message)> RunAll()
    {
        var results = new List<(string, bool, string)>();
        results.Add(Test_ColorConversion_RGBToYCbCr_Roundtrip());
        results.Add(Test_ColorConversion_RGBToGrayscale());
        results.Add(Test_TensorLayout_NCHWToNHWC_Roundtrip());
        results.Add(Test_TensorLayout_InterleavedToPlanar_Roundtrip());
        results.Add(Test_ImageOps_CenterCrop());
        results.Add(Test_ImageOps_FlipHorizontal_DoubleFlip());
        results.Add(Test_ImageOps_Resize_Identity());
        results.Add(Test_ImageOps_PSNR_Identical());
        results.Add(Test_ImageOps_PSNR_Different());
        results.Add(Test_ImagePreprocessor_NCHW_Shape());
        results.Add(Test_ImagePreprocessor_Letterbox_Padding());
        results.Add(Test_ImageNetLabels_TopK());
        results.Add(Test_CocoLabels_Range());
        results.Add(Test_DepthColorMaps_Endpoints());
        results.Add(Test_YoloPostProcessor_NMS());
        results.Add(Test_PoseSkeleton_Decode());
        results.Add(Test_TextPreprocessor_CosineSimilarity());
        results.Add(Test_TextPreprocessor_Softmax());
        results.Add(Test_AudioPreprocessor_HannWindow());
        results.Add(Test_AudioPreprocessor_Resample());
        return results;
    }

    static (string, bool, string) Test_ColorConversion_RGBToYCbCr_Roundtrip()
    {
        var rgb = new byte[] { 255, 0, 0, 0, 255, 0, 0, 0, 255, 128, 128, 128 };
        var (y, cb, cr) = ColorConversion.RGBToYCbCr(rgb);
        var back = ColorConversion.YCbCrToRGB(y, cb, cr);

        int maxError = 0;
        for (int i = 0; i < rgb.Length; i++)
            maxError = Math.Max(maxError, Math.Abs(rgb[i] - back[i]));

        bool passed = maxError <= 2; // Allow small rounding error
        return ("RGBToYCbCr_Roundtrip", passed, $"maxError={maxError} (tolerance=2)");
    }

    static (string, bool, string) Test_ColorConversion_RGBToGrayscale()
    {
        var rgb = new byte[] { 255, 255, 255, 0, 0, 0 };
        var gray = ColorConversion.RGBToGrayscale(rgb);
        bool passed = gray[0] == 255 && gray[1] == 0;
        return ("RGBToGrayscale", passed, $"white={gray[0]} black={gray[1]}");
    }

    static (string, bool, string) Test_TensorLayout_NCHWToNHWC_Roundtrip()
    {
        var original = new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }; // [3, 2, 2]
        var nhwc = TensorLayout.NCHWToNHWC(original, 3, 2, 2);
        var back = TensorLayout.NHWCToNCHW(nhwc, 3, 2, 2);

        bool passed = original.SequenceEqual(back);
        return ("NCHWToNHWC_Roundtrip", passed, passed ? "exact match" : "MISMATCH");
    }

    static (string, bool, string) Test_TensorLayout_InterleavedToPlanar_Roundtrip()
    {
        var rgba = new byte[] { 100, 150, 200, 255, 50, 75, 100, 255 }; // 2 pixels
        var planar = TensorLayout.InterleavedRGBAToPlanarFloat(rgba, 2, 1);
        var back = TensorLayout.PlanarFloatToInterleavedRGBA(planar, 2, 1);

        int maxError = 0;
        for (int i = 0; i < 2; i++) // Check RGB only (alpha is always 255)
            for (int c = 0; c < 3; c++)
                maxError = Math.Max(maxError, Math.Abs(rgba[i * 4 + c] - back[i * 4 + c]));

        bool passed = maxError <= 1;
        return ("InterleavedToPlanar_Roundtrip", passed, $"maxError={maxError}");
    }

    static (string, bool, string) Test_ImageOps_CenterCrop()
    {
        // 4x4 image, center crop to 2x2
        var rgba = new byte[4 * 4 * 4];
        // Mark center pixels
        rgba[(1 * 4 + 1) * 4] = 42; // pixel (1,1)
        rgba[(2 * 4 + 2) * 4] = 99; // pixel (2,2)

        var cropped = ImageOps.CenterCrop(rgba, 4, 4, 2, 2);
        bool passed = cropped[0] == 42 && cropped[(1 * 2 + 1) * 4] == 99;
        return ("CenterCrop", passed, $"topLeft={cropped[0]} bottomRight={cropped[(1 * 2 + 1) * 4]}");
    }

    static (string, bool, string) Test_ImageOps_FlipHorizontal_DoubleFlip()
    {
        var original = new byte[3 * 2 * 4]; // 3x2
        for (int i = 0; i < original.Length; i++) original[i] = (byte)(i % 256);

        var flipped = ImageOps.FlipHorizontal(original, 3, 2);
        var back = ImageOps.FlipHorizontal(flipped, 3, 2);

        bool passed = original.SequenceEqual(back);
        return ("FlipHorizontal_DoubleFlip", passed, passed ? "identity" : "MISMATCH");
    }

    static (string, bool, string) Test_ImageOps_Resize_Identity()
    {
        // Resize to same dimensions should be near-identity
        var original = new byte[4 * 4 * 4];
        for (int i = 0; i < original.Length; i++) original[i] = (byte)(i % 256);

        var resized = ImageOps.Resize(original, 4, 4, 4, 4);

        int maxError = 0;
        for (int i = 0; i < original.Length; i++)
            maxError = Math.Max(maxError, Math.Abs(original[i] - resized[i]));

        bool passed = maxError <= 1; // Bilinear may have tiny rounding
        return ("Resize_Identity", passed, $"maxError={maxError}");
    }

    static (string, bool, string) Test_ImageOps_PSNR_Identical()
    {
        var img = new byte[10 * 10 * 4];
        for (int i = 0; i < img.Length; i++) img[i] = 128;

        float psnr = ImageOps.ComputePSNR(img, img);
        bool passed = float.IsPositiveInfinity(psnr);
        return ("PSNR_Identical", passed, $"psnr={psnr}");
    }

    static (string, bool, string) Test_ImageOps_PSNR_Different()
    {
        var imgA = new byte[10 * 10 * 4];
        var imgB = new byte[10 * 10 * 4];
        for (int i = 0; i < imgA.Length; i++) { imgA[i] = 100; imgB[i] = 200; }

        float psnr = ImageOps.ComputePSNR(imgA, imgB);
        bool passed = psnr > 0 && psnr < 100 && !float.IsInfinity(psnr);
        return ("PSNR_Different", passed, $"psnr={psnr:F2}dB");
    }

    static (string, bool, string) Test_ImagePreprocessor_NCHW_Shape()
    {
        var rgba = new byte[10 * 10 * 4]; // 10x10 RGBA
        var tensor = ImagePreprocessor.PreprocessToNCHW(rgba, 10, 10, 4, 4);

        bool passed = tensor.Length == 3 * 4 * 4; // [3, 4, 4] = 48
        return ("NCHW_Shape", passed, $"length={tensor.Length} expected=48");
    }

    static (string, bool, string) Test_ImagePreprocessor_Letterbox_Padding()
    {
        var rgba = new byte[100 * 50 * 4]; // 100x50 (2:1 aspect)
        var (tensor, info) = ImagePreprocessor.PreprocessLetterbox(rgba, 100, 50, 64, 64);

        bool passed = tensor.Length == 3 * 64 * 64 && info.PadY > 0 && info.PadX == 0;
        return ("Letterbox_Padding", passed, $"padX={info.PadX} padY={info.PadY} scale={info.Scale:F2}");
    }

    static (string, bool, string) Test_ImageNetLabels_TopK()
    {
        var logits = new float[1000];
        logits[281] = 10f; // tabby cat should be #1
        logits[282] = 5f;  // tiger cat should be #2

        var results = Data.ImageNetLabels.TopK(logits, 3);
        bool passed = results.Length == 3 && results[0].Probability > results[1].Probability;
        return ("ImageNetLabels_TopK", passed, $"top={results[0].Label} ({results[0].Probability:P1})");
    }

    static (string, bool, string) Test_CocoLabels_Range()
    {
        bool passed = Data.CocoLabels.Labels.Length == 80
            && Data.CocoLabels.GetLabel(0) == "person"
            && Data.CocoLabels.GetLabel(79) == "toothbrush"
            && Data.CocoLabels.GetLabel(999) == "class_999";
        return ("CocoLabels_Range", passed, $"count={Data.CocoLabels.Labels.Length}");
    }

    static (string, bool, string) Test_DepthColorMaps_Endpoints()
    {
        var (r0, g0, b0) = DepthColorMaps.GetColor("plasma", 0f);
        var (r1, g1, b1) = DepthColorMaps.GetColor("plasma", 1f);

        bool passed = r0 != r1 || g0 != g1 || b0 != b1; // Endpoints should differ
        return ("DepthColorMaps_Endpoints", passed, $"start=({r0},{g0},{b0}) end=({r1},{g1},{b1})");
    }

    static (string, bool, string) Test_YoloPostProcessor_NMS()
    {
        // Create fake YOLO output: [1, 84, 3] — 3 detections, 80 classes
        var output = new float[84 * 3];
        // Detection 0: box at (100,100,50,50), class 0 score 0.9
        output[0 * 3 + 0] = 100; output[1 * 3 + 0] = 100; output[2 * 3 + 0] = 50; output[3 * 3 + 0] = 50;
        output[4 * 3 + 0] = 0.9f;
        // Detection 1: overlapping box, class 0 score 0.7 (should be suppressed)
        output[0 * 3 + 1] = 105; output[1 * 3 + 1] = 105; output[2 * 3 + 1] = 50; output[3 * 3 + 1] = 50;
        output[4 * 3 + 1] = 0.7f;
        // Detection 2: far away, class 0 score 0.8 (should survive)
        output[0 * 3 + 2] = 500; output[1 * 3 + 2] = 500; output[2 * 3 + 2] = 50; output[3 * 3 + 2] = 50;
        output[4 * 3 + 2] = 0.8f;

        var results = YoloPostProcessor.Process(output, numClasses: 80, numDetections: 3, confThreshold: 0.5f);
        bool passed = results.Count == 2; // NMS should suppress detection 1
        return ("YoloPostProcessor_NMS", passed, $"detections={results.Count} (expected 2)");
    }

    static (string, bool, string) Test_PoseSkeleton_Decode()
    {
        var output = new float[17 * 3];
        output[0] = 0.5f; output[1] = 0.5f; output[2] = 0.9f; // nose at center, high confidence

        var keypoints = PoseSkeleton.DecodeMoveNetOutput(output, 640, 480);
        bool passed = keypoints.Length == 17
            && keypoints[0].Name == "nose"
            && Math.Abs(keypoints[0].X - 320) < 1
            && Math.Abs(keypoints[0].Y - 240) < 1;
        return ("PoseSkeleton_Decode", passed, $"nose=({keypoints[0].X:F0},{keypoints[0].Y:F0}) conf={keypoints[0].Confidence:F1}");
    }

    static (string, bool, string) Test_TextPreprocessor_CosineSimilarity()
    {
        var a = new float[] { 1, 0, 0 };
        var b = new float[] { 1, 0, 0 };
        var c = new float[] { 0, 1, 0 };

        float simSame = TextPreprocessor.CosineSimilarity(a, b);
        float simOrth = TextPreprocessor.CosineSimilarity(a, c);

        bool passed = Math.Abs(simSame - 1f) < 0.001f && Math.Abs(simOrth) < 0.001f;
        return ("CosineSimilarity", passed, $"same={simSame:F3} orthogonal={simOrth:F3}");
    }

    static (string, bool, string) Test_TextPreprocessor_Softmax()
    {
        var logits = new float[] { 1, 2, 3 };
        var probs = TextPreprocessor.Softmax(logits);

        float sum = probs.Sum();
        bool passed = Math.Abs(sum - 1f) < 0.001f && probs[2] > probs[1] && probs[1] > probs[0];
        return ("Softmax", passed, $"sum={sum:F4} ordering={probs[0]:F3}<{probs[1]:F3}<{probs[2]:F3}");
    }

    static (string, bool, string) Test_AudioPreprocessor_HannWindow()
    {
        var window = AudioPreprocessor.GenerateHannWindow(256);
        bool passed = window.Length == 256
            && Math.Abs(window[0]) < 0.001f      // Starts at ~0
            && Math.Abs(window[128] - 1f) < 0.01f // Peak at center
            && Math.Abs(window[255]) < 0.001f;    // Ends at ~0
        return ("HannWindow", passed, $"start={window[0]:F4} mid={window[128]:F4} end={window[255]:F4}");
    }

    static (string, bool, string) Test_AudioPreprocessor_Resample()
    {
        // 1 second at 44100Hz → 16000Hz
        var samples = new float[44100];
        for (int i = 0; i < samples.Length; i++)
            samples[i] = MathF.Sin(2 * MathF.PI * 440 * i / 44100f); // 440Hz sine

        var resampled = AudioPreprocessor.Resample(samples, 44100, 16000);
        bool passed = Math.Abs(resampled.Length - 16000) <= 1; // Should be ~16000 samples
        return ("Resample", passed, $"output={resampled.Length} samples (expected ~16000)");
    }
}
