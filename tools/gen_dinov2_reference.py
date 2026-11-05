"""
DINOv2 image-embedding references (Xenova/dinov2-small and -base ONNX = facebook/dinov2-*, Apache-2.0) for
MLTestBase.Dinov2Tests, written to SpawnDev.ILGPU.ML.Demo/wwwroot/references/dinov2/.

Per model and per sample image (samples/street_rgba.bin, samples/cat_rgba.bin: int32 w, int32 h, RGBA bytes):
  {model}_{img}_pixels.bin   float32 [3, 224, 224]  Hugging Face BitImageProcessor preprocessing (PIL bicubic resize of the
                                                    shortest edge to 256, center crop 224, /255, ImageNet mean/std)
  {model}_{img}_tokens.bin   float32 [257, D]       onnxruntime last_hidden_state (CLS, then 16 x 16 patch tokens; post-layernorm)
  {model}_{img}_pooled.bin   float32 [2 D]          L2norm(concat(L2norm(CLS), L2norm(mean of patch tokens)))
The pooled descriptor is the DINOv2 paper's linear-evaluation feature (class token + average-pooled patch tokens), each part
L2-normalised so neither dominates, then the whole normalised for cosine retrieval.

Run: python tools/gen_dinov2_reference.py <dir with dinov2-small/model.onnx and dinov2-base/model.onnx>
"""
import os, struct, sys
import numpy as np
import onnxruntime as ort
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
WWW = os.path.join(HERE, "..", "SpawnDev.ILGPU.ML.Demo", "wwwroot")
OUT = os.path.join(WWW, "references", "dinov2")
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def load_rgba(path):
    b = open(path, "rb").read()
    w, h = struct.unpack("<ii", b[:8])
    a = np.frombuffer(b[8:8 + w * h * 4], np.uint8).reshape(h, w, 4)
    return Image.fromarray(a[:, :, :3].copy(), "RGB")


def preprocess(img):
    # transformers BitImageProcessor: resize shortest_edge 256 (PIL BICUBIC), center_crop 224, rescale 1/255, normalize
    w, h = img.size
    if w <= h:
        nw, nh = 256, int(256 * h / w)
    else:
        nw, nh = int(256 * w / h), 256
    img = img.resize((nw, nh), Image.BICUBIC)
    top, left = (nh - 224) // 2, (nw - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))
    x = np.asarray(img, np.float32) / 255.0
    x = (x - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)).astype(np.float32)


def l2n(v):
    return v / max(np.linalg.norm(v), 1e-12)


def main():
    models_dir = sys.argv[1] if len(sys.argv) > 1 else r"D:\users\tj\Projects\_Models"
    os.makedirs(OUT, exist_ok=True)
    for model in ("small", "base"):
        sess = ort.InferenceSession(os.path.join(models_dir, f"dinov2-{model}", "model.onnx"))
        for name in ("street", "cat"):
            px = preprocess(load_rgba(os.path.join(WWW, "samples", f"{name}_rgba.bin")))
            tok = sess.run(None, {"pixel_values": px[None]})[0][0].astype(np.float32)
            pooled = l2n(np.concatenate([l2n(tok[0]), l2n(tok[1:].mean(axis=0))])).astype(np.float32)
            px.tofile(os.path.join(OUT, f"{model}_{name}_pixels.bin"))
            tok.tofile(os.path.join(OUT, f"{model}_{name}_tokens.bin"))
            pooled.tofile(os.path.join(OUT, f"{model}_{name}_pooled.bin"))
            print(f"{model} {name}: tokens {tok.shape}, |cls| {np.linalg.norm(tok[0]):.2f}, pooled {pooled.shape}")
        # the two images must differ (else the end-to-end gate proves nothing)
        a = np.fromfile(os.path.join(OUT, f"{model}_street_pooled.bin"), np.float32)
        b = np.fromfile(os.path.join(OUT, f"{model}_cat_pooled.bin"), np.float32)
        print(f"{model}: cosine(street, cat) = {float(a @ b):.4f}")


if __name__ == "__main__":
    main()
