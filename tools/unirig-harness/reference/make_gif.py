"""Assemble the README GIF from the frames make-gif-frames.mjs captured from the demo page.

  python -I make_gif.py <frames_dir> <out.gif> "<prompt>" [width=480] [fps=15]

Stages (file prefixes): A = the generated model turning, B = the skeleton over the model, C / D / E = dance, wave, walk. A card with the prompt and the picture opens it.
Each frame is cropped to the subject, captioned, and the whole GIF shares one adaptive palette so it stays small.
"""
import sys, glob, os
from PIL import Image, ImageDraw, ImageFont

src, out, prompt = sys.argv[1], sys.argv[2], sys.argv[3]
W = int(sys.argv[4]) if len(sys.argv) > 4 else 480
FPS = int(sys.argv[5]) if len(sys.argv) > 5 else 15
BG = (15, 20, 31)          # the viewer clears to this (0.06, 0.08, 0.12); frames are tested against it


def font(size, bold=False):
    for f in (r'C:\Windows\Fonts\segoeuib.ttf' if bold else r'C:\Windows\Fonts\segoeui.ttf', r'C:\Windows\Fonts\arial.ttf'):
        if os.path.exists(f): return ImageFont.truetype(f, size)
    return ImageFont.load_default()


import json
RECT = json.load(open(os.path.join(src, 'rect.json'))) if os.path.exists(os.path.join(src, 'rect.json')) else None


def frames(prefix):
    """The captured viewport frames, cropped to the canvas (rect.json holds its viewport rectangle)."""
    out = []
    for f in sorted(glob.glob(os.path.join(src, prefix + '_*.png'))):
        im = Image.open(f).convert('RGB')
        if RECT: im = im.crop((round(RECT['x']), round(RECT['y']), round(RECT['x'] + RECT['width']), round(RECT['y'] + RECT['height'])))
        out.append(im)
    return out


def subject_box(images):
    """Union bounding box of the pixels that differ from the background across the frames (so every frame is cropped identically); a margin is ignored (the canvas border)."""
    import numpy as np
    box = None
    for im in images[::2]:
        a = np.asarray(im).astype(int); h, w = a.shape[:2]; m = 10
        inner = a[m:h - m, m:w - m]
        bg = np.median(np.concatenate([inner[:6, :6].reshape(-1, 3), inner[-6:, -6:].reshape(-1, 3)]), axis=0)
        mask = np.abs(inner - bg).sum(axis=2) > 45
        ys, xs = np.nonzero(mask)
        if len(xs) < 50: continue
        b = (xs.min() + m, ys.min() + m, xs.max() + m, ys.max() + m)
        box = b if box is None else (min(box[0], b[0]), min(box[1], b[1]), max(box[2], b[2]), max(box[3], b[3]))
    return box


allf = {k: frames(k) for k in 'ABCDE'}
every = [im for v in allf.values() for im in v]
box = subject_box(every)
cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
side = int(max(box[2] - box[0], box[3] - box[1]) * 1.12)
crop = (cx - side // 2, cy - side // 2, cx + side // 2, cy + side // 2)
CAP = 64
H = W + CAP


def card(stage, title, sub):
    def make(im):
        c = Image.new('RGB', (W, H), BG)
        c.paste(im.crop(crop).resize((W, W), Image.LANCZOS), (0, 0))
        d = ImageDraw.Draw(c)
        d.rectangle([0, W, W, H], fill=(10, 14, 22))
        d.text((16, W + 8), f'{stage}  {title}', font=font(20, True), fill=(235, 238, 245))
        d.text((16, W + 36), sub, font=font(14), fill=(140, 155, 180))
        return c
    return make


def wrap(text, f, width, d):
    words, lines, cur = text.split(), [], ''
    for w in words:
        t = (cur + ' ' + w).strip()
        if d.textlength(t, font=f) <= width: cur = t
        else: lines.append(cur); cur = w
    return lines + [cur]


# opening card: the prompt and the picture FLUX.2 klein drew
intro = Image.new('RGB', (W, H), BG); d = ImageDraw.Draw(intro)
d.text((20, 18), 'Text to 3D, rigged and animated', font=font(26, True), fill=(235, 238, 245))
d.text((20, 54), 'on your GPU, in your browser', font=font(18), fill=(140, 155, 180))
f = font(17); y = 112
for ln in wrap('"' + prompt + '"', f, W - 250, d): d.text((20, y), ln, font=f, fill=(180, 200, 255)); y += 24
pic = os.path.join(src, 'picture.png')
if os.path.exists(pic):
    p = Image.open(pic).convert('RGB').resize((200, 200), Image.LANCZOS); intro.paste(p, (W - 220, 100))
d.text((20, H - 90), 'FLUX.2 klein  >  TRELLIS.2  >  SkinTokens  >  Kimodo', font=font(15), fill=(140, 155, 180))
d.text((20, H - 62), 'all on the GPU, all in the browser', font=font(15), fill=(140, 155, 180))
d.text((20, H - 34), 'SpawnDev.ILGPU.ML', font=font(16, True), fill=(110, 130, 170))

seq = [(intro, 38)]
A = [card('1', 'Text to 3D', 'FLUX.2 klein draws it, TRELLIS.2 builds the mesh and PBR textures')(im) for im in allf['A']]
B = [card('2', 'Rig', 'SkinTokens writes the skeleton and learned skin weights')(im) for im in allf['B']]
CDE = [card('3', 'Animate', 'Kimodo moves retargeted onto the generated skeleton')(im) for k in 'CDE' for im in allf[k]]
for im in A + B + CDE: seq.append((im, 1))
images = [im for im, n in seq for _ in range(1)]
durations = [int(1000 / FPS) * n if n == 1 else int(1000 / FPS) * n for im, n in seq]
# one shared palette: quantise a montage of sample frames, then map every frame onto it
sample = Image.new('RGB', (W, H * 6)); step = max(1, len(images) // 6)
for i in range(6): sample.paste(images[min(i * step, len(images) - 1)], (0, i * H))
pal = sample.quantize(colors=96, method=Image.Quantize.MEDIANCUT)
q = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in images]
q[0].save(out, save_all=True, append_images=q[1:], duration=durations, loop=0, optimize=True, disposal=1)
print(len(q), 'frames', (W, H), round(os.path.getsize(out) / 1e6, 2), 'MB ->', out)
