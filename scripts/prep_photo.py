#!/usr/bin/env python3
"""
Prepare the avatar/photo for clean ASCII conversion:
  1. Optionally remove background (if rembg installed) or tone-isolate
  2. Smooth texture while keeping key edges sharp
  3. Stretch tones and boost contrast
  4. Output grayscale source-prepped.png for make_ascii_svg.py
Usage:
    python scripts/prep_photo.py [source-photo.png] [source-prepped.png]
"""
import os
import sys
from PIL import Image, ImageEnhance, ImageOps, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
INP = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "source-photo.png")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "source-prepped.png")

def prep_image(inp_path, out_path):
    # Try using rembg and opencv if available
    has_rembg = False
    try:
        import cv2
        import numpy as np
        from rembg import remove
        has_rembg = True
    except ImportError:
        pass

    if has_rembg:
        try:
            print("Processing with rembg + OpenCV...")
            import cv2
            import numpy as np
            from rembg import remove

            cut = remove(Image.open(inp_path).convert("RGBA"))
            rgb = np.array(cut.convert("RGB"))
            alpha = np.array(cut.split()[-1])
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

            smooth = gray
            for _ in range(3):
                smooth = cv2.bilateralFilter(smooth, 9, 40, 9)

            lo, hi = np.percentile(smooth[alpha > 128], [2, 92])
            tone = np.clip((smooth.astype(np.float32) - lo) / (hi - lo), 0, 1)

            fine = cv2.GaussianBlur(smooth, (0, 0), 1.5).astype(np.float32)
            coarse = cv2.GaussianBlur(smooth, (0, 0), 6).astype(np.float32)
            lines = np.clip((coarse - fine) / 40.0, 0, 1)
            out = np.clip(tone - 0.6 * lines, 0, 1) * 255

            mask = cv2.GaussianBlur(alpha.astype(np.float32) / 255.0, (0, 0), 1.0)
            out = out * mask + 255.0 * (1.0 - mask)

            ys, xs = np.where(alpha > 20)
            if len(xs) > 0 and len(ys) > 0:
                side = max(xs.max() - xs.min(), ys.max() - ys.min()) + 60
                cx, cy = (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2
                canvas = np.full((side, side), 255, np.uint8)
                x0, y0 = cx - side // 2, cy - side // 2
                sx0, sy0 = max(x0, 0), max(y0, 0)
                sx1, sy1 = min(x0 + side, out.shape[1]), min(y0 + side, out.shape[0])
                canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = out[sy0:sy1, sx0:sx1].astype(np.uint8)
                Image.fromarray(canvas, mode="L").save(out_path)
                print("Wrote (rembg):", out_path)
                return
        except Exception as e:
            print(f"rembg processing skipped ({e}), falling back to PIL pipeline...")

    # High quality PIL/NumPy pipeline
    print("Processing with Pillow pipeline...")
    img = Image.open(inp_path).convert("RGBA")
    
    # If transparent, composite onto white
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    img = Image.alpha_composite(bg, img).convert("L")

    # Square crop center
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))

    # Auto-contrast & tone enhancement
    img = ImageOps.autocontrast(img, cutoff=1)
    img = ImageEnhance.Contrast(img).enhance(1.35)
    img = ImageEnhance.Sharpness(img).enhance(1.5)

    img.save(out_path)
    print("Wrote (PIL):", out_path, img.size)

if __name__ == "__main__":
    prep_image(INP, OUT)
