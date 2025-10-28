"""GrabCut-based background removal script (robust)

Usage:
    python scripts/remove_bg_grabcut.py input.jpg output.png [iterations] [padding]

Follows Method B from the repository README: initialize rect -> grabCut -> mask -> refine -> save PNG with alpha.
"""
import sys
import cv2
import numpy as np
from pathlib import Path


def remove_bg_grabcut(in_path: str, out_path: str, iter_count: int = 5, rect_padding: int = 10) -> None:
    img = cv2.imread(in_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {in_path}")

    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)

    # rectangle slightly inside the image
    rect = (rect_padding, rect_padding, max(1, w - rect_padding * 2), max(1, h - rect_padding * 2))

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, iter_count, cv2.GC_INIT_WITH_RECT)

    # mask: 0/2 background, 1/3 foreground
    mask2 = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')

    # refine
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask2 = cv2.GaussianBlur(mask2, (3, 3), 0)

    b, g, r = cv2.split(img)
    rgba = cv2.merge([b, g, r, mask2])

    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, rgba)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/remove_bg_grabcut.py input.jpg output.png [iterations] [padding]")
        sys.exit(1)

    in_p = sys.argv[1]
    out_p = sys.argv[2]
    iters = int(sys.argv[3]) if len(sys.argv) >= 4 else 5
    padding = int(sys.argv[4]) if len(sys.argv) >= 5 else 10

    try:
        remove_bg_grabcut(in_p, out_p, iter_count=iters, rect_padding=padding)
        print(f"Saved: {out_p}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(2)
