"""Simple background removal script (uniform backgrounds)

Usage:
    python scripts/remove_bg_simple.py input.jpg output.png

This follows Method A from the repository README: grayscale -> threshold -> largest contour -> mask -> save PNG with alpha.
"""
import sys
import cv2
import numpy as np
from pathlib import Path


def remove_bg_simple(in_path: str, out_path: str) -> None:
    img = cv2.imread(in_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {in_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # keep largest contour
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(th)
    if contours:
        c = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [c], -1, 255, -1)

    # refine mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    b, g, r = cv2.split(img)
    rgba = cv2.merge([b, g, r, mask])

    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, rgba)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/remove_bg_simple.py input.jpg output.png")
        sys.exit(1)

    in_p = sys.argv[1]
    out_p = sys.argv[2]
    try:
        remove_bg_simple(in_p, out_p)
        print(f"Saved: {out_p}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(2)
