"""Streamlit app for background removal

Features:
- Upload an image
- Choose method: Simple (threshold + largest contour) or GrabCut
- Preview the processed image and download as PNG with alpha

Run:
    streamlit run background_removal.py
"""
from io import BytesIO
from pathlib import Path

import numpy as np
import cv2
import streamlit as st
from PIL import Image


def remove_bg_simple_array(img_array: np.ndarray) -> np.ndarray:
    # img_array is BGR (from cv2) or RGB? We'll expect RGB from PIL and convert.
    if img_array is None:
        raise ValueError("Empty image array")

    # ensure BGR for cv2 operations
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(th)
    if contours:
        c = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [c], -1, 255, -1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    b, g, r = cv2.split(img_bgr)
    rgba = cv2.merge([b, g, r, mask])
    # convert back to RGBA for PIL
    rgba_rgb = cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGBA)
    return rgba_rgb


def remove_bg_grabcut_array(img_array: np.ndarray, iter_count: int = 5, padding: int = 10) -> np.ndarray:
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    rect = (padding, padding, max(1, w - padding * 2), max(1, h - padding * 2))

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(img_bgr, mask, rect, bgdModel, fgdModel, iter_count, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask2 = cv2.GaussianBlur(mask2, (3, 3), 0)

    b, g, r = cv2.split(img_bgr)
    rgba = cv2.merge([b, g, r, mask2])
    rgba_rgb = cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGBA)
    return rgba_rgb


def pil_image_to_numpy(img_pil: Image.Image) -> np.ndarray:
    return np.array(img_pil.convert('RGB'))


def numpy_to_pil_image(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr)


def main():
    st.set_page_config(page_title="Background Removal", layout="centered")
    st.title("Background Removal (OpenCV)")

    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    method = st.radio("Method", ["Simple (threshold)", "GrabCut"])

    iter_count = None
    padding = None
    if method == "GrabCut":
        iter_count = st.slider("GrabCut iterations", 1, 10, 5)
        padding = st.slider("Rect padding (pixels)", 0, 100, 10)

    if uploaded is not None:
        img_pil = Image.open(uploaded).convert('RGB')
        st.image(img_pil, caption="Original", use_column_width=True)

        arr = pil_image_to_numpy(img_pil)

        try:
            if method == "Simple (threshold)":
                out_arr = remove_bg_simple_array(arr)
            else:
                out_arr = remove_bg_grabcut_array(arr, iter_count=iter_count, padding=padding)

            out_pil = numpy_to_pil_image(out_arr)

            st.subheader("Processed")
            st.image(out_pil, caption="Processed (alpha preserved)", use_column_width=True)

            # prepare download
            buf = BytesIO()
            out_pil.save(buf, format="PNG")
            buf.seek(0)

            st.download_button("Download PNG", data=buf, file_name=Path(uploaded.name).stem + "_bg_removed.png", mime="image/png")
        except Exception as e:
            st.error(f"Processing error: {e}")


if __name__ == "__main__":
    main()
