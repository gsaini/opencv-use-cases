# Open CV Use Cases

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Streamlit](https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)

## Remove Background from Images Using OpenCV

Quick, practical steps and two working approaches: (A) fast for simple/uniform backgrounds, (B) robust using GrabCut for general photos.

Prerequisites

- Python 3.x
- pip install opencv-python numpy

Method A — Simple background removal (uniform background)

1. Convert image to grayscale, blur, and threshold.
2. Find the largest contour (assumed foreground).
3. Create a mask, refine with morphology, apply mask and save as PNG with alpha.

Example:

```python
import cv2
import numpy as np

def remove_bg_simple(in_path, out_path):
    img = cv2.imread(in_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # keep largest contour
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(th)
    if contours:
        c = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [c], -1, 255, -1)

    # refine mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (5,5), 0)

    b, g, r = cv2.split(img)
    rgba = [b, g, r, mask]
    out = cv2.merge(rgba, 4)
    cv2.imwrite(out_path, out)
```

Method B — Robust removal using GrabCut

1. Initialize a mask (rectangle or rough mask) and run cv2.grabCut.
2. Convert the result to a binary mask, refine with morphology.
3. Apply the mask to the original and save with alpha.

Example:

```python
import cv2
import numpy as np

def remove_bg_grabcut(in_path, out_path, iter_count=5, rect_padding=10):
    img = cv2.imread(in_path)
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)

    # rectangle slightly inside the image
    rect = (rect_padding, rect_padding, w-rect_padding*2, h-rect_padding*2)

    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)

    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, iter_count, cv2.GC_INIT_WITH_RECT)

    # mask: 0/2 background, 1/3 foreground
    mask2 = np.where((mask==2)|(mask==0), 0, 255).astype('uint8')

    # refine
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask2 = cv2.GaussianBlur(mask2, (3,3), 0)

    b, g, r = cv2.split(img)
    rgba = cv2.merge([b, g, r, mask2])
    cv2.imwrite(out_path, rgba)
```

Tips

- For complex scenes, combine user input (scribbles) with GrabCut by initializing mask values cv2.GC_FGD/cv2.GC_BGD.
- Adjust morphological kernel sizes and blur to control edge softness.
- Save output as PNG to preserve transparency.
- For batch processing, run these functions over your dataset and tune parameters per image type.

Streamlit app

This repository includes a simple Streamlit app `background_removal.py` that lets you upload an image, choose a background removal method (Simple or GrabCut), preview the result and download a PNG with transparency.

Run the app:

```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run background_removal.py
```

Use the sliders when GrabCut is selected to tune iterations and rectangle padding.
