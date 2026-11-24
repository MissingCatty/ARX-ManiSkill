"""Draw exact visible-instance contours and readable target labels for the gallery."""
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

YELLOW = (255, 224, 58)
DARK = (23, 28, 32)


@lru_cache(maxsize=8)
def load_font(path, size):
    return ImageFont.truetype(str(path), size)


def mark_target(image, mask, category, font_path, thumbnail=False):
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (image.height, image.width) or not mask.any():
        raise ValueError('Target mask must be nonempty and aligned with the RGB image')
    pixels = np.asarray(image.convert('RGB')).copy()
    pixels[mask] = np.rint(.86 * pixels[mask] + .14 * np.asarray(YELLOW)).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(pixels, contours, -1, DARK, 4, lineType=cv2.LINE_AA)
    cv2.drawContours(pixels, contours, -1, YELLOW, 2, lineType=cv2.LINE_AA)
    marked = Image.fromarray(pixels)
    draw = ImageDraw.Draw(marked)
    label = '目标：' + category
    font = load_font(str(font_path), 11 if thumbnail else 18)
    padding = 4 if thumbnail else 6
    gap = 5 if thumbnail else 9
    text = draw.textbbox((0, 0), label, font=font)
    width, height = text[2] - text[0] + 2 * padding, text[3] - text[1] + 2 * padding
    ys, xs = np.nonzero(mask)
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
    positions = [(center_x - width / 2, y0 - height - gap),
                 (center_x - width / 2, y1 + gap),
                 (x0 - width - gap, center_y - height / 2),
                 (x1 + gap, center_y - height / 2)]
    margin = 3
    candidates = []
    for x, y in positions:
        x = int(np.clip(x, margin, image.width - width - margin))
        if margin <= y <= image.height - height - margin:
            candidates.append((x, int(y)))
    if not candidates:
        candidates = [(margin, margin), (image.width - width - margin, margin),
                      (margin, image.height - height - margin),
                      (image.width - width - margin, image.height - height - margin)]
    # Preserve the visible target surface when positioning the label.
    x, y = min(candidates, key=lambda p: (int(mask[p[1]:p[1]+height, p[0]:p[0]+width].sum()),
                                         (p[0]+width/2-center_x)**2 + (p[1]+height/2-center_y)**2))
    tag_center = (x + width / 2, y + height / 2)
    nearest = int(np.argmin((xs - tag_center[0])**2 + (ys - tag_center[1])**2))
    endpoint = (int(xs[nearest]), int(ys[nearest]))
    draw.line([tag_center, endpoint], fill=DARK, width=4)
    draw.line([tag_center, endpoint], fill=YELLOW, width=2)
    draw.rounded_rectangle((x, y, x + width, y + height), radius=3, fill=DARK, outline=YELLOW, width=1)
    draw.text((x + padding - text[0], y + padding - text[1]), label, font=font, fill=YELLOW)
    return marked, {'label': label, 'visible_target_pixels': int(mask.sum()),
                    'visible_bbox_xyxy': [x0, y0, x1, y1], 'external_contours': len(contours),
                    'label_box_xyxy': [x, y, x + width, y + height]}


def resize_mask(mask, size):
    # Keep small/thin visible components when creating the thumbnail outline.
    return cv2.resize(np.asarray(mask, dtype=np.float32), size, interpolation=cv2.INTER_AREA) > 0.
