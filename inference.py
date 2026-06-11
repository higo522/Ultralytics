from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ========= EDIT THESE =========
MODEL = "/home/higo522/YOLO/runs/15_YOLO11l_CV_mosaic1_fold1/weights/last.pt"
IMAGES_DIR = Path("/home/higo522/moose_deer/COCO_sizes/coco_split_f1/large/images")
OUT_DIR = Path("/home/higo522/moose_deer/COCO_sizes/coco_split_f1/large/predictions")
CONF_THR = 0.30
IMGSZ = 640
# ==============================

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# class ids: 0=moose, 1=deer
CLASS_NAMES = {0: "Moose", 1: "Deer"}

# hex colors per predicted class
PR_COLORS = {0: "#F7731B", 1: "#01FF45"} # light blue, light green

def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    """'#RRGGBB' -> (B, G, R)"""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"Expected hex color like '#RRGGBB', got: {hex_color}")
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return (b, g, r)


def draw_boxes_bgr(im_bgr: np.ndarray, boxes_xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> np.ndarray:
    out = im_bgr.copy()

    # thickness scaled to image size
    h, w = out.shape[:2]
    tl = max(2, round(0.0018 * (h + w) / 2) + 1)
    tf = max(1, tl - 1)

    for (x1, y1, x2, y2), c, k in zip(boxes_xyxy, conf, cls):
        k = int(k)
        color = hex_to_bgr(PR_COLORS.get(k, "#FF00FF"))  # fallback magenta
        label = f"{CLASS_NAMES.get(k, str(k))} {float(c):.2f}"

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness=tl, lineType=cv2.LINE_AA)
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(MODEL)

    # Predict directly on a directory. conf applies the threshold you want.
    results = model.predict(
        source=str(IMAGES_DIR),
        conf=CONF_THR,
        imgsz=IMGSZ,
        stream=True,   # don't accumulate all results in RAM
        verbose=False,
        save=False,    # we will save our own annotated images
    )

    n = 0
    for r in results:
        # r.orig_img is BGR uint8 (what we want for cv2)
        im = r.orig_img
        if r.boxes is None or len(r.boxes) == 0:
            annotated = im
        else:
            boxes = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy()
            annotated = draw_boxes_bgr(im, boxes, conf, cls)

        out_path = OUT_DIR / (Path(r.path).stem + ".jpg")
        cv2.imwrite(str(out_path), annotated)
        n += 1

    print(f"Saved {n} annotated images to: {OUT_DIR}")


if __name__ == "__main__":
    main()