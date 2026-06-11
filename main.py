from ultralytics import YOLO
import numpy as np

model = YOLO("runs/test5_val1/weights/best.pt")
yaml = "fold5.yaml"

metrics = model.val(data=yaml, conf = 0.30, imgsz=640)

# Confusion-matrix-derived P/R/F1 (fixed conf=0.30, match IoU=0.50 inside CM code)
cm_stats = metrics.confusion_matrix.prf()

# Per-class
for c, name in metrics.names.items():
    p = cm_stats["precision"][c]
    r = cm_stats["recall"][c]
    f1 = cm_stats["f1"][c]
    tp = int(cm_stats["tp"][c])
    fp = int(cm_stats["fp"][c])
    fn = int(cm_stats["fn"][c])
    print(f"{c:>3} {name:<20} TP= {tp} FP= {fp} FN= {fn} P= {p:.4f} R= {r:.4f} F1= {f1:.4f}")

map = model.val(data=yaml, imgsz=640)