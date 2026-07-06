from ultralytics import YOLO
import numpy as np
import os, sys

model = YOLO("runs/test5_val2/weights/best.pt")
yaml = f"yaml/{'fold5.yaml'}"

# Suppress output during val calls
devnull = open(os.devnull, 'w')

sys.stdout, sys.stderr = devnull, devnull
metrics = model.val(data=yaml, conf=0.30, imgsz=640, verbose=False)
map_metrics = model.val(data=yaml, imgsz=640, verbose=False)
sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__

devnull.close()

cm_stats = metrics.confusion_matrix.prf()

print(f"\n{'Class':<20} {'P':>7} {'R':>7} {'F1':>7} {'AP50':>7} {'AP50-95':>9}")
print("-" * 60)
for c, name in metrics.names.items():
    p  = cm_stats["precision"][c]
    r  = cm_stats["recall"][c]
    f1 = cm_stats["f1"][c]
    ap50    = map_metrics.box.ap50[c]
    ap5095  = map_metrics.box.ap[c]
    print(f"{name:<20} {p:>7.4f} {r:>7.4f} {f1:>7.4f} {ap50:>7.4f} {ap5095:>9.4f}")

mean_p  = float(np.mean([cm_stats["precision"][c] for c in metrics.names]))
mean_r  = float(np.mean([cm_stats["recall"][c]    for c in metrics.names]))
mean_f1 = float(np.mean([cm_stats["f1"][c]        for c in metrics.names]))
print("-" * 60)
print(f"{'average':<20} {mean_p:>7.4f} {mean_r:>7.4f} {mean_f1:>7.4f} {map_metrics.box.map50:>7.4f} {map_metrics.box.map:>9.4f}")