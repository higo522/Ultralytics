from ultralytics import YOLO
import numpy as np
import os, sys, csv, glob

devnull = open(os.devnull, 'w')

with open("results_YOLOv11x_CV_narval.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Test Fold", "Val Model", "Class", "Precision", "Recall", "F1", "mAP50", "mAP50:95"])

    for test in range(1, 6):
        yaml = f"yaml/fold{test}.yaml"
        checkpoints = sorted(glob.glob(f"YOLOv11x_CV_narval/test{test}_val*/weights/best.pt"))
        for ckpt in checkpoints:
            val_model = ckpt.split("/")[1]  # e.g. test1_val2
            print(f"Evaluating {val_model} on fold{test}.yaml ...")

            model = YOLO(ckpt)
            sys.stdout, sys.stderr = devnull, devnull
            metrics = model.val(data=yaml, conf=0.30, imgsz=640, verbose=False)
            map_metrics = model.val(data=yaml, imgsz=640, verbose=False)
            sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__

            cm_stats = metrics.confusion_matrix.prf()
            for c, name in metrics.names.items():
                writer.writerow([
                    f"fold{test}", val_model, name,
                    round(cm_stats["precision"][c], 4),
                    round(cm_stats["recall"][c], 4),
                    round(cm_stats["f1"][c], 4),
                    round(map_metrics.box.ap50[c], 4),
                    round(map_metrics.box.ap[c], 4),
                ])
            mean_p  = float(np.mean([cm_stats["precision"][c] for c in metrics.names]))
            mean_r  = float(np.mean([cm_stats["recall"][c]    for c in metrics.names]))
            mean_f1 = float(np.mean([cm_stats["f1"][c]        for c in metrics.names]))
            writer.writerow([
                f"fold{test}", val_model, "average",
                round(mean_p, 4), round(mean_r, 4), round(mean_f1, 4),
                round(map_metrics.box.map50, 4), round(map_metrics.box.map, 4),
            ])

devnull.close()
print("Done. Results saved to results_YOLOv11x_CV_narval.csv")