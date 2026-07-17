import os
import sys
from pathlib import Path

import wandb
from ultralytics import YOLO

REPO_ROOT = Path.home() / "scratch" / "Ultralytics"

FOLD_YAMLS = [
    "ssh/yaml/test1/val2.yaml",
    "ssh/yaml/test1/val3.yaml",
    "ssh/yaml/test1/val4.yaml",
    "ssh/yaml/test1/val5.yaml",
    "ssh/yaml/test2/val1.yaml",
    "ssh/yaml/test2/val3.yaml",
    "ssh/yaml/test2/val4.yaml",
    "ssh/yaml/test2/val5.yaml",
    "ssh/yaml/test3/val1.yaml",
    "ssh/yaml/test3/val2.yaml",
    "ssh/yaml/test3/val4.yaml",
    "ssh/yaml/test3/val5.yaml",
    "ssh/yaml/test4/val1.yaml",
    "ssh/yaml/test4/val2.yaml",
    "ssh/yaml/test4/val3.yaml",
    "ssh/yaml/test4/val5.yaml",
    "ssh/yaml/test5/val1.yaml",
    "ssh/yaml/test5/val2.yaml",
    "ssh/yaml/test5/val3.yaml",
    "ssh/yaml/test5/val4.yaml",
]


def main():
    # Fold index comes from the SLURM array task ID (or a CLI arg for manual runs)
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ["SLURM_ARRAY_TASK_ID"])
    data_yaml = REPO_ROOT / FOLD_YAMLS[idx]

    p = Path(data_yaml)
    run_name = f"{p.parent.name}_{p.stem}"

    with wandb.init(project="YOLO11x_Heldout_CV_narval_rerun", name=run_name, reinit=True):
        model = YOLO(str(REPO_ROOT / "yolo11x.pt"))
        model.train(
            data=str(data_yaml),
            epochs=30,
            imgsz=640,
            device=0,
            name=run_name,
            project=REPO_ROOT / "YOLOv11x_CV_narval_rerun",
            lr0=0.0001,
            batch=32,
            warmup_epochs=3,
            hsv_h=0.0,
            hsv_s=0.0,
            flipud=0.5,
            mosaic=1.0,
            cos_lr=True,
            lrf=0.05,
            optimizer="AdamW",
        )


if __name__ == "__main__":
    main()
