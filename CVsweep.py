import wandb
from pathlib import Path
from ultralytics import YOLO


# Edit this to point at your fold-specific dataset YAMLs
FOLD_YAMLS = [
    "test1/val2.yaml",
    "test1/val3.yaml",
    "test1/val4.yaml",
    "test1/val5.yaml",
    "test2/val1.yaml",
    "test2/val3.yaml",
    "test2/val4.yaml",
    "test2/val5.yaml",
    "test3/val1.yaml",
    "test3/val2.yaml",
    "test3/val4.yaml",
    "test3/val5.yaml",
    "test4/val1.yaml",
    "test4/val2.yaml",
    "test4/val3.yaml",
    "test4/val5.yaml",
    "test5/val1.yaml",
    "test5/val2.yaml",
    "test5/val3.yaml",
    "test5/val4.yaml",
]

# Fixed best settings from your sweep
MODEL = "yolo11l.pt"
LR0 = 0.0001
BATCH = 32
MOSAICS = [1.0]

# Keep everything else the same as sweep.py
EPOCHS = 30
IMGSZ = 640
DEVICE = 0
BASE_RUN_NAME = "YOLO11l_Heldout_CV"
ULTRA_PROJECT = "runs"
WANDB_PROJECT = "YOLO11l_Heldout_CV"

WARMUP_EPOCHS = max(1, EPOCHS // 10)


def train_one_fold(data_yaml: str, run_name: str, mosaic: float):
    model = YOLO(MODEL)
    return model.train(
        data=data_yaml,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        device=DEVICE,
        name=run_name,
        project=ULTRA_PROJECT,
        lr0=LR0,
        batch=BATCH,
        warmup_epochs=WARMUP_EPOCHS,
        hsv_h=0.0,
        hsv_s=0.0,
        flipud=0.5,
        mosaic=mosaic,
        cos_lr=True,
        lrf=0.05,
        optimizer="AdamW",
    )


def main():
    for mosaic in MOSAICS:
        mosaic_tag = f"mosaic{int(mosaic)}"
        group_name = f"{BASE_RUN_NAME}_{mosaic_tag}"

        for data_yaml in FOLD_YAMLS:
            p = Path(data_yaml)
            name = f"{p.parent.name}_{p.stem}"
            run_name = name

            with wandb.init(
                project=WANDB_PROJECT,
                name=run_name,
                group=group_name,
                config={
                    "model": MODEL,
                    "data": data_yaml,
                    "epochs": EPOCHS,
                    "imgsz": IMGSZ,
                    "device": DEVICE,
                    "lr0": LR0,
                    "batch": BATCH,
                    "warmup_epochs": WARMUP_EPOCHS,
                    "hsv_h": 0.0,
                    "hsv_s": 0.0,
                    "flipud": 0.5,
                    "mosaic": mosaic,
                    "cos_lr": True,
                    "lrf": 0.05,
                    "optimizer": "AdamW",
                },
                reinit=True,
            ):
                train_one_fold(data_yaml, run_name, mosaic)


if __name__ == "__main__":
    main()