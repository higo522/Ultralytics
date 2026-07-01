import wandb
from pathlib import Path
from ultralytics import YOLO

FOLD_YAMLS = [
    "yaml/test1/val2.yaml",
    "yaml/test1/val3.yaml",
    "yaml/test1/val4.yaml",
    "yaml/test1/val5.yaml",
    "yaml/test2/val1.yaml",
    "yaml/test2/val3.yaml",
    "yaml/test2/val4.yaml",
    "yaml/test2/val5.yaml",
    "yaml/test3/val1.yaml",
    "yaml/test3/val2.yaml",
    "yaml/test3/val4.yaml",
    "yaml/test3/val5.yaml",
    "yaml/test4/val1.yaml",
    "yaml/test4/val2.yaml",
    "yaml/test4/val3.yaml",
    "yaml/test4/val5.yaml",
    "yaml/test5/val1.yaml",
    "yaml/test5/val2.yaml",
    "yaml/test5/val3.yaml",
    "yaml/test5/val4.yaml",
]


def main():
    for data_yaml in FOLD_YAMLS:
        p = Path(data_yaml)
        run_name = f"{p.parent.name}_{p.stem}"

        with wandb.init(project="YOLO11x_Heldout_CV", name=run_name, reinit=True):
            model = YOLO("yolo11x.pt")
            model.train(
                data=data_yaml,
                epochs=30,
                imgsz=640,
                device=0,
                name=run_name,
                project="YOLOv11x_CV",
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