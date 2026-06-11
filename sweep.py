import wandb
from ultralytics import YOLO


# --- 1) Training function called by wandb.agent ---
def train():
    with wandb.init() as run:
        cfg = run.config

        model = YOLO(cfg.model)

        # warmup = 1/5 of total epochs (e.g., 5->1, 40->8)
        warmup_epochs = max(1, int(cfg.epochs // 5))

        model.train(
            data=cfg.data,
            epochs=cfg.epochs,
            imgsz=cfg.imgsz,
            device=cfg.device,
            name=cfg.run_name,  # local runs/<name>
            project=cfg.ultra_project,

            # --- sweep params ---
            lr0=cfg.lr0,
            batch=cfg.batch,
            mosaic=cfg.mosaic,

            # --- fixed hyps you requested ---
            warmup_epochs=warmup_epochs,
            hsv_h=0.0,
            hsv_s=0.0,
            flipud=0.5,
            cos_lr=True,
            lrf=0.05,
            optimizer="AdamW",
        )


def main():
    # --- 2) Define grid search space (edit values as you like) ---
    sweep_configuration = {
        "method": "grid",
        "metric": {"goal": "maximize", "name": "metrics/mAP50-95(B)"},
        "parameters": {
            # fixed config
            "model": {"value": "yolo11l.pt"},
            "data": {"value": "fold3.yaml"},
            "epochs": {"values": [5, 10, 15, 20, 30, 40]},
            "imgsz": {"value": 640},
            "device": {"value": 0},
            "run_name": {"value": "yolo11l_fold3_sweep"},
            "ultra_project": {"value": "runs"},

            # grid search params
            "lr0": {"values": [1e-3, 1e-4]},
            "batch": {"values": [16, 32, 64]},
            "mosaic": {"values": [0.0, 1.0]},
        },
    }

    # --- 3) Start sweep + agent ---
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="yolo11l-fold3-grid-sweep")
    wandb.agent(sweep_id, function=train)


if __name__ == "__main__":
    main()