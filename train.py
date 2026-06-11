from ultralytics import YOLO

def main():
    model = YOLO("yolo11n.pt") 
    results = model.train(
        data="data.yaml",   # Path to your dataset config
        epochs=50,          # 50-100 is a good starting point
        imgsz=640,          # Standard image size
        batch=16,           # Lower this to 8 or 4 if you run out of GPU memory
        device=0,           # Use GPU 0
        name="my_custom_yolo11" # Name of the run folder
    )

if __name__ == '__main__':
    main()