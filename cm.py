from ultralytics import YOLO
import numpy as np

model = YOLO("runs/test5_val4/weights/best.pt")

small = "test5/small.yaml"
metrics = model.val(data=small, conf=0.30, imgsz=640)
cm = metrics.confusion_matrix.matrix.T
print(cm)
map = model.val(data=small, imgsz=640)


medium = "test5/medium.yaml"
metrics = model.val(data=medium, conf=0.30, imgsz=640)
cm = metrics.confusion_matrix.matrix.T
print(cm)
map = model.val(data=medium, imgsz=640)


large = "test5/large.yaml"
metrics = model.val(data=large, conf=0.30, imgsz=640)
cm = metrics.confusion_matrix.matrix.T
print(cm)
map = model.val(data=large, imgsz=640)
