from ultralytics import YOLO

model = YOLO("D:/projects/workplace-tracking/runs/detect/train14/weights/best.pt")

model.export(format="torchscript", dynamic=True, optimize=True)