from ultralytics import YOLO

model = YOLO("D:/projects/workplace-tracking/runs/detect/train/weights/best.pt")

model.export(format="onnx", dynamic=True, device=0)