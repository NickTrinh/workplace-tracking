from ultralytics import YOLO
import torch

def run():
    model = YOLO("yolov8m.pt")
    
    model.train(data="D:/projects/workplace-tracking/data/data.yaml", batch=-1, epochs=900, patience=90, 
                imgsz=1280, hsv_h=0, hsv_v=0, hsv_s=0, scale=0, shear=0, translate=0, device=0, amp=True, cos_lr=True)

if __name__ == '__main__':
    run()