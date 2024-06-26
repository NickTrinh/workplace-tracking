from ultralytics import YOLO
import torch

def run():
    # Check if CUDA is available
    model = YOLO("yolov8s.pt").load("workerguardmodel.pt")
    
    model.train(data="D:/projects/workplace-tracking/data/data.yaml", batch=-1, epochs=300, patience=50, 
                imgsz=1280, cos_lr=True, amp=True, save=True, save_period=50, device=0)  # train the model

if __name__ == '__main__':
    run()