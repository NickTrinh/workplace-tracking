from ultralytics import YOLO
import torch

def run():
    # Check if CUDA is available
    device = torch.device('cuda')
    model = YOLO("yolov8n.pt").load("workerguardmodel.pt")
    model.to(device) #Use GPU
    
    model.train(data="D:/projects/workplace-tracking/data/data.yaml", batch=-1, epochs=300, patience=30, 
                imgsz=1280, cos_lr=True)  # train the model

if __name__ == '__main__':
    run()