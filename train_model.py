from ultralytics import YOLO
import torch

def run():
    # Check if CUDA is available
    device = torch.device('cuda')
    
    model = YOLO("yolov8n.pt")
    
    model.to(device) #Use GPU
    
    model.train(data="D:/projects/workplace-tracking/data/data.yaml", batch=-1, epochs=150,     patience=20)  # train the model

if __name__ == '__main__':
    run()
