from ultralytics import YOLO
import torch

def run():
    # Check if CUDA is available
    device = torch.device('cuda')
    
    model = YOLO("yolov8m.pt")
    
    model.to(device) #Use GPU
    
    model.train(data="C:/Users/nhatt/OneDrive/Desktop/workplace-tracking/data/data.yaml", batch=-1, epochs=150,patience=50)  # train the model

if __name__ == '__main__':
    run()
