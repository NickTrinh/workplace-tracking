from ultralytics import YOLO

def run():
    model = YOLO("yolov8n-p2.yaml").load("workerguardmodel.pt")
    
    model.train(data="D:/projects/workplace-tracking/data/data.yaml", batch=-1, epochs=600, patience=60, 
                imgsz=1280, cos_lr=True, exist_ok=True, device=0)

if __name__ == '__main__':
    run()