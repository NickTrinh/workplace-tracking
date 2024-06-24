import os
import cv2
import torch
from ultralytics import YOLO

model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'best.pt')
model = YOLO(model_path)  # load a custom model
#threshold = 0.5

VIDEOS_DIR = os.path.join('.', 'videos') # get test video path
video_path = os.path.join(VIDEOS_DIR, 'test.mp4')
video_path_out = '{}_out.mp4'.format(os.path.splitext(video_path)[0]) # output video

cap = cv2.VideoCapture(video_path) # in video
ret, frame = cap.read()
H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'mp4v'), 
                    int(cap.get(cv2.CAP_PROP_FPS)), (W, H)) # out video

direction={}
direction_counter=set()

# Function to check if two boxes overlap
def boxes_overlap(box1, box2):
    x_min1, y_min1, x_max1, y_max1 = box1
    x_min2, y_min2, x_max2, y_max2 = box2
    
    return not (x_max1 < x_min2 or x_max2 < x_min1 or y_max1 < y_min2 or y_max2 < y_min1)

mask = cv2.imread('videos/mask.png')

while cap.isOpened():
    # Read frame from video
    ret, frame = cap.read()
    
    if ret:
        imgRegion = cv2.bitwise_and(frame, mask)
        # Run YOLOv8 tracking on the frame
        results = model.track(imgRegion, persist=True, tracker='botsort.yaml', imgsz=1280, 
                            device='cuda:0', iou=0.3, conf=0.2)
        
        # Visualize results on the frame
        annotated_frame = results[0].plot(img=frame)
        
        
        line_y = 300
        x_start, x_end = 550, 850
        cv2.line(annotated_frame,(x_start,line_y),(x_end,line_y),(0,0,255),2)
        
        boxes = results[0].boxes.xyxy.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []
        #print(track_ids)
        
        class_0_boxes = []
        class_2_boxes = []
        
        for box, class_id, track_id in zip(boxes, class_ids, track_ids):
            # Ensure the tensor is on CPU and convert it to numpy for cv2 compatibility
            x_min, y_min, x_max, y_max = box
            
            if class_id == 0:
                class_0_boxes.append((box, track_id))
            elif class_id == 2:
                class_2_boxes.append((box, track_id))
            
            # Calculate the center point of the bounding box
            center_x = int(x_min + x_max) // 2
            center_y = int(y_min + y_max) // 2
            
            # Draw a red circle at the center point
            cv2.circle(annotated_frame, (center_x, center_y), radius=5, color=(0, 0, 255), thickness=-1)
            
            # Ensure track_id exists in direction
            if track_id not in direction:
                direction[track_id] = 'not_tested'  # Initialize with a default status
            
            
            if direction[track_id] == 'tested':
                cv2.putText(annotated_frame, "Tested", (center_x, center_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if line_y < (center_y + 7) and line_y > (center_y - 7) and  int(class_id) == 2:
                direction[track_id] = center_y
                
                if track_id in direction:
                    direction_counter.add(track_id)
                    cv2.line(annotated_frame,(x_start,line_y),(x_end,line_y),(0,255,0),2)
            
        for box2, track_id2 in class_2_boxes:
            for box0, _ in class_0_boxes:
                if boxes_overlap(box2, box0):
                    # Mark as tested
                    # Update the 'tested' status
                    direction[track_id2] = 'tested'
                    x_min, y_min, x_max, y_max = box2
                    center_x = int(x_min + x_max) // 2
                    center_y = int(y_min + y_max) // 2
                    break  # Assuming only one breathalyzer test is needed per worker
        
        worker_count = (len(direction_counter))
        cv2.putText(annotated_frame, "Worker count: " + str(worker_count), (10,30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        # Display the frame with the annotated results
        out.write(annotated_frame)
    else:
        break

# Release the video capture object and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()