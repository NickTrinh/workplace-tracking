import os
import cv2
import torch
from ultralytics import YOLO

model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'best.pt')
model = YOLO(model_path)  # load a custom model
class_list = ['guard', 'worker']
#threshold = 0.5

VIDEOS_DIR = os.path.join('.', 'videos') # get test video path
video_path = os.path.join(VIDEOS_DIR, 'test.mp4')
video_path_out = '{}_out.mp4'.format(os.path.splitext(video_path)[0]) # output video

cap = cv2.VideoCapture(video_path) # in video
ret, frame = cap.read()
H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H)) # out video

# Loop through the video frames
direction={}
direction_counter=set()

while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()
    
    if ret:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, tracker='bytetrack.yaml')
        
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        
        line_y = 320
        x_start, x_end = 550, 850
        cv2.line(annotated_frame,(x_start,line_y),(x_end,line_y),(255,255,255),2)
        
        boxes = results[0].boxes.xyxy.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []
        #print(track_ids)
        
        for box, class_id, track_id in zip(boxes, class_ids, track_ids):
            # Ensure the tensor is on CPU and convert it to numpy for cv2 compatibility
            x_min, y_min, x_max, y_max = box
            
            # Calculate the center point of the bounding box
            center_x = int(x_min + x_max) // 2
            center_y = int(y_min + y_max) // 2
            
            # Draw a red circle at the center point
            cv2.circle(annotated_frame, (center_x, center_y), radius=5, color=(0, 0, 255), thickness=-1)
            
            #print(x_min, y_min, x_max, y_max, class_id)
            #print('\n')
            
            if line_y < (center_y + 7) and line_y > (center_y - 7) and  int(class_id) == 1:
                direction[track_id] = center_y
                
                if track_id in direction:
                    direction_counter.add(track_id)
        
        worker_count = (len(direction_counter))
        cv2.putText(annotated_frame, "Worker count: " + str(worker_count), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        # Display the frame with the annotated results
        out.write(annotated_frame)

# Release the video capture object and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()