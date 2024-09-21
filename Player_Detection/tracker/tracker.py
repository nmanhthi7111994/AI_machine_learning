import supervision as sv
from ultralytics import YOLO

class Tracker:
  def __init__(self,model_path):
    self.model = YOLO(model_path)
    self.tracker =sv.ByteTrack()

#Summary:
#The detect_frames method processes a list of video frames in batches of 20.
#For each batch, it uses a pre-trained model to perform object detection with a confidence threshold of 0.1.
#The detection results from each batch are accumulated into a list, which is then returned at the end.  

  def detect_frames(self, frames):
        batch_size=20 
        detections = [] 
        for i in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.1)
            detections += detections_batch
            break
        return detections

#The get_object_tracks method processes a sequence of video frames, detects objects within them, and then converts these detections into a format compatible with the supervision library.
#The method loops through each frame's detections, inverts the class name dictionary for easier lookups, and then converts and prints the detection results.
#The method is likely intended to be part of a larger object tracking system, where the detections could be used to track objects across frames.

  def get_object_tracks(self, frames):
    # Get detections for all frames
    detections = self.detect_frames(frames)
    
    for frame_num, detection in enumerate(detections):
        # Access the class names through the 'names' attribute
        cls_names = detection.names  # 'names' is a dictionary mapping class IDs to class names
        cls_names_inv = {v: k for k, v in cls_names.items()}  # Invert the dictionary if needed

        # Convert to supervision format
        detection_supervision = sv.Detections.from_ultralytics(detection)  

        print(detection_supervision)