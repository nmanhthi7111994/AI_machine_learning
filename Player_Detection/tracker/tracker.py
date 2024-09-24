from numpy.lib import utils
from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import sys 
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width

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
        return detections

  def draw_ellipse(self,frame,bbox,color,track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        #Continue to mark the id for the collected object of the 
        rectangle_width = 40
        rectangle_heigh= 20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2 - rectangle_heigh//2) + 15
        y2_rect = (y2 + rectangle_heigh//2) + 15

        if track_id is not None:
              cv2.rectangle(frame,
                            (int(x1_rect),int(y1_rect)),
                            (int(x2_rect),int(y2_rect)),
                            color,
                            cv2.FILLED
                )
              #This aim to designer for the number more than 99             
              x1_text=x1_rect+12
              if track_id > 99: 
                  x1_text -=10
              
              #Put the track id into frame objec go along with design pattern 
              cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
              )


        return frame


#The get_object_tracks method processes a sequence of video frames, detects objects within them, and then converts these detections into a format compatible with the supervision library.
#The method loops through each frame's detections, inverts the class name dictionary for easier lookups, and then converts and prints the detection results.
#The method is likely intended to be part of a larger object tracking system, where the detections could be used to track objects across frames.

  def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        
    if read_from_stub and stub_path is not None and os.path.exists(stub_path):
        with open(stub_path,'rb') as f:
            tracks = pickle.load(f)
        return tracks

    detections = self.detect_frames(frames) 
    tracks={
            "players":[],
            "referees":[],
            "ball":[]
        }

    for frame_num, detection in enumerate(detections):
        # Access the class names through the 'names' attribute
        cls_names = detection.names  # 'names' is a dictionary mapping class IDs to class names
        cls_names_inv = {v: k for k, v in cls_names.items()}  # Invert the dictionary if needed

        # Convert to supervision format
        detection_supervision = sv.Detections.from_ultralytics(detection)  

        # Convert GoalKeeper to player object
        for object_ind , class_id in enumerate(detection_supervision.class_id):
            if cls_names[class_id] == "goalkeeper":
                detection_supervision.class_id[object_ind] = cls_names_inv["player"]

        #Track the object
        destionation_with_tracks = self.tracker.update_with_detections(detection_supervision)
        print(destionation_with_tracks)

        #This part initializes empty dictionaries in tracks for storing the tracked bounding boxes (bbox) of players, referees, and the ball for each frame.
        # These dictionaries are stored as lists, where each list entry corresponds to a specific frame.

        tracks["players"].append({})
        tracks["referees"].append({})
        tracks["ball"].append({})

        # If the detection is classified as a player or referee, 
        # the bounding box is stored in the appropriate dictionary (players or referees) under the corresponding track_id.
        # This way, you can keep track of where each player or referee is located across frames.
        # Bounding boxes for players, referees, and the ball are stored in dictionaries that track their locations across frames.

        for frame_detection in destionation_with_tracks:
            bbox = frame_detection[0].tolist()
            cls_id = frame_detection[3]
            track_id = frame_detection[4]

            if cls_id == cls_names_inv['player']:
                tracks["players"][frame_num][track_id] = {"bbox":bbox}
            
            if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}

        for frame_detection in detection_supervision:
            bbox = frame_detection[0].tolist()
            cls_id = frame_detection[3]

            if cls_id == cls_names_inv['ball']:
                tracks["ball"][frame_num][1] = {"bbox":bbox}

    #Subsequent Runs: If the same frames need to be processed again, and read_from_stub=True, 
    # the method can skip the tracking process and load the tracks data from the stub file, providing an immediate result.           

    if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)      

    return tracks

  #Draw a triangle to get the dimension (horizon and veritical )for tracking the ball moving/progressing
  # The dimension we used here is the 2-dimensions X and Y
  # This corrected code will properly draw a triangle on the frame at the bounding box's location.
  def draw_triangle(self,frame, bbox, color):
      y= int(bbox[1])#This line extracts the vertical position (y-coordinate) from the bbox.
      x,_ = get_center_of_bbox(bbox) #. The _ suggests that the function returns two values, and we are only interested in the x-coordinate.
      #Provide triangle with specific information
      triangle_points = np.array([
          [x,y],
          [x-10,y-20],
          [x+10,y-20],
      ])

      #This line draws the triangle again, but with a black border (color (0, 0, 0)) of thickness 2 pixels.
      # This creates an outline around the filled triangle for better visibility.
      cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
      cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

      
      return frame

  def draw_annotations(self,video_frames, tracks):
    output_video_frames= []
    for frame_num, frame in enumerate(video_frames):
        frame = frame.copy()

        player_dict = tracks["players"][frame_num] if frame_num < len(tracks["players"]) else {}
        ball_dict = tracks["ball"][frame_num] if frame_num < len(tracks["ball"]) else {}
        referee_dict = tracks["referees"][frame_num] if frame_num < len(tracks["referees"]) else {}

        # Draw Players
        for track_id, player in player_dict.items():
            #color = player.get("team_color",(0,0,255))
            frame = self.draw_ellipse(frame, player["bbox"],(0,0,255), track_id)

        #Draw Referee
        for _,referee in referee_dict.items():
          frame = self.draw_ellipse(frame,referee["bbox"],(0,255,255))

        
        #Draw a ball
        for track_id,ball in ball_dict.items():
          frame = self.draw_triangle(frame,ball["bbox"],(0,255,0))

        

        output_video_frames.append(frame)

    return output_video_frames
    