

from matplotlib.image import Bbox


class TeamAssigner:
    def __init__(self):
        pass

    def get_player_color(self,frame,bbox)
        image = frame

    def assgin_team_color(self,frame,player_detections):

      player_color = []
      for _,player_detection in player_detections.items():
          bbox = player_detection["bbox"]
          player_color = self.get_player_color(frame,bbox)