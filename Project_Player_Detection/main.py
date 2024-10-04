from utils import read_video, save_video
from tracker import Tracker
import cv2
from team_assigner import TeamAssigner

def main():
    # Read the video
    video_frames = read_video('/content/Juve_2.mp4')

    #Intializer Tracker
    tracker = Tracker('/content/AI_machine_learning/Player_Detection/models/2nd_approach/3th_player_detection_best.pt.pt')
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='/content/AI_machine_learning/Player_Detection/stubs/track_stubs_3.pkl')

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

     # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], 
                                    tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],   
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    #Save cropped image of a player
    for track_id, player in tracks['players'][0].items():
      bbox = player['bbox']
      frame = video_frames[0]

      #crop bbox from the frame
      cropped_image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]

      #Save the cropped image
      cv2.imwrite(f'/content/cropped_img.jpg', cropped_image)
      break
               

    #Draw output
    ## Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames,tracks)
    
    # Save video
    save_video(output_video_frames, '/content/output_2.avi')

   

if __name__ == '__main__':
    main()
