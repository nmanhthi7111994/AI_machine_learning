from utils import read_video, save_video
from tracker import Tracker
import cv2
from team_assigner import TeamAssigner
import numpy as np
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator

def main():
    # Read the video
    video_frames = read_video(r'C:\Users\Admin\Documents\Github\AI_machine_learning\Juve_2.mp4')


    #Intializer Tracker
    tracker = Tracker(r'C:\Users\Admin\Documents\Github\AI_machine_learning\Project_Player_Detection\models\2nd_approach\8th_training_result.pt')
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path=r'C:\Users\Admin\Documents\Github\AI_machine_learning\Project_Player_Detection\stubs\track_stubs_5.pkl')
    
    #Get Object Postions
    tracker.add_position_to_tracks(tracks)

    # camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                read_from_stub=True,
                                                                                stub_path='stubs/camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)

    # View Trasnformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    #Assign Player Teams
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


    # Assign Ball Aquisition
    player_assigner = PlayerBallAssigner()
    team_ball_control= []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])##team_ball_control = [1,2,2] -> this is an array for the code progressing per frame
        else:
            # If team_ball_control is empty, add a default value (e.g., 0 or another team)
            if team_ball_control:
                team_ball_control.append(team_ball_control[-1])  # Continue with the last team
            else:
                team_ball_control.append(0)  # Default value for the first frame

    team_ball_control = np.array(team_ball_control)

    #Save cropped image of a player
    for track_id, player in tracks['players'][0].items():
      bbox = player['bbox']
      frame = video_frames[0]

      #crop bbox from the frame
      cropped_image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]

      #Save the cropped image
      cv2.imwrite(r'C:\Users\Admin\Documents\Github\AI_machine_learning\Image_cropped.mp4', cropped_image)
      break
               

    #Draw output
    ## Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames,tracks,team_ball_control)

    ## Draw Camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)

    ## Draw Speed and Distance
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames,tracks)
    
    # Save video
    save_video(output_video_frames, r'C:\Users\Admin\Documents\Github\AI_machine_learning\Output_detection.avi')

   

if __name__ == '__main__':
    main()
