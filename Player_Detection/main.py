from utils import read_video, save_video
from tracker import Tracker
import cv2

def main():
    # Read the video
    video_frames = read_video('/content/Juev_2.mp4')

    #Intializer Tracker
    tracker = Tracker('/content/AI_machine_learning/Player_Detection/models/2nd_approach/model_1.pt')
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='/content/AI_machine_learning/Player_Detection/stubs/track_stubs_3.pkl')

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
