from utils import read_video, save_video
from tracker import Tracker

def main():
    # Read the video
    video_frames = read_video('/content/Juve_1.mp4')

    #Intializer Tracker
    tracker = Tracker('/content/AI_machine_learning/Player_Detection/models/2nd_approach/model_1.pt')
    tracks = tracker.get_object_tracks(video_frames,read_from_stub=True,stub_paths='/content/stubs/track_stubs.pkl')
    # Save video
    save_video(video_frames, '/content/output/output_1.avi')

   

if __name__ == '__main__':
    main()
