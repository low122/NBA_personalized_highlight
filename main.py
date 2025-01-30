from utils import read_video, save_video
from trackers import Tracker
import cv2
from team_assigner import TeamAssigner

def main():
    # Read Video
    video_frames, fps = read_video('input_videos/Screen Recording 2025-01-25 at 23.48.48.mov')

    #Initialize tracker
    tracker = Tracker("models/best.pt")

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs_realline.pkl')


    
    # Draw output and tracks
    output_video_frames = tracker.draw_annotations(video_frame=video_frames, tracks=tracks)


    # save video
    save_video(output_video_frames, 'output_videos/output_video3.mp4', fps=fps)


if __name__ == "__main__":
    main()