from utils import read_video, save_video
from trackers import Tracker
import cv2
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner

def main():
    # Read Video
    video_frames, fps = read_video('input_videos/Screen Recording 2025-02-08 at 00.10.37.mov')

    # Initialize pre-trained model
    tracker = Tracker("models/best.pt")

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/old_version.pkl')
    
    # Interpolate ball position
    tracks['balls'] = tracker.interpolate_ball_positions(tracks['balls'])


    # Assign player team
    team_assigner = TeamAssigner()
    calibaration_check = team_assigner.calibrate_team_colors(video_frames[0], tracks['players'][0])

    if not calibaration_check:
        print("Team Color Calibaration failed")

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'],
                                                 player_id)
            
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Assigned ball Aquisition
    player_assigner = PlayerBallAssigner()
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['balls'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            tracks['players'][frame_num][assigned_player]['with_ball'] = ball_bbox

    # Draw output video
    output_video_frames = tracker.draw_annotations(video_frame=video_frames, tracks=tracks)

    # save video
    save_video(output_video_frames, 'output_videos/team_assigner4.mp4', fps=fps)


if __name__ == "__main__":
    main()