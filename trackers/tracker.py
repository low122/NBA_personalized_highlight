from ultralytics import YOLO
import supervision as sv
import pickle
import os
import cv2
import sys
import math
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, calculate_distance
from team_assigner import TeamAssigner

class Tracker:
    
    def __init__(self, model_path):
        self.model = YOLO(model_path)  # Trained model
        self.tracker = sv.ByteTrack()
        self.pixels_per_foot = 50
        self.team_assigner = TeamAssigner()
        self.temporal_team_buffer = {}

    def detect_frames(self, frames):
        batch_size = 20
        detections = []

        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detections_batch

        return detections
    

    """
    output: return a dictionary form of tracking information e.g. bbox, including the player, ball, and rim
    """
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks
        
        detections = self.detect_frames(frames)

        # Dictionary format
        tracks={
            'players':[], # [{0:{"bbox":[0,0,0,0]}, ....}]
            'balls':[],
            'rims':[]
        }

        for frame_num, detection in enumerate(detections):
            cls_name = detection.names

            cls_name_inv = {v:k for k,v in cls_name.items()}

            """
            1. Convert to supervision Detection format

            2. Track Objects
                give the position of detected objects
            """
            detection_supervision = sv.Detections.from_ultralytics(detection)
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks['players'].append({})
            tracks['balls'].append({})
            tracks['rims'].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_name_inv['Players']:
                    tracks['players'][frame_num][track_id] = {"bbox":bbox}

            # Below is the code for tracking the ball and rim
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_name_inv['Ball']:
                    tracks['balls'][frame_num][1] = {"bbox":bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_name_inv['Rim']:
                    tracks['rims'][frame_num][track_id] = {"bbox":bbox}

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox=bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)),
            angle=0,
            startAngle=-50,
            endAngle=260,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        return frame

    def calculate_real_distance(self, p1, p2):
        """Calculate real-world distance between any two points"""
        if not self.pixels_per_foot:
            raise ValueError("Run calibration first!")
            
        pixel_distance = math.dist(p1, p2)
        return pixel_distance / self.pixels_per_foot
    
    """
    To visualize the outcome of the tracking
    """
    def draw_annotations(self, video_frame, tracks):
        output_video_frames = []
        current_teams = {}

        for frame_num, frame in enumerate(video_frame):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["balls"][frame_num]
            rim_dict = tracks['rims'][frame_num]

            """
            For simplicity, only abstract the rim that first appeared
            """
            rim_bbox = None
            if rim_dict:
                rim_bbox = next(iter(rim_dict.values()))["bbox"]
                rim_center = get_center_of_bbox(rim_bbox)

            # Draw players
            for track_id, player in player_dict.items(): 

                player_bbox = player["bbox"]
                player_center = get_center_of_bbox(player_bbox)

                color = player.get("team_color")

                if rim_bbox and player_bbox:
                    pixels_per_foot = 10
                    distance_ft = self.calculate_real_distance(player_center, rim_center)
                    cv2.putText(
                        frame, 
                        f"{distance_ft:.1f} ft", 
                        (player_center[0], player_center[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        (255, 255, 255), 
                        2
                    )

                # This is the player's frame
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

            self.temporal_team_buffer.update(current_teams)

            # Draw ball
            for track_id, ball in ball_dict.items(): 
                frame = self.draw_ellipse(frame, ball["bbox"], (0,255,255), track_id)

                ball_bbox = ball["bbox"]
                ball_center = get_center_of_bbox(ball_bbox)

                """
                Find the closest player to the ball
                """
                min_distance = float('inf')
                closest_player = None
                for track_id, player in player_dict.items():
                    player_center = get_center_of_bbox(player["bbox"])
                    distance = calculate_distance(player_center, ball_center)
                    if distance < min_distance:
                        min_distance = distance
                        closest_player = player_center

                
                if closest_player:
                    cv2.line(frame, ball_center, closest_player, (0, 255, 0), 2)
                    distance_ft = self.calculate_real_distance(ball_center, closest_player)
                    mid_point = (
                        (ball_center[0] + closest_player[0]) // 2,
                        (ball_center[1] + closest_player[1]) // 2
                    )
                    cv2.putText(
                        frame, 
                        f"{distance_ft:.1f} ft", 
                        mid_point, 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        (255, 255, 255), 
                        2
                    )

            output_video_frames.append(frame)

        return output_video_frames