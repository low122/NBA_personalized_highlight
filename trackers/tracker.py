from collections import defaultdict
from ultralytics import YOLO
import supervision as sv
import pickle
import os
import cv2
import sys
import math
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from transformers import AutoProcessor, AutoModel
from PIL import Image

sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, crop_bbox_from_frame
from player_ball_assigner import PlayerBallAssigner
from team_assigner import TeamAssignerFeatureBased

class Tracker:
    
    def __init__(self, model_path, reid_model_name = 'resnet50', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = YOLO(model_path)  # Trained model
        self.tracker = sv.ByteTrack()
        self.pixels_per_foot = 50
        self.team_assigner = TeamAssignerFeatureBased()
        self.player_ball_assigner = PlayerBallAssigner()
        self.class_name_inv = {
            'Player': 1,
            'Ball': 0,
            "Rim":2
        }

        # self.siglip_device = 'cpu' # Store device for SigLIP
        # siglip_model_name = "google/siglip-base-patch16-224" # Choose SigLIP model
        # self.siglip_processor = AutoProcessor.from_pretrained(siglip_model_name)
        # self.siglip_model = AutoModel.from_pretrained(siglip_model_name).to(self.siglip_device).eval() # Load model to device and set to eval mode

    def detect_frames(self, frames):
        batch_size = 20
        detections = []

        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detections_batch

        return detections
    
    # tracker.py
    def interpolate_ball_positions(self, ball_positions):
    # Convert list-based structure back to track format
        ball_track_data = []
        for frame_num, frame_data in enumerate(ball_positions):
            for track_id, data in frame_data.items():
                bbox = data['bbox']
                if bbox:
                    ball_track_data.append({'frame': frame_num, 'track_id': track_id, 
                                            'x1': bbox[0], 
                                            'y1': bbox[1], 
                                            'x2': bbox[2], 
                                            'y2': bbox[3]})
                else:
                    ball_track_data.append({'frame': frame_num, 
                                            'track_id': track_id, 
                                            'x1': None, 
                                            'y1': None, 'x2': None, 'y2': None})
        
        # Interpolate each track
        interpolated = defaultdict(dict)
        track_ids = set(entry['track_id'] for entry in ball_track_data)

        for track_id in track_ids:
            track_positions = [entry for entry in ball_track_data if entry['track_id'] == track_id]
            if track_positions: # Check if there are positions for this track_id
                df = pd.DataFrame(track_positions)
                df = df.set_index('frame')
                numeric_cols = ['x1', 'y1', 'x2', 'y2']
                for col in numeric_cols: # Ensure numeric type for interpolation
                    df[col] = pd.to_numeric(df[col])

                full_index = pd.RangeIndex(start=0, stop=len(ball_positions))
                df = df.reindex(full_index).interpolate().bfill().ffill()

                # Reconstruct bbox and assign interpolated values
                interpolated_bboxes = df[['x1', 'y1', 'x2', 'y2']].values.tolist()
                for idx, bbox in enumerate(interpolated_bboxes):
                    interpolated[idx][track_id] = {'bbox': bbox}
            else:
                # Handle case where track_id has no positions
                for idx in range(len(ball_positions)):
                    interpolated[idx][track_id] = {'bbox': []}

        return [interpolated[i] for i in range(len(ball_positions))]
        
    # tracker.py
    def _smooth_tracks(self, tracks):
        """Improved temporal smoothing with track-aware handling"""
        max_age = 5

        for obj_type in ['balls', 'players', 'rims']:
            all_track_ids = set()
            # Collect all track IDs
            for frame in tracks[obj_type]:
                all_track_ids.update(frame.keys())
            
            # Smooth each track individually
            for track_id in all_track_ids:
                prev_data = None
                consecutive_missing = 0
                for frame in range(len(tracks[obj_type])):
                    frame_data = tracks[obj_type][frame]
                    current_data = frame_data.get(track_id)

                    if current_data is not None:
                        prev_data = current_data
                        consecutive_missing = 0  # Reset counter on detection
                    else:
                        if prev_data is not None and consecutive_missing < max_age:
                            # Carry forward previous data
                            tracks[obj_type][frame][track_id] = prev_data
                            consecutive_missing += 1
                        else:
                            # Remove the track from this frame if it's too old
                            if track_id in tracks[obj_type][frame]:
                                del tracks[obj_type][frame][track_id]
                            consecutive_missing = 0  # Reset counter if desired
        return tracks

    """
    output: return a dictionary form of tracking information e.g. bbox, including the player, ball, and rim
    """
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path:
            if os.path.exists(stub_path):
                if os.path.getsize(stub_path) == 0:
                    print(f"Warning: Empty stub file {stub_path}, regenerating tracks")
                    read_from_stub = False
                else:
                    try:
                        with open(stub_path, 'rb') as f:
                            return pickle.load(f)
                    except (EOFError, pickle.UnpicklingError) as e:
                        print(f"Corrupted stub file: {e}, regenerating tracks")
                        read_from_stub = False

        tracking_config = {
            'persist': True,
            'tracker': 'botsort.yaml',
            'conf': 0.4,
            'iou': 0.3,
            'verbose': False
        }

        tracks = {
            'players': [{} for _ in range(len(frames))],
            'balls': [{} for _ in range(len(frames))],
            'rims': [{} for _ in range(len(frames))]
        }

        print("\n--- Starting get_object_tracks ---") # Added print at start

        for frame_num, frame in tqdm(enumerate(frames), total=len(frames), desc="Tracking"):
            result = self.model.track(frame, **tracking_config)[0]

            for det in result.boxes:
                cls_id = int(det.cls)
                track_id = int(det.id)
                bbox = det.xyxy[0].tolist()
                conf = float(det.conf)

                if cls_id == self.class_name_inv['Player']:
                    tracks['players'][frame_num][track_id] = {"bbox": bbox, "confidence": conf} # Store in frame_players
                elif cls_id == self.class_name_inv['Ball']:
                    tracks['balls'][frame_num][1] = {"bbox": bbox, "confidence": conf}        # Store in frame_balls
                elif cls_id == self.class_name_inv['Rim']:
                    tracks['rims'][frame_num][track_id] = {"bbox": bbox, "confidence": conf} # Store in frame_rims

        # tracks = self._smooth_tracks(tracks) # Comment out smoothing for now for debugging

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks, f)

        return tracks
    
    """
    To visualize the outcome of the tracking
    """
    def draw_annotations(self, video_frame, tracks):
        output_video_frames = []
        current_teams = {}

        for frame_num, original_frame in enumerate(video_frame):
            frame = original_frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["balls"][frame_num]
            rim_dict = tracks['rims'][frame_num]

            """
            For simplicity, only abstract the rim that first appeared
            """
            rim_center = None  # Initialize with default
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
                    # pixels_per_foot = 10
                    distance_ft = self.player_ball_assigner.calculate_real_distance(player_center, rim_center)
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

                if player.get('has_ball', False):
                    frame = self.draw_rectangle(frame, player["bbox"])
                    ball_with_player = player.get('with_ball')
                    if ball_with_player is not None:
                        center_of_bbox_ball = get_center_of_bbox(ball_with_player)
                        frame = self.draw_line(frame, center_of_bbox_ball, player_center)

            # Draw ball
            for track_id, ball in ball_dict.items(): 
                frame = self.draw_rectangle(frame, ball["bbox"])


            output_video_frames.append(frame)

        return output_video_frames
    

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
    
    def draw_rectangle(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)

        cv2.rectangle(frame,
                      (x1, y1), # Top left corner
                      (x2, y2), # Top right corner
                      (0,0,255),
                      3)
        
        return frame
    
    def draw_line(self, frame, ball_center, player_center):
        cv2.line(frame, ball_center, player_center, (0,0,255), 3)

        distance_ft = self.player_ball_assigner.calculate_real_distance(ball_center, player_center)
        mid_point = (
            (ball_center[0] + player_center[0]) // 2,
            (ball_center[1] + player_center[1]) // 2
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

        return frame