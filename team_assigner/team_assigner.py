from sklearn.cluster import KMeans
from collections import deque, defaultdict
import cv2
import numpy as np

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
        
        self.color_buffer = defaultdict(lambda: deque(maxlen=15))
        self.is_calibrated = False
        self.is_done_calibaration = False

    def get_clustering_model(self, image):
        """
        Get KMeans clustering model
        """
        image_2d = image.reshape(-1, 3)

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self, frame, bbox):

        x1, y1, x2, y2 = map(int, bbox)
        bbox_width = x2 - x1
        bbox_height = y2 - y1

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        region_width = int(bbox_width * 0.2)
        region_height = int(bbox_height * 0.2)

        center_region = frame[
        max(0, center_y - region_height // 2):min(frame.shape[0], center_y + region_height // 2),
        max(0, center_x - region_width // 2):min(frame.shape[1], center_x + region_width // 2)
        ]

        kmeans = self.get_clustering_model(center_region)

        # Get the cluster labels for each pixel
        labels = kmeans.labels_

        dominant_cluster = np.argmax(np.bincount(labels))
        player_color = kmeans.cluster_centers_[dominant_cluster]

        return player_color

    # def assign_team_color(self, frame, player_detections):
    #     """
    #     Assign team color to player detections, and assigned ONLY once
    #     """
    #     player_colors = []
    #     for _, player_detection in player_detections.items():
    #         bbox = player_detection["bbox"]
    #         player_color = self.get_player_color(frame, bbox)
    #         player_colors.append(player_color)


    #     kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
    #     kmeans.fit(player_colors)

    #     self.kmeans = kmeans

    #     self.team_colors[1] = kmeans.cluster_centers_[0]
    #     self.team_colors[2] = kmeans.cluster_centers_[1]
    #     self.is_calibrated = True

    #     return True
    
    def calibrate_team_colors(self, frame, player_detections):
        """
        Calibrates team colors based on initial player detections.
        This should be called only once at the beginning.
        """
        if self.is_done_calibaration:
            return True # Calibration already done

        player_colors = []
        valid_detections = [] # Keep track of valid detections with colors

        for player_id, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            if player_color is not None: # Only consider valid colors
                player_colors.append(player_color)
                valid_detections.append(player_detection) # Keep valid detection

        if not player_colors or len(player_colors) < 2: # Need at least 2 player colors for 2 clusters
            print("Warning: Not enough valid player color detections to calibrate team colors.")
            return False # Indicate calibration failure


        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=42) # Added random_state for reproducibility
        kmeans.fit(player_colors)

        # Assign team colors based on average color (you might need a more robust method)
        avg_colors = [np.mean([player_colors[i] for i in np.where(kmeans.labels_ == j)[0]], axis=0) for j in range(2)]

        # Assign team colors based on cluster centers directly
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

        # Optional: Order team colors by average RGB value to make team 1 and team 2 consistent across runs
        avg_rgb_team1 = np.mean(self.team_colors[1])
        avg_rgb_team2 = np.mean(self.team_colors[2])
        if avg_rgb_team2 < avg_rgb_team1: # Ensure team 1 is "darker" or consistently assigned
            self.team_colors[1], self.team_colors[2] = self.team_colors[2], self.team_colors[1] # Swap

        self.kmeans = kmeans # Keep kmeans model if needed for debugging
        self.is_calibrated = True
        self.calibration_done = True # Mark calibration as done
        print(f"Team colors calibrated: Team 1: {self.team_colors[1]}, Team 2: {self.team_colors[2]}")
        return True


    def update_player_color_buffer(self, player_id, color):
        """
        Update the color buffer for a player with a new color sample.
        """
        if color is not None:
            self.color_buffer[player_id].append(color)

    def get_dominant_color_from_buffer(self, player_id):
        """
        Get the dominant color for a player from their color buffer.
        """
        colors = list(self.color_buffer.get(player_id, []))
        if not colors:
            return None

        # Use K-means to find the dominant color in the buffer
        kmeans = KMeans(n_clusters=1, init="k-means++", n_init=10)
        kmeans.fit(colors)
        return kmeans.cluster_centers_[0]

    
    def get_player_team(self, frame, player_bbox, player_id):

        if not self.is_calibrated:
            return None

        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        player_color = self.get_player_color(frame, player_bbox)

        # Buffer for robustness puporses
        self.update_player_color_buffer(player_id, player_color)
        dominant_color = self.get_dominant_color_from_buffer(player_id)
        
        # team_id = self.kmeans.predict([player_color])[0]
        # team_id += 1

        distances = [
            np.linalg.norm(dominant_color - self.team_colors[1]),
            np.linalg.norm(dominant_color - self.team_colors[2])
        ]
        team_id = np.argmin(distances) + 1

        self.player_team_dict[player_id] = team_id

        return team_id