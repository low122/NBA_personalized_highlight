from sklearn.cluster import KMeans
from collections import deque, Counter
import cv2
import numpy as np

class TeamAssigner:

    def __init__(self, buffer_size=15):
        self.team_colors = {}
        self.kmeans_model = KMeans(n_clusters=2, random_state=0, init="k-means++", n_init=1)
        self.player_team_dict = {}
        self.color_space = "LAB"
        self.player_team_buffer = {}
        self.min_color_dominance = 0.6
        self.buffer_size = buffer_size

    """
    Get the clustering model for the image

    Output: kmeans: KMeans model
    """
    def get_clustering_model(self, image):
        image_2d = image.reshape(-1,3) # Reshape the image to 2D

        kmeans = self.kmeans_model.fit(image_2d)

        return kmeans
    
    def convert_color_img(self, image):
        if self.color_space == "HSV":
            return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.color_space == "LAB":
            return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        return image
    
    def get_dominant_color(self, frame, top_n=1):
        image = frame.reshape(-1,3)

        criteria = (cv2.TERM_CRITERIA_EPS +cv2.TERM_CRITERIA_MAX_ITER, 200, .1)

        _, labels, centers = cv2.kmeans(image.astype(np.float32), 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

        # Calculate the dominance ratio
        counts = np.bincount(labels.flatten())
        dominance_idx = np.argsort(-counts)[:top_n]
        return centers[dominance_idx], counts[dominance_idx]/len(labels)

    """
    Get the color of the player by clustering the top half of the player's image

    Outpot: player_color: [r, g, b]
    """
    def get_player_color(self, frame, bbox):
        """Improved color extraction with outlier rejection"""
        crop = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        
        # Focus on upper torso region (avoid shorts/background)

        torso = crop[0:int(crop.shape[0]*0.4), :]  # Top 40% of bounding box
        
        # Convert color space and get dominant color
        converted = self.convert_color_img(torso)
        centers, ratios = self.get_dominant_color(converted, top_n=2)
        
        # Ensure dominant color meets threshold
        if ratios[0] < self.min_color_dominance:
            return None  # Reject ambiguous colors
        
        return centers[0]


    """
    Robust team assignment with outlier filtering
    """
    def assign_teams(self, player_colors):
        valid_colors = [c for c in player_colors if c is not None]
        
        if len(valid_colors) < 2:
            print("Not enough data")
            return  # Not enough data
        
        # Use K-means++ with multiple initializations
        kmeans = KMeans(n_clusters=2, n_init=10, random_state=0).fit(valid_colors)
        
        # Order clusters by luminance (helps maintain consistent team IDs)
        self.team_colors = {
            1: kmeans.cluster_centers_[np.argmax(kmeans.cluster_centers_[:,0])],
            2: kmeans.cluster_centers_[np.argmin(kmeans.cluster_centers_[:,0])]
        }

    def get_player_team(self, player_id, current_color):
        """Returns team ID (1 or 2) based on color analysis"""
        if current_color is None:
            # Maintain previous assignment if available
            if player_id in self.player_team_buffer:
                return Counter(self.player_team_buffer[player_id]).most_common(1)[0][0]
            return None

        # Initialize buffer if needed
        if player_id not in self.player_team_buffer:
            self.player_team_buffer[player_id] = deque(maxlen=self.buffer_size)

        # Calculate team distances
        if len(self.team_colors) >= 2:
            distances = [
                np.linalg.norm(current_color - self.team_colors[1]),
                np.linalg.norm(current_color - self.team_colors[2])
            ]
            team_id = np.argmin(distances) + 1
        else:
            # Fallback if clustering hasn't happened yet
            team_id = 1 if np.mean(current_color) > 127 else 2

        # Update buffer
        self.player_team_buffer[player_id].append(team_id)
        
        # Return majority vote
        return Counter(self.player_team_buffer[player_id]).most_common(1)[0][0]