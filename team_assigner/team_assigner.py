from sklearn.cluster import KMeans
from collections import deque, Counter
import cv2
import numpy as np

class TeamAssigner:

    def __init__(self, buffer_size=15):
        self.team_colors = None
        self.is_calibrated = False
        self.reference_colors = {1:None, 2:None}
        self.player_team_dict = {}
        self.kmeans_model = KMeans(n_clusters=2, random_state=0, init="k-means++", n_init=1)
        self.player_team_dict = {}
        self.color_space = "LAB"
        self.player_team_buffer = {}
        self.min_color_dominance = 0.6
        self.buffer_size = buffer_size

    def initial_calibration(self, player_colors):
        """One-time calibration using first valid frame"""
        valid_colors = [c for c in player_colors if c is not None]
        
        if len(valid_colors) < 2:
            return False
            
        kmeans = KMeans(n_clusters=2, n_init=20, random_state=42).fit(valid_colors)
        
        # Store reference colors in LAB space
        self.reference_colors = {
            1: kmeans.cluster_centers_[0],
            2: kmeans.cluster_centers_[1]
        }
        self.is_calibrated = True
        return True
        
    
    # def convert_color_img(self, image):
    #     if self.color_space == "HSV":
    #         return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #     elif self.color_space == "LAB":
    #         return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    #     return image
    
    def get_dominant_color(self, frame, top_n=1):
        image = frame.reshape(-1,3).astype(np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

        _, labels, centers = cv2.kmeans(
        image, 
        K=2,  # Reduced from 3 to focus on jersey vs background
        bestLabels=None,
        criteria=criteria,
        attempts=10,
        flags=cv2.KMEANS_PP_CENTERS
    )

        # Calculate the dominance ratio
        counts = np.bincount(labels.flatten())
        dominance_idx = np.argmax(counts)
        return centers[dominance_idx], counts[dominance_idx]/len(labels)

    """
    Get the color of the player by clustering the top half of the player's image

    Outpot: player_color: [r, g, b]
    """
    def get_player_color(self, frame, bbox):
        """Improved color extraction with outlier rejection"""
        crop = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        
        # Focus on the jersey instead of the entire bounding box
        torso = crop[0:int(crop.shape[0]*0.75), :]
        
        # Convert color space and get dominant color
        lab = cv2.cvtColor(torso, cv2.COLOR_BGR2LAB)
        l_channel = lab[:,:,0]
        mask = l_channel > 40  # Filter out dark shadows
        filtered_lab = lab[mask]
        
        if filtered_lab.size == 0:
            return None
        
        centers, dominance = self.get_dominant_color(filtered_lab)
        
        # Ensure dominant color meets threshold
        if dominance < self.min_color_dominance:
            return None  # Reject ambiguous colors
        
        return cv2.cvtColor(np.uint8([[centers]]), cv2.COLOR_LAB2BGR)[0][0]


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

    def get_player_team(self, current_color):
        """Match to reference colors using LAB distance"""
        if not self.is_calibrated or current_color is None:
            return None
            
        distances = [
            np.linalg.norm(current_color - self.reference_colors[1]),
            np.linalg.norm(current_color - self.reference_colors[2])
        ]
        return np.argmin(distances) + 1