from sklearn.cluster import KMeans
from collections import deque, defaultdict
import cv2
import numpy as np

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}

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

    def assign_team_color(self, frame, player_detections):
        """
        Assign team color to player detections
        """
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)


        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):

        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        player_color = self.get_player_color(frame, player_bbox)
        
        team_id = self.kmeans.predict([player_color])[0]
        team_id += 1

        self.player_team_dict[player_id] = team_id

        return team_id