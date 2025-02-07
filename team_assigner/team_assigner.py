from sklearn.cluster import KMeans
from collections import deque, defaultdict, Counter
import cv2
import numpy as np

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}

        self.team_palettes = {}
        
        self.team_assignment_buffer = defaultdict(lambda: deque(maxlen=15)) # Temporal buffer for team assignments (length 10 frames)
        self.is_calibrated = False
        self.is_done_calibaration = False
        self.team_names = {1: "Team A (Palette 1)", 2: "Team B (Palette 2)"}




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

        center_region_hsv = cv2.cvtColor(center_region, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([center_region_hsv], [0,1,2], None, [4,4,4], [0, 180, 0, 256, 0, 256])
        hist = hist.flatten()

        hist = hist / hist.sum()
        hist = hist.astype(np.float32)

        return hist
    
    
    def calibrate_team_colors(self, frame, player_detections):
        """
        Calibrates team color palettes (histograms) AND representative RGB colors for drawing.
        Uses KMeans clustering on color histograms to find palettes.
        Calculates representative RGB colors from the histogram cluster centers for drawing.
        """
        if self.is_done_calibaration:
            return True

        player_histograms = [] # Store histograms
        player_rgb_colors_for_clustering = [] # Store original RGB colors for getting representative color
        valid_detections = []

        for player_id, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_histogram = self.get_player_color(frame, bbox) # Get histogram
            player_color_rgb = self.get_player_color_rgb_approx(frame, bbox) # Get approx RGB color for clustering

            if player_histogram is not None and player_color_rgb is not None: # Check both are valid
                player_histograms.append(player_histogram) # Append histogram for palette clustering
                player_rgb_colors_for_clustering.append(player_color_rgb) # Append RGB color for representative color calculation
                valid_detections.append(player_detection)

        if not player_histograms or len(player_histograms) < 2:
            print("Warning: Not enough valid player color detections for calibration.")
            return False

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=42) # Cluster player histograms
        kmeans.fit(player_histograms) # Fit KMeans to histograms

        self.team_palettes[1] = kmeans.cluster_centers_[0].astype(np.float32) # Palette 1 is cluster center 1 (histogram)
        self.team_palettes[2] = kmeans.cluster_centers_[1].astype(np.float32) # Palette 2 is cluster center 2 (histogram)

        # Calculate representative RGB colors from the cluster centers (histograms) - Approximation
        # Using the mean of the original RGB colors that belong to each cluster
        labels = kmeans.labels_ # Get cluster labels assigned to each histogram
        cluster_rgb_colors_1 = [player_rgb_colors_for_clustering[i] for i, label in enumerate(labels) if label == 0] # RGB colors in cluster 1
        cluster_rgb_colors_2 = [player_rgb_colors_for_clustering[i] for i, label in enumerate(labels) if label == 1] # RGB colors in cluster 2

        if cluster_rgb_colors_1:
            self.team_colors[1] = np.mean(cluster_rgb_colors_1, axis=0).astype(int).tolist() # Mean RGB for palette 1
        else:
            self.team_colors[1] = [255, 0, 0] # Default color if no detections in cluster 1 (e.g., Blue)

        if cluster_rgb_colors_2:
            self.team_colors[2] = np.mean(cluster_rgb_colors_2, axis=0).astype(int).tolist() # Mean RGB for palette 2
        else:
            self.team_colors[2] = [0, 0, 255] # Default color if no detections in cluster 2 (e.g., Red)


        # Optional: Order palettes - based on average V channel (no change here)
        def get_avg_v_channel(histogram):
            v_bins = 4
            h_bins = 4
            s_bins = 4
            v_hist_1d = histogram.reshape((h_bins, s_bins, v_bins)).sum(axis=(0, 1))
            v_channel_values = np.linspace(0, 255, v_bins)
            avg_v = np.sum(v_hist_1d * v_channel_values) / np.sum(v_hist_1d) if np.sum(v_hist_1d) > 0 else 0
            return avg_v

        avg_v_palette1 = get_avg_v_channel(self.team_palettes[1])
        avg_v_palette2 = get_avg_v_channel(self.team_palettes[2])

        if avg_v_palette2 < avg_v_palette1:
            self.team_palettes[1], self.team_palettes[2] = self.team_palettes[2], self.team_palettes[1]
            self.team_colors[1], self.team_colors[2] = self.team_colors[2], self.team_colors[1] # Also swap RGB colors

        self.kmeans = kmeans
        self.is_calibrated = True
        self.calibration_done = True
        print(f"Team palettes calibrated (histograms) - Palette 1 (avg V: {avg_v_palette1:.2f}), Palette 2 (avg V: {avg_v_palette2:.2f})")
        print(f"Team colors (for drawing) - Team 1: {self.team_colors[1]}, Team 2: {self.team_colors[2]}") # Print RGB colors
        return True

    def get_player_color_rgb_approx(self, frame, bbox): # New helper function to get approx RGB color
        """
        Quickly gets an approximate dominant RGB color (for clustering representative colors).
        Uses the original simpler method to get a dominant RGB color.
        """
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
        if center_region.size == 0:
            return None

        image_2d = center_region.reshape(-1, 3)
        kmeans_rgb_approx = KMeans(n_clusters=1, init="k-means++", n_init=3) # Quick KMeans for approx RGB
        kmeans_rgb_approx.fit(image_2d)
        dominant_color_rgb_approx = kmeans_rgb_approx.cluster_centers_[0]
        return dominant_color_rgb_approx


    
    def get_player_team(self, frame, player_bbox, player_id):
        """
        Assigns player to a team (palette) based on HISTOGRAM COMPARISON to the two palettes
        and applies TEMPORAL SMOOTHING using majority voting.
        """
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id] # Return cached assignment if already assigned

        if not self.is_calibrated:
            print("Warning: Team palettes not calibrated yet.")
            return None

        player_histogram = self.get_player_color(frame, player_bbox) # Get player's color histogram
        if player_histogram is None:
            return None

        # Histogram Comparison - Use Correlation (cv2.HIST_CORREL). Higher value = more similar.
        correlation_1 = cv2.compareHist(player_histogram, self.team_palettes[1], cv2.HISTCMP_INTERSECT)
        correlation_2 = cv2.compareHist(player_histogram, self.team_palettes[2], cv2.HISTCMP_INTERSECT)

        # "Distance" based on correlation (higher correlation = smaller distance)
        distance_1 = 1.0 - correlation_1 # Convert correlation to distance-like measure
        distance_2 = 1.0 - correlation_2

        distances = {
            1: distance_1, # Distance to Palette 1
            2: distance_2  # Distance to Palette 2
        }
        initial_team_id = min(distances, key=distances.get) # Team with minimum distance (highest correlation)


        # Temporal Smoothing - Majority Voting
        self.team_assignment_buffer[player_id].append(initial_team_id) # Add current assignment to buffer
        team_assignment_counts = Counter(self.team_assignment_buffer[player_id]) # Count team assignments in buffer
        majority_team_id = team_assignment_counts.most_common(1)[0][0] # Get the most frequent team

        final_team_id = majority_team_id # Final team after temporal smoothing

        self.player_team_dict[player_id] = final_team_id # Cache the final team assignment
        print(f"Player {player_id} assigned to {self.team_names[final_team_id]} (Histogram Comparison + Temporal Smoothing)")
        return final_team_id