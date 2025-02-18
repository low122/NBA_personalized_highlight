import numpy as np
from sklearn.cluster import KMeans
import umap

class TeamAssignerFeatureBased:

    def __init__(self, n_clusters=2): 
        self.n_clusters = n_clusters
        self.umap_reducer = umap.UMAP(n_neighbors = 2)


    """
    Assign teams to players based on clustering their SigLIP visual features using KMeans after UMAP dimensionality reduction.

    Args:
        player_features_list (list of numpy arrays): List of SigLIP feature vectors for each player.

    Returns:
        list of int: Cluster labels (team assignments) for each player.
    """
    def assign_teams_by_feature_clustering(self, player_features_list):

        if not player_features_list:
            print("Debug: player_features_list is empty at start of assign_teams_by_feature_clustering")
            return []

        feature_matrix = np.vstack(player_features_list)

        if feature_matrix.size == 0:
            print("Warning, n player features available for UMAP")
            return []

        if feature_matrix.shape[0] == 1:
            print("Warning: Too few players detected in this frame for UMAP")
            return [0] * feature_matrix.shape[0]
        
        print(f"Debug: Applying UMAP to {feature_matrix.shape[0]} samples")

        # Apply UMAP for dimensionality reduction BEFORE clustering

        reduced_features = self.umap_reducer.fit_transform(feature_matrix.astype(np.float32))

        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10) # Explicitly set n_init
        clusters = kmeans.fit_predict(reduced_features) # Cluster on reduced features

        return clusters.tolist()
    

    # def get_team_colors(self, cluster_labels):
    #     """
    #     Assigns a color to each team based on cluster labels (simple example).
    #     In a real application, you might want a more robust color assignment strategy.

    #     Args:
    #         cluster_labels (list of int): Cluster labels from KMeans.

    #     Returns:
    #         dict: Mapping of cluster label to color (e.g., {0: (255, 0, 0), 1: (0, 0, 255)}).
    #     """
    #     unique_labels = np.unique(cluster_labels)
    #     colors = {}
    #     # Simple color assignment (for 2 clusters: team 0 = red, team 1 = blue)
    #     if len(unique_labels) == 2:
    #         colors = {unique_labels[0]: (255, 0, 0), unique_labels[1]: (0, 0, 255)} # Red and Blue
    #     elif len(unique_labels) == 3: # Example for 3 clusters
    #         colors = {unique_labels[0]: (255, 0, 0), unique_labels[1]: (0, 0, 255), unique_labels[2]: (0, 255, 0)} # Red, Blue, Green
    #     else: # More than 3 clusters, use distinct random colors (for demonstration)
    #         for label in unique_labels:
    #             colors[label] = tuple(np.random.randint(0, 256, 3).tolist()) # Random RGB color

    #     return colors