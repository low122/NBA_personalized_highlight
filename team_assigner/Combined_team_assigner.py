from .team_assigner_color_based import TeamAssignerColorBased
from .team_assigner_feature_based import TeamAssignerFeatureBased

class CombinedTeamAssigner: # Simplified Combined Team Assigner
    def __init__(self, umap_n_neighbors=2, umap_n_components=2):
        self.color_team_assigner = TeamAssignerColorBased()
        self.feature_team_assigner = TeamAssignerFeatureBased(umap_n_neighbors=umap_n_neighbors, umap_n_components=umap_n_components)

    def calibrate_team_colors(self, frame, player_detections):
        return self.color_team_assigner.calibrate_team_colors(frame, player_detections)

    def get_player_team_color_based(self, frame, player_bbox, player_id):
        return self.color_team_assigner.get_player_team(frame, player_bbox, player_id)

    def assign_teams_by_features(self, player_features_list):
        return self.feature_team_assigner.assign_teams_by_feature_clustering(player_features_list)