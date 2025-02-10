import sys
import math
sys.path.append('../')
from utils import get_center_of_bbox, calculate_distance

"""
# 情况：目前还是无法锁定离球最近的玩家

1. 思路就是更换计算distance的方式；包括了distancelefthand和righthand
2. 必须了解这里pixels是需要多少，比如说一个boundingbox里面有多少个pixel。
"""

class PlayerBallAssigner():

    def __init__(self):
        self.pixels_per_foot = 50


    def calculate_real_distance(self, p1, p2):
        """Calculate real-world distance between any two points"""
        if not self.pixels_per_foot:
            raise ValueError("Run calibration first!")
            
        pixel_distance = math.dist(p1, p2)
        return pixel_distance / self.pixels_per_foot
    

    def assign_ball_to_player(self, players, ball_bbox):
        ball_center = get_center_of_bbox(ball_bbox)
        

        minimum_distance = float('inf')
        assigned_player = -1

        for player_id, player in players.items():
            player_center = get_center_of_bbox(player['bbox'])
            distance = calculate_distance(player_center, ball_center)

            if distance < minimum_distance and self.calculate_real_distance(ball_center, player_center) <= 7:
                minimum_distance = distance
                assigned_player = player_id
            
        return assigned_player

