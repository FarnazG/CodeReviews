import numpy as np
import scipy.stats as stats
import cv2
from scipy.ndimage import gaussian_filter

class OccupancyGrid:
    def __init__(self, lpc, pos, input_map_width=500, input_map_height=500):
        self.mapH = input_map_width
        self.mapW = input_map_height

        self.lpc = lpc
        [self.T, self.lidar_points, self.dim_visible] = lpc.shape

        self.pos = pos

    def draw_map(self, T, map_og):

        layout = map_og.copy()

        for t in range(0, T):
            cv2.circle(layout, (int(self.pos[t, 0]), int(self.pos[t, 1])), 2, 0, -1)

        cv2.imshow('predictions', layout)
        cv2.waitKey(2)

    def transition_state(self, state, state_next):
   
        v = 10.0

        x = state[0]
        y = state[1]
        theta = state[2]

        c = np.cos(np.deg2rad(theta))
        s = np.sin(np.deg2rad(theta))

        state_next_est = np.array([x + v * c, y + v * s])
        state_next_obs = np.squeeze(state_next[0:2])
        dis = np.linalg.norm([state_next_est - state_next_obs], axis=1)
        state_weights = stats.multivariate_normal(0, 25).pdf(dis)
        return state_weights

