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

 
