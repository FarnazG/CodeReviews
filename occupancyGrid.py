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

    def extract_beam_points(self, yk, state):
        x = state[0]
        y = state[1]
        theta = state[2]

        deg = np.arctan2(yk[:, 1], yk[:, 0]) + np.deg2rad(theta)
        r = np.linalg.norm([yk[:, 0], yk[:, 1]], axis=0)

        defection_coef = 0.1
        wr = np.where(np.random.random(len(r)) < defection_coef)
        r[wr[0]] = r[wr[0]] / 2.0

        c = np.cos(deg)
        s = np.sin(deg)
        hit_pnts = np.array([np.asarray(x + r * c, int), np.asarray(y + r * s, int)])
        beam_pnts = []

        for i in range(0, len(r)):
            for br in range(0, int(r[i]), 2):
                xr = x + int(br * c[i])
                yr = y + int(br * s[i])
                beam_pnts.append([xr, yr])

        return hit_pnts.transpose(),  np.asarray(beam_pnts, dtype=np.int)

    def start(self):
        map_og1 = 0.5 * np.ones(shape=[self.mapH, self.mapW])
        
        sigma0 = 1.5
        coef0 = 1.0 / stats.multivariate_normal(0, sigma0 ** 2).pdf(0.0)

        
        sigma1 = 2.5
        coef1 = 1.0 / stats.multivariate_normal(0, sigma1 ** 2).pdf(0.0)

        for t in range(0, self.T - 1):
            print("t: ", t)
            yk = self.lpc[t, :, :]
            state = self.pos[t, :]
                        
            hit_pnts, beam_pnts = self.extract_beam_points(yk, state)

            conv_map1 = gaussian_filter(map_og1, sigma=sigma1, cval=0.5)

            tmp_conv_map_og0 = (coef1 * conv_map1 - map_og1) * (1.0 - map_og1)
            tmp_conv_map_og1 = coef1 * conv_map1 * map_og1
            conv_map_og1 = tmp_conv_map_og1 / (tmp_conv_map_og0 + tmp_conv_map_og1)

        return map_og1
