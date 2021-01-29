import numpy as np
import scipy.stats as stats
import cv2



class LocalizationParticleFilter:
    def __init__(self, input_visible, map_name, dim_hidden=3, particle_num=2000):

        self.layout_org = cv2.imread(map_name)
        self.layout = np.copy(self.layout_org)
        [self.mapH, self.mapW, self.mapC] = self.layout.shape
        self.particle_num = particle_num

        self.dim_hidden = dim_hidden
        self.input_visible = input_visible
        [self.T, self.lidar_points, self.dim_visible] = input_visible.shape

        self.particles = np.zeros([self.particle_num, self.dim_hidden])
        self.weights = np.zeros(self.particle_num)
        self.state_average = []

        self.edge_margin = 10

    def draw_predictions(self, particles, weights):
        self.layout = np.copy(self.layout_org)
        for k in range(0, particles.shape[0]):
            cv2.circle(self.layout, (int(particles[k, 0]), int(particles[k, 1])), 1, (255, 0, 0), -1)

        self.state_average.append(np.average(particles[:, 0:2], axis=0, weights=weights))

        for mean_state_k in self.state_average:
            cv2.circle(self.layout, (int(mean_state_k[0]), int(mean_state_k[1])), 2, (0, 0, 255), -1)

        cv2.imshow('predictions', self.layout)
        cv2.waitKey(2)

