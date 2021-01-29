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

