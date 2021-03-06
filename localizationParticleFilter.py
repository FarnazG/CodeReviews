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

    def transition_state(self, state, weights):
    
        v = 10.0

        x = state[:, 0]
        y = state[:, 1]
        theta = state[:, 2]

        rand_theta = np.random.normal(0.0, 60.0, size=state.shape[0])
        next_weights = weights * stats.multivariate_normal(0, 3600).pdf(rand_theta)
        next_weights = next_weights / next_weights.sum()

        theta_next = theta + rand_theta
        theta_next = theta_next.astype(int)
        theta_next[theta_next < -179] += 360
        theta_next[180 < theta_next] -= 360

        c = np.cos(np.deg2rad(theta_next))
        s = np.sin(np.deg2rad(theta_next))

        randn = np.random.normal(0.0, 2.5, size=state.shape[0])  # + np.random.gamma(1, 3, size=state.shape[0])
        x_next = x + (v + randn) * c
        x_next = x_next.astype(int)
        x_next[x_next < self.edge_margin] = self.edge_margin
        x_next[(self.mapW - self.edge_margin) <= x_next] = self.mapW - self.edge_margin

      
        y_next = y + (v + randn) * s
        y_next = y_next.astype(int)
        y_next[y_next < self.edge_margin] = self.edge_margin
        y_next[(self.mapH - self.edge_margin) <= y_next] = self.mapH - self.edge_margin

        return np.array([x_next, y_next, theta_next]).transpose(), next_weights

    def intersect_lidar_beam(self, x, y, deg):
        c = np.cos(np.deg2rad(deg))
        s = np.sin(np.deg2rad(deg))
        max_range = int(np.linalg.norm([self.mapW, self.mapH]))

        for r in range(1, max_range, 9):
            xr = x + int(r * c)
            yr = y + int(r * s)
            if self.layout_org[yr, xr, 0] == 0:
                return r

        return 10*max_range

    def likelihood_func(self, yk, state, variance):
        x = state[0]
        y = state[1]
        theta = state[2]
        dis_yk = np.linalg.norm(yk, axis=1)
        dis = []
        for deg in range(theta, 360 + theta, 5):
            dis.append(self.intersect_lidar_beam(x, y, deg))

        log_dis_ratio = np.sort(np.abs(np.log(np.asarray(dis) / dis_yk)))
        likelihood = np.array(log_dis_ratio[0:int(1.0*len(log_dis_ratio))])
        likelihood = stats.multivariate_normal(0.0, variance).pdf(likelihood)

        
        return likelihood.mean()

    
    def hist_resampler(self, particles, weights, new_particle_num, drop_percentage=10):
        w_percentile = np.percentile(weights, drop_percentage)
        w = np.asarray(weights / w_percentile, int)
        wint = w.copy()
        for i in range(1, len(w)):
            wint[i] = w[i] + wint[i-1]

        next_particles = np.zeros([new_particle_num, 3])
        next_weights = np.zeros(new_particle_num)
        rand_num = np.random.randint(1, wint[len(w) - 1], new_particle_num)
        print("max number: ", wint[len(w) - 1])

        for i in range(0, new_particle_num):
            for j in range(0, len(w)):
                if rand_num[i] <= wint[j]:
                    next_weights[i] = weights[j]
                    next_particles[i, :] = particles[j, :]
                    break

        next_weights = next_weights / next_weights.sum()
        return next_particles, next_weights

 
