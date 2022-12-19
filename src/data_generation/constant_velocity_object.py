import numpy as np
import csv

import matplotlib.pyplot as plt

from dataclasses import dataclass

@dataclass
class CVObject:
    pos: np.ndarray = None
    vel: np.ndarray = None
    id: int = None
    sigma: float = 1.0
    dt: float = 0.1
    t: float = 0
    
    def __post_init__(self):
        self.process_noise_matrix = self.sigma*np.array([[self.dt ** 3 / 3, self.dt ** 2 / 2], [self.dt ** 2 / 2, self.dt]])
        self.track = np.array(self.current_gt())
        
    def current_gt(self):
        return [[int(self.id), self.pos[0], self.pos[1], self.vel[0], self.vel[1], np.round(self.t, 3)]]
        
    def step(self):
        process_noise = np.random.multivariate_normal([0, 0], self.process_noise_matrix, size=len(self.pos))
        self.pos += self.dt * self.vel + process_noise[:,0]
        self.vel += process_noise[:,1]
        self.t += self.dt

        self.track = np.append(self.track, self.current_gt(), axis=0)

    def scatter(self, decay_opacity=True):
        if decay_opacity:
            alpha = np.linspace(0.25, 1, len(self.track))
        else:
            alpha = 1

        plt.scatter(self.track[:, 1], self.track[:, 2], alpha=alpha)

    def plot(self):
        plt.plot(self.track[:, 1], self.track[:, 2])
        plt.scatter(self.track[0, 1], self.track[0, 2], marker="x")

        #for i in range(int(self.t / self.dt)):
        #    if i % 10 == 0 and i != 0:
        #        plt.scatter(self.track[i, 1], self.track[i, 2], s=10, marker="x", c="k")

    
