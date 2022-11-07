"""

TODO: More realistic track evolution.

"""

import numpy as np
import csv

import matplotlib.pyplot as plt

from dataclasses import dataclass
dt = 0.1
sog = 1.0


@dataclass
class CVObject:
    x: float
    y: float
    vx: float
    vy: float
    id: int = None
    dt: float = 0.1
    t: float = 0
    
    def __post_init__(self):
        self.track = np.array(self.current_gt())
        
    def current_gt(self):
        return [[int(self.id), self.x, self.y, self.vx, self.vy, np.round(self.t)]]
        

    def step(self):
        self.x += self.vx * self.dt
        self.y += self.vy * self.dt
        self.t += self.dt

        self.track = np.append(self.track, self.current_gt(), axis=0)

    def scatter(self, decay_opacity=True):
        if decay_opacity:
            alpha = np.linspace(0.25, 1, len(self.track))
        else:
            alpha = 1

        plt.scatter(self.track[:, 1], self.track[:, 2], alpha=alpha)



def cv_object(r, theta, id):
    return CVObject(-r * np.cos(theta),
                    -r * np.sin(theta),
                    5*np.cos(theta),
                    5*np.sin(theta),
                    id)
    


def generate_csv(objects, n_steps):

    track_sorted_by_time = []
    for step in range(n_steps):
        tracks = []
        
        for object in objects:
            tracks.append(object.track[step].tolist())
        
        track_sorted_by_time.append(tracks)

    with open("tracks.csv", "w", newline="") as f:
        writer = csv.writer(f)
        for step in track_sorted_by_time:

            writer.writerows(step)
        
    

if __name__ == "__main__":

    n_traj = 8
    n_steps = 20

    beta = np.pi / n_traj
    theta = np.pi / 2

    objects = []
    for i in range(n_traj):
        objects.append(cv_object(5, theta - i*beta, id=i))

    for step in range(n_steps):
        for object in objects:
            object.step()

    for object in objects:
        object.scatter()

    legend = []
    for i in range(n_traj):
        legend.append(f"T{i}")

    leg = plt.legend(legend)
    for lh in leg.legendHandles: 
        lh.set_alpha(1)

    generate_csv(objects, n_steps)


        
    plt.title("Manually generated test scenario. Points with greater alpha are more recent.")
    plt.xlim([-11, 11])
    plt.ylim([-11, 11])
    plt.show()