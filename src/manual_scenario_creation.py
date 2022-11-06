"""
TODO: Recreate data generator based on constant velocity vectors 
      Which are euler stepped to generate positions, under the constraint that the 
      velocity length vector is fixed for all tracks (to ensure collision)

TODO: Generate CSV 


"""

import numpy as np

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
    dt: float = 1
    
    def __post_init__(self):
        self.track = np.array([[self.x, self.y]])
        

    def step(self):
        self.x += self.vx * self.dt
        self.y += self.vy * self.dt
        self.track = np.append(self.track, [[self.x, self.y]], axis=0)

    def scatter(self, decay_opacity=True):
        if decay_opacity:
            alpha = np.linspace(0.25, 1, len(self.track))
        else:
            alpha = 1

        plt.scatter(self.track[:, 0], self.track[:, 1], alpha=alpha)


def plot_lambda(x, f):
    decaying_opacity = np.linspace(0.25, 1, len(x))
    plt.scatter(f(x)[0], f(x)[1], alpha=decaying_opacity)


def cv_object(r, theta):
    return CVObject(-r * np.cos(theta),
                    -r * np.sin(theta),
                    np.cos(theta),
                    np.sin(theta))
    

# def generate_csv(X, functions):
    
#     for step in X:
#         for f in functions:
#             x = f(X[step])[0]
#             y = f(X[step])[1]
#             t = np.round(step * 0.1, 4)
#             vx = 
#             vy = 



if __name__ == "__main__":

    n_traj = 8
    n_steps = 20

    beta = np.pi / n_traj
    theta = np.pi / 2

    objects = []
    for i in range(n_traj):
        objects.append(cv_object(10, theta - i*beta))

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
        
    plt.title("Manually generated test scenario. Points with greater alpha are more recent.")
    plt.xlim([-11, 11])
    plt.ylim([-11, 11])
    plt.show()