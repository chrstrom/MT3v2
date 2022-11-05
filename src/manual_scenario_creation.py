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
    dt: float = 0.1
    
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



# def generate_csv(X, functions):
    
#     for step in X:
#         for f in functions:
#             x = f(X[step])[0]
#             y = f(X[step])[1]
#             t = np.round(step * 0.1, 4)
#             vx = 
#             vy = 



if __name__ == "__main__":

    T0 = CVObject(0, -10, sog, 0)

    n_steps = 20

    for step in range(n_steps):
        T0.step()


    T0.scatter()

    # T0 = lambda x : (np.repeat(0, len(x)), x)
    # T1 = lambda x : (x, x)
    # T2 = lambda x : (x, np.repeat(0, len(x)))
    # T3 = lambda x : (x, -x)

    # X = np.linspace(-10, 10, 21)

    # plot_lambda(X, T0)
    # plot_lambda(X, T1)
    # plot_lambda(X, T2)
    # plot_lambda(X, T3)

    # leg = plt.legend(["T0","T1","T2","T3"])
    # for lh in leg.legendHandles: 
    #     lh.set_alpha(1)
    # plt.title("Manually generated test scenario. Points with greater alpha are more recent.")
    plt.show()