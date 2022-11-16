import numpy as np
import csv

import matplotlib.pyplot as plt

from dataclasses import dataclass
dt = 0.1
sog = 1.0


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



def cv_object(r, theta, id, sigma):
    return CVObject(np.array((-r * np.cos(theta), -r * np.sin(theta))),
                    np.array(((r/5)*np.cos(theta), (r/5)*np.sin(theta))),
                    id, sigma)
    


def generate_csv(objects, n_steps):

    track_sorted_by_time = []
    for step in range(n_steps):
        tracks = []
        
        for object in objects:
            tracks.append(object.track[step].tolist())
        
        track_sorted_by_time.append(tracks)

    with open(f"{n_traj}tracks.csv", "w", newline="") as f:
        writer = csv.writer(f)
        for step in track_sorted_by_time:

            writer.writerows(step)


def generate_pmbm_data(objects, n_steps, n_traj):
    X_gt = np.empty((n_traj*4, n_steps))
    for step in range(n_steps):
        m_at_step = []
        for object in objects:

            # Offset is done since the clutter PPP in matlab pmbm is only at positive xy
            offset = 15
            m_at_step += [object.track[step][1] + offset]
            m_at_step += [object.track[step][3]]
            m_at_step += [object.track[step][2] + offset]
            m_at_step += [object.track[step][4]]

        X_gt[:, step] = m_at_step

    t_birth = np.ones(n_traj)
    t_death = n_steps * np.ones(n_traj)

    return X_gt, t_birth, t_death
        

def align_objects(objects, timestep):

    aligned_objects = []


    for object in objects:
        x_at_timestep = object.track[timestep][1]
        y_at_timestep = object.track[timestep][2]

        object.track[:, 1] -= x_at_timestep
        object.track[:, 2] -= y_at_timestep

        aligned_objects.append(object)


    return aligned_objects

        

if __name__ == "__main__":

    n_traj = 4
    n_steps = 120

    beta = np.pi / n_traj
    theta = np.pi / 2

    objects = []
    for i in range(n_traj):
        objects.append(cv_object(12.5, theta - i*beta, id=i, sigma=0.2))

    for step in range(n_steps):
        for object in objects:
            object.step()

    objects = align_objects(objects, 60)

    for object in objects:
        object.scatter()

    legend = []
    for i in range(n_traj):
        legend.append(f"T{i}")

    leg = plt.legend(legend)
    for lh in leg.legendHandles: 
        lh.set_alpha(1)

    generate_csv(objects, n_steps)

    X_gt, t_birth, t_death = generate_pmbm_data(objects, n_steps, n_traj)
    np.savetxt(f"./X_gt_{n_traj}.dat", X_gt, delimiter=',')
    np.savetxt(f"./t_birth_{n_traj}.dat", t_birth)
    np.savetxt(f"./t_death_{n_traj}.dat", t_death)

    plt.title(f"Manually generated test scenario with {n_traj} objects. Points with greater alpha are more recent.")

    plt.show()