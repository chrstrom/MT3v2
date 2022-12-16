import numpy as np
import csv

import matplotlib.pyplot as plt

from dataclasses import dataclass
import argparse

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

    def plot(self):
        plt.plot(self.track[:, 1], self.track[:, 2])
        plt.scatter(self.track[0, 1], self.track[0, 2], marker="x")

        #for i in range(int(self.t / self.dt)):
        #    if i % 10 == 0 and i != 0:
        #        plt.scatter(self.track[i, 1], self.track[i, 2], s=10, marker="x", c="k")


def cv_object(r, theta, id, sigma):
    return CVObject(np.array((-r * np.cos(theta), -r * np.sin(theta))),
                    np.array(((r/5)*np.cos(theta), (r/5)*np.sin(theta))),
                    id, sigma)
    


def generate_csv(objects, n_steps, title = ""):

    track_sorted_by_time = []
    for step in range(n_steps):
        tracks = []
        
        for object in objects:
            tracks.append(object.track[step].tolist())
        
        track_sorted_by_time.append(tracks)

    if title == "":
        title = f"{n_traj}tracks"
    with open(title + ".csv", "w", newline="") as f:
        writer = csv.writer(f)
        for step in track_sorted_by_time:

            writer.writerows(step)


def generate_pmbm_data(objects, n_steps, n_traj):
    X_gt = np.empty((n_traj*4, n_steps))
    for step in range(n_steps):
        m_at_step = []
        for object in objects:

            # # Offset is done since the clutter PPP in matlab pmbm is only at positive xy
            # offset = 25
            m_at_step += [object.track[step][1]]
            m_at_step += [object.track[step][3]]
            m_at_step += [object.track[step][2]]
            m_at_step += [object.track[step][4]]

        X_gt[:, step] = m_at_step

    t_birth = np.ones(n_traj)
    t_death = n_steps * np.ones(n_traj)

    return X_gt, t_birth, t_death
        

def align_objects(objects, timestep, x=0, y=0):

    aligned_objects = []


    for object in objects:
        x_at_timestep = object.track[timestep][1]
        y_at_timestep = object.track[timestep][2]

        object.track[:, 1] -= x_at_timestep - x
        object.track[:, 2] -= y_at_timestep - y

        aligned_objects.append(object)


    return aligned_objects

def all_positive_objects(objects):

    aligned_objects = []


    for object in objects:
        min_x = np.min(object.track[:][1])
        min_y = np.min(object.track[:][2])

        if min_x < 0:
            object.track[:, 1] += np.abs(min_x)
        if min_y < 0:
            object.track[:, 2] += np.abs(min_y)

        aligned_objects.append(object)


    return aligned_objects

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-nt", "--num_targets")
    parser.add_argument("-ns", "--num_steps")
    args = parser.parse_args()

    n_traj = int(args.num_targets)
    n_steps = int(args.num_steps)


    crossing = True
    if crossing:
        beta = np.pi / n_traj
        theta = np.pi / 2

        objects = []
        for i in range(n_traj):
            objects.append(cv_object(12.5, theta - i*beta, id=i, sigma=0.2))

        for step in range(n_steps):
            for object in objects:
                object.step()

        objects = align_objects(objects, 50, x=7, y=3)

        for object in objects:
            object.plot()

        # legend = []
        # for i in range(n_traj):
        #     legend.append(f"T{i} track")
        #     legend.append(f"T{i} start")

        # leg = plt.legend(legend)
        # for lh in leg.legendHandles: 
        #     lh.set_alpha(1)

        generate_csv(objects, n_steps)

        X_gt, t_birth, t_death = generate_pmbm_data(objects, n_steps, n_traj)
        np.savetxt(f"./X_gt_{n_traj}.dat", X_gt, delimiter=',')
        #np.savetxt(f"./t_birth_{n_traj}.dat", t_birth)
        #np.savetxt(f"./t_death_{n_traj}.dat", t_death)

        #plt.title(f"Manually generated test scenario with {n_traj} objects.")
        fontsize = 26
        plt.xlabel("x [m]", fontsize=fontsize)
        plt.ylabel("y [m]", fontsize=fontsize)

        plt.grid()
        plt.xlim([-10, 20])
        plt.ylim([-10, 20])

        plt.gca().set_aspect('equal')

        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        plt.show()

    else:
 
        objects = []
        for i in range(n_traj):
            objects.append(cv_object(10, np.pi*np.random.random(), id=i, sigma=0.2))

        for step in range(n_steps):
            for object in objects:
                object.step()

        objects = all_positive_objects(objects)

        for object in objects:
            object.scatter()

        legend = []
        for i in range(n_traj):
            legend.append(f"T{i}")

        leg = plt.legend(legend)
        for lh in leg.legendHandles: 
            lh.set_alpha(1)

        generate_csv(objects, n_steps, "X_gt_no_crossing")

        X_gt, t_birth, t_death = generate_pmbm_data(objects, n_steps, n_traj)
        np.savetxt(f"./X_gt_no_cross_{n_traj}.dat", X_gt, delimiter=',')
        np.savetxt(f"./t_birth_no_cross_{n_traj}.dat", t_birth)
        np.savetxt(f"./t_death_no_cross_{n_traj}.dat", t_death)

        plt.title(f"Manually generated test scenario with {n_traj} objects. Points with greater alpha are more recent.")

        plt.show()
