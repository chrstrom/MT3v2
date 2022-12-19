import numpy as np
import matplotlib.pyplot as plt

import argparse



from utility import ground_truth_csv_from_cv_object, ground_truth_dat_from_cv_object, cv_object_from_r_theta

dt = 0.1
sog = 1.0


def align_objects(objects, timestep, x=0, y=0):

    aligned_objects = []

    for object in objects:
        x_at_timestep = object.track[timestep][1]
        y_at_timestep = object.track[timestep][2]

        object.track[:, 1] -= x_at_timestep - x
        object.track[:, 2] -= y_at_timestep - y

        aligned_objects.append(object)

    return aligned_objects

    
def create_crossing_scenario(n_targets, n_timesteps, sigma, intersect, x, y):

        beta = np.pi / n_targets
        theta = np.pi / 2

        objects = []
        for i in range(n_traj):
            objects.append(cv_object_from_r_theta(12.5, theta - i*beta, id=i, sigma=sigma))

        for step in range(n_steps):
            for object in objects:
                object.step()

        objects = align_objects(objects, intersect, x, y)

        return objects

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-nt", "--num_targets")
    parser.add_argument("-ns", "--num_steps")
    args = parser.parse_args()

    n_traj = int(args.num_targets)
    n_steps = int(args.num_steps)

    objects = create_crossing_scenario(n_traj, n_steps, 0.2, 50, 7, 3)

    for object in objects:
        object.plot()

    ground_truth_csv_from_cv_object(objects, n_steps)

    X_gt, t_birth, t_death = ground_truth_dat_from_cv_object(objects, n_steps, n_traj)
    np.savetxt(f"./X_gt_{n_traj}.dat", X_gt, delimiter=',')
    np.savetxt(f"./t_birth_{n_traj}.dat", t_birth)
    np.savetxt(f"./t_death_{n_traj}.dat", t_death)

    do_plot = True # Make arg

    if do_plot:

        plt.title(f"Manually generated test scenario with {n_traj} objects.")
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
