import numpy as np

from constant_velocity_object import CVObject

def cv_object_from_r_theta(r, theta, id, sigma):
    return CVObject(np.array((-r * np.cos(theta), -r * np.sin(theta))),
                    np.array(((r/5)*np.cos(theta), (r/5)*np.sin(theta))),
                    id, sigma)
    

def ground_truth_csv_from_cv_object(objects, n_steps, title = ""):

    track_sorted_by_time = []
    for step in range(n_steps):
        tracks = []
        
        for object in objects:
            tracks.append(object.track[step].tolist())
        
        track_sorted_by_time.append(tracks)

    if title == "":
        title = f"{len(objects)}tracks"
    with open(title + ".csv", "w", newline="") as f:
        writer = csv.writer(f)
        for step in track_sorted_by_time:

            writer.writerows(step)

def ground_truth_dat_from_cv_object(objects, n_steps, n_traj):

    X_gt = np.empty((n_traj*4, n_steps))

    for step in range(n_steps):
        m_at_step = []
        for object in objects:
            m_at_step += [object.track[step][1]]
            m_at_step += [object.track[step][3]]
            m_at_step += [object.track[step][2]]
            m_at_step += [object.track[step][4]]

        X_gt[:, step] = m_at_step

    t_birth = np.ones(n_traj)
    t_death = n_steps * np.ones(n_traj)

    return X_gt, t_birth, t_death
        