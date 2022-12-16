"""
TODO: Recreate the Mahler N crossing target scenario, found in the paper
        https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5744132.

TODO: Create loader for the 9 ravens scenario from Brekke

For scenario generation: Create a linear function for each track, then sample along the track.
Use constant velocity for all objects. Different velocities, but such that they all meet at the center
Use the same center as the spawning region in the original data generator.

Run predictions using the parameters from all tasks.
"""

import argparse
import random
import torch

import scipy.io as io

import numpy as np
import pandas as pd

import regex as re

from util.load_config_files import load_yaml_into_dotdict

def measurement_from_gt(ground_truth, add_noise = False, params = None, starting_time = 0.0):
    x = ground_truth[:, 1]
    y = ground_truth[:, 2]
    vx = ground_truth[:, 3]
    vy = ground_truth[:, 4]
    t = ground_truth[:, 5]

    range_m = np.sqrt(x*x + y*y)
    range_m[range_m < 1e-8] = 1e-8 # Avoid div by zero
    bearing = np.arctan2(y, x)
    range_rate = (x * vx + y * vy) / range_m 

    steps_max = int(np.max(t) / 0.1) + 1

    if add_noise:
        assert params != None, "Parameter file needs to be specified when adding noise!"
        noise = params.data_generation.measurement_noise_stds

        range_m = np.random.normal(range_m, noise[0])
        range_rate = np.random.normal(range_rate, noise[1])
        bearing = np.random.normal(bearing, noise[2])

    # Find number of occurences of each timestep, used in order to put measurements in a 3D array, indexed by timestep
    times = np.zeros(steps_max)
    for step in range(steps_max):
        times[step] = (t == np.round(step * 0.1, 3)).sum()

    step = 0
    true_measurements = []
    for i in times:
        gt_at_time_t = []
        for _ in range(int(i)):
            gt_at_time_t.append(np.column_stack((range_m[step],
                                                 range_rate[step],
                                                 bearing[step],
                                                 np.round(t[step] + starting_time, 3)))
                                                 [0].tolist())
            step += 1
        true_measurements.append(gt_at_time_t)
        

    return ground_truth[:, 0], true_measurements, steps_max

def generate_false_measurements(n, params):
        config = params.data_generation
        fov = config.field_of_view
        false_measurements = []
        for t in range(n):
            # Generate false measurements (uniformly distributed over measurement FOV)
            n_false_measurements = np.random.poisson(config.n_avg_false_measurements)
            false_measurement = \
                np.random.uniform(low=[fov.min_range, -fov.max_range_rate, fov.min_theta],
                                high=[fov.max_range, fov.max_range_rate, fov.max_theta],
                                size=(n_false_measurements, 3))

            time = np.repeat(np.round(t * 0.1, 3), n_false_measurements)
            false_measurements.append(np.column_stack((false_measurement, time)).tolist())

        return false_measurements


def get_measurement_at_step(step, measurements):
    return measurements[step]

def generate_measurement_set(true_measurements, false_measurements, steps_max, params):
    """
    Generate the final measurement set by combining true and false measurements
    """
    measurements = []

    for step in range(steps_max):
        true_at_step = get_measurement_at_step(step, true_measurements)
        false_at_step = get_measurement_at_step(step, false_measurements)

        # Drop true measurements according to the detection probability

        detected_true_at_step = []
        for m in true_at_step:
            if random.uniform(0, 1) < params.data_generation.p_meas:
                detected_true_at_step.append(m)

        measurement = detected_true_at_step + false_at_step
        random.shuffle(measurement) # Important to shuffle so the network doesnt learn that the first elements are true and the other are false measurements

        measurements.append(measurement)

    return measurements


def tensor_from_measurements(measurements, steps_max):
    """
    Take measurements and match the MT3v2 input format
    """

    tensor_out = torch.tensor([])

    for step in range(steps_max):
        measurements_at_step = measurements[step]
        for measurement in measurements_at_step:
                tensor_out = torch.cat((tensor_out, torch.tensor([measurement])))

    
    tensor_out = tensor_out[None, :, :] # Add dummy dimension to adhere to MT3v2
    return tensor_out


def tensor_from_gt(gt, params, warmup_steps = 0):
    """
    Turn a ground-truth sequence into a tensor of measurements that is compatible
    with MT3, using the clutter intensity and radar model as specified by the
    task parameters.
    """
    ids, true_measurements, steps_max = measurement_from_gt(gt.values, True, params)
    false_measurements = generate_false_measurements(steps_max + warmup_steps, params)

    measurements = generate_measurement_set(true_measurements, false_measurements, steps_max, params)

    tensor = tensor_from_measurements(measurements, steps_max)

    true_measurements_out = []
    for measurement in true_measurements:
        for target in measurement:
            true_measurements_out.append(target)

    false_measurements_out = []
    for measurement in false_measurements:
        for target in measurement:
            false_measurements_out.append(target)

    true_measurements_out = np.array((true_measurements_out))
    false_measurements_out = np.array((false_measurements_out))

    trajectory = {}

    for target in gt.values:
        id = int(target[0])
        single_trajectory = target[1:].tolist()

        if id in trajectory:
            trajectory |=  {id: trajectory[id] + [single_trajectory]}
        else:
            trajectory[id] = [single_trajectory]

    for key, value in trajectory.items():
        trajectory[key] = np.array(value)

        

    return tensor, (true_measurements_out,), (false_measurements_out,), [trajectory]


def label_at_step_from_gt(gt, step):
    time = np.round(step * 0.1, 3)

    labels = []
    for data in gt.values:
        if data[5] == time:
            labels.append(data[1:5].tolist())
    
    label_tensor = torch.tensor(labels)

    return [label_tensor]

def label_at_time_from_gt(gt, time):
    time = np.round(time, 3)

    labels = []
    for data in gt.values:
        if data[5] == time:
            labels.append(data[1:5].tolist())
    
    label_tensor = torch.tensor(labels)

    return [label_tensor]


def to_matlab(tensor):
    """
    Angels Matlab PMBM implementation uses z = [cell, cell, ..., cell]
    where each cell is a 2xN array of measurements in x, y for that step
    """

    tensor = tensor[0]
    t_max = tensor[-1].cpu().detach().numpy()[-1]
    n_max = int(t_max / 0.1)

    z_matlab = []

    for step in range(n_max):
        time = np.float32(np.round(step * 0.1, 1))
        z_at_time = []
        for measurement in tensor:
            measurement = measurement.cpu().detach().numpy()
            time_at_measurement = np.round(measurement[-1], 1)

            if time_at_measurement == time:
                z_range = measurement[0]
                z_bearing = measurement[2]
                x = z_range * np.cos(z_bearing)
                y = z_range * np.sin(z_bearing)
                z_at_time.append([x, y])
        z_matlab.append(z_at_time)

    return z_matlab

def to_matlab_range_bearing(tensor):
    """
    Angels Matlab PMBM implementation uses z = [cell, cell, ..., cell]
    where each cell is a 2xN array of measurements in x, y for that step
    """

    tensor = tensor[0]
    t_max = tensor[-1].cpu().detach().numpy()[-1]
    n_max = int(t_max / 0.1)

    z_matlab = []

    for step in range(n_max):
        time = np.float32(np.round(step * 0.1, 1))
        z_at_time = []
        for measurement in tensor:
            measurement = measurement.cpu().detach().numpy()
            time_at_measurement = np.round(measurement[-1], 1)

            if time_at_measurement == time:
                z_range = measurement[0]
                z_bearing = measurement[2]
                c = np.cos(z_bearing)
                s = np.sin(z_bearing)
                z_at_time.append([c, s, z_range])
        z_matlab.append(z_at_time)

    return z_matlab


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--scenario")
    parser.add_argument("-tp", "--task_params")
    parser.add_argument("-nc", "--num_runs")
    args = parser.parse_args()
    params = load_yaml_into_dotdict(args.task_params)
    if args.num_runs is None:
        num_runs = 1
    else:
        num_runs = int(args.num_runs)


    gt = pd.read_csv(args.scenario, delimiter=',') # assume: [id, x, y, vx, vy, t]


    #matlab_tensor = to_matlab(tensor)
    #io.savemat('Z_matlab.mat', {'Z': matlab_tensor})  
    num = re.search(r'\d+', args.scenario).group()
    ts = re.search(r'\d+', args.task_params).group()
    for i in range(num_runs):
        tensor, _, _, traj = tensor_from_gt(gt, params)
        matlab_tensor = to_matlab_range_bearing(tensor)
        io.savemat(f'Z_range_bearing_{num}_tracks_task_{ts}_{i+1}.mat', {'Z': matlab_tensor})   
    
