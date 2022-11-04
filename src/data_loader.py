"""
    TODO: Recreate the Mahler N crossing target scenario, found in the paper
          https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5744132.

    TODO: Create loader for the 9 ravens scenario from Brekke
"""

import argparse
import random
import torch

import numpy as np
import pandas as pd

from util.load_config_files import load_yaml_into_dotdict



def measurement_from_gt(ground_truth, add_noise = False, params = None):
    x = ground_truth[:, 1]
    y = ground_truth[:, 2]
    vx = ground_truth[:, 3]
    vy = ground_truth[:, 4]
    t = ground_truth[:, 5]

    range_m = np.sqrt(x*x + y*y)
    bearing = np.arctan2(y, x)
    range_rate = (x * vx + y * vy) / range_m 

    steps_max = int(np.max(t) / 0.1) + 1

    if add_noise:
        assert params != None, "Parameter file needs to be specified when adding noise!"
        noise = params.data_generation.measurement_noise_stds

        range_m = np.random.normal(range_m, noise[0])
        bearing = np.random.normal(bearing, noise[1])
        range_rate = np.random.normal(range_rate, noise[2])

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
                                                 bearing[step], 
                                                 range_rate[step],
                                                 np.round(t[step], 3)))
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

    # drop true measurements
        # Draw from uniform. If under p_meas, do not add a true measurement
    # combine shuffled elements at a given timestep
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



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--scenario")
    parser.add_argument("-tp", "--task_params")
    args = parser.parse_args()
    params = load_yaml_into_dotdict(args.task_params)

    dt = 0.1

    gt = pd.read_csv(args.scenario, delimiter=',') # assume: [id, x, y, vx, vy, t]

    # NOTE: True and false measurements are a 3d array, where the first index
    # can be used to retrieve all measurements at a given timestep
    # Each invidivdual element is on the form [r, theta, rdot, t]
    ids, true_measurements, steps_max = measurement_from_gt(gt.values, True, params)
    false_measurements = generate_false_measurements(steps_max, params)

    measurements = generate_measurement_set(true_measurements, false_measurements, steps_max, params)

    tensor = tensor_from_measurements(measurements, steps_max)
    print(tensor)
