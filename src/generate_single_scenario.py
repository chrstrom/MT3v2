import sys
import argparse
import os

import torch
import numpy as np

from data_generation.data_generator import DataGenerator
from util.load_config_files import load_yaml_into_dotdict, dotdict
from util.misc import NestedTensor

import matplotlib.pyplot as plt

from data_generation.data_generator import DataGenerator
from util.misc import save_checkpoint, update_logs, nested_tensor_from_tensor_list
from util.load_config_files import load_yaml_into_dotdict
from util.plotting import compute_avg_certainty, get_constrastive_ax, get_false_ax, \
    get_total_loss_ax, get_state_uncertainties_ax
from util.logger import Logger
from modules.loss import MotLoss
from modules.contrastive_loss import ContrastiveLoss
from modules import evaluator
from modules.models.mt3v2.mt3v2 import MT3V2
import copy


def clip_measurements(measurements, N, offset=0):
    """
    Clip measurements on the form [r, rdot, theta, t] to the time step
    range (offset, N + offset)
    """
    dt = 0.1
    measurements = measurements[0]
    clipped_measurements = []
    for measurement in measurements:
        if offset * dt < measurement[3] < (N + offset) * dt:
            clipped_measurements.append(measurement.tolist())

    return tuple(np.array([clipped_measurements]))


def clip_trajectories(trajectories, N, offset=0):
    """
    Clip ground truth data to the time step range (offset, N + offset)
    """
    dt = 0.1
    trajectories = trajectories[0]
    print(type(trajectories))

    clipped_trajectories = copy.deepcopy(trajectories)
    for track_id in trajectories:
        trajectory = trajectories[track_id]
        clipped_trajectory = []
        for i in range(len(trajectory)):
            if offset * dt < trajectory[i, 4] < (offset + N) * dt:
                clipped_trajectory.append(trajectory[i])

        clipped_trajectories[track_id] = clipped_trajectory

    clipped_trajectories = {k: v for k, v in clipped_trajectories.items() if v}

    return [clipped_trajectories]

        


def pad_to_batch_max(training_data, max_len):
    batch_size = len(training_data)
    d_meas = training_data[0].shape[1]
    training_data_padded = np.zeros((batch_size, max_len, d_meas))
    mask = np.ones((batch_size, max_len))
    for i, ex in enumerate(training_data):
        training_data_padded[i,:len(ex),:] = ex
        mask[i,:len(ex)] = 0

    return training_data_padded, mask


def sliding_window(tensor: NestedTensor, offset:int, window_size=20) -> NestedTensor:
    window = tuple([tensor.tensors.numpy()[0][offset:offset+window_size]])
    max_len = max(list(map(len, window)))
    padded_window, mask = pad_to_batch_max(window, max_len)
    nested_tensor = NestedTensor(torch.Tensor(padded_window).to(torch.device(params.training.device)),
                                 torch.Tensor(mask).bool().to(torch.device(params.training.device)))
    return nested_tensor

def get_xy_from_data(data):
    Xgt = []
    Ygt = []
    for point in data:
        Xgt.append(point[0])
        Ygt.append(point[1])

    return Xgt, Ygt

from typing import Tuple
def get_xy_from_rtheta(r: float, theta: float) -> Tuple[float, float]:
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    return x, y


def get_rtheta_from_data(data):
    r = []
    theta = []
    for point in data:
        r.append(point[0])
        theta.append(point[2])

    return r, theta

@torch.no_grad()
# When using the network as a tracker (fully trained), we will have a batch size of 1, and so training_idx_to_plot is omitted.
def output_truth_plot(prediction, labels, matched_idx, params, time=1):

    assert hasattr(prediction, 'positions'), 'Positions should have been predicted for plotting.'
    assert hasattr(prediction, 'logits'), 'Logits should have been predicted for plotting.'
    if params.data_generation.prediction_target == 'position_and_velocity':
        assert hasattr(prediction, 'velocities'), 'Velocities should have been predicted for plotting.'



    if params.data_generation.prediction_target == 'position_and_shape':
        raise NotImplementedError('Plotting not working yet for shape predictions.')

    truth = labels[0].cpu().numpy()
    indices = tuple([t.cpu().detach().numpy() for t in matched_idx[0]])

    pos = prediction.positions[0]
    vel = prediction.velocities[0]
    out = torch.cat((pos, vel), dim=1).cpu().detach().numpy()

    once = True
    for i, posvel in enumerate(out):
        pos_x = posvel[0]
        pos_y = posvel[1]

        if i in indices[0]:
            gt_idx = indices[1][np.where(indices[0] == i)[0][0]]
            #print(f"{i}, {gt_idx}")
            gt_x = truth[gt_idx][0]
            gt_y = truth[gt_idx][1]
            plt.plot(pos_x, pos_y, marker='o', color='r', markersize=5, alpha=time/params.data_generation.n_timesteps) # Plot predicted positions
            plt.plot(gt_x, gt_y, marker='D', color='g', markersize=5) # Plot ground-truth


def plot_measurements(true_measurements, false_measurements):
    r, theta = get_rtheta_from_data(true_measurements[0])
    Xm, Ym = get_xy_from_rtheta(r, theta)
    plt.scatter(Xm, Ym, c='r', marker='x', alpha = 0.75, label="true measurements")

    r, theta = get_rtheta_from_data(false_measurements[0])
    Xf, Yf = get_xy_from_rtheta(r, theta)
    plt.scatter(Xf, Yf, c='k', marker='x', alpha = 0.1, label="false measurements")

def plot_ground_truth(trajectories):
    cmap = plt.cm.get_cmap('nipy_spectral', len(trajectories[0])) 

    for color_idx, (track_id, track) in enumerate(trajectories[0].items()):
        Xgt, Ygt = get_xy_from_data(track)
        for i in range(len(Xgt)):
            plt.plot(Xgt[i:i+2], Ygt[i:i+2], 'o-',c=cmap(color_idx), alpha = i/len(Xgt))




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-tp', '--task_params', help='filepath to configuration yaml file defining the task', required=True)
    parser.add_argument('-mp', '--model_params', help='filepath to configuration yaml file defining the model', required=True)
    parser.add_argument('--continue_training_from', help='filepath to folder of an experiment to continue training from')
    parser.add_argument('--exp_name', help='Name to give to the results folder')
    args = parser.parse_args()
    print(f'Task configuration file: {args.task_params}')
    print(f'Model configuration file: {args.model_params}')

    # Load hyperparameters
    params = load_yaml_into_dotdict(args.task_params)
    params.update(load_yaml_into_dotdict(args.model_params))
    eval_params = load_yaml_into_dotdict(args.task_params)
    eval_params.update(load_yaml_into_dotdict(args.model_params))
    eval_params.recursive_update(load_yaml_into_dotdict('configs/eval/default.yaml'))
    # Generate 32-bit random seed, or use user-specified one
    random_data = os.urandom(4)
    params.general.pytorch_and_numpy_seed = 2335718734 #int.from_bytes(random_data, byteorder="big")
    print(f'Using seed: {params.general.pytorch_and_numpy_seed}')

    # Seed pytorch and numpy for reproducibility
    torch.manual_seed(params.general.pytorch_and_numpy_seed)
    torch.cuda.manual_seed_all(params.general.pytorch_and_numpy_seed)
    np.random.seed(params.general.pytorch_and_numpy_seed)

    if params.training.device == 'auto':
        params.training.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if eval_params.training.device == 'auto':
        eval_params.training.device = 'cuda' if torch.cuda.is_available() else 'cpu'


    params.data_generation.seed = params.general.pytorch_and_numpy_seed #random.randint(0, 9999999999999999999) # Tune this to get different runs

    #params.data_generation.n_timesteps = 20
    params.training.batch_size = 1


    # model = MT3V2(params)
    # mot_loss = MotLoss(params)
    # mot_loss.to(params.training.device)

    # last_filename = "C:/Users/chris/MT3v2/task1/checkpoints/" + "checkpoint_gradient_step_999999"
    # checkpoint = torch.load(last_filename, map_location=params.training.device)
    # model.load_state_dict(checkpoint['model_state_dict'])

    # model.to(torch.device(params.training.device))
    data_gen = DataGenerator(params)

    #params.data_generation.n_timesteps = 40
    # Training tensor has rows that contains [r, rdot, theta, time]
    training_tensor, labels, unique_ids, unique_measurement_ids, trajectories, true_measurements, false_measurements = data_gen.get_batch()

    offset = 50
    N = 50
    trajectories = clip_trajectories(trajectories, N, offset)
    true_measurements = clip_measurements(true_measurements, N, offset)
    false_measurements = clip_measurements(false_measurements, N, offset)

    plot_measurements(true_measurements, false_measurements)
    plot_ground_truth(trajectories)


    # window_size = 20
    # for k in range(params.data_generation.n_timesteps - window_size):
    #     training_nested_tensor = sliding_window(training_tensor, k, window_size)
    #     prediction, intermediate_predictions, encoder_prediction, aux_classifications, _ = model.forward(training_nested_tensor)
    #     loss_dict, indices = mot_loss.forward(labels, prediction, intermediate_predictions, encoder_prediction, loss_type=params.loss.type)
    #     time = params.data_generation.n_timesteps - k
    #     params.data_generation.n_prediction_lag = k
    #     output_truth_plot(prediction, labels, indices, params, time)


    plt.xlabel("East [m]")
    plt.ylabel("North [m]")
    plt.legend()
    plt.show()
  