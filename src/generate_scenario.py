import os
import time
import datetime
import re
import shutil
from collections import deque
import argparse
import random

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from data_generation.data_generator import DataGenerator
from util.misc import save_checkpoint, update_logs
from util.load_config_files import load_yaml_into_dotdict
from util.plotting import compute_avg_certainty, get_constrastive_ax, get_false_ax, \
    get_total_loss_ax, get_state_uncertainties_ax
from util.logger import Logger
from modules.loss import MotLoss
from modules.contrastive_loss import ContrastiveLoss
from modules import evaluator
from modules.models.mt3v2.mt3v2 import MT3V2
from util.misc import NestedTensor

from matplotlib.patches import Ellipse

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


def plot_ground_truth(trajectories):
    cmap = plt.cm.get_cmap('nipy_spectral', len(trajectories[0])) 

    for color_idx, (track_id, track) in enumerate(trajectories[0].items()):
        Xgt, Ygt = get_xy_from_data(track)
        for i in range(len(Xgt)):
            if i == 0:
                plt.plot(Xgt[i:i+2], Ygt[i:i+2], 'o-',c=cmap(color_idx), alpha = i/len(Xgt), label=f"Track #{track_id}")
            else:
                plt.plot(Xgt[i:i+2], Ygt[i:i+2], 'o-',c=cmap(color_idx), alpha = i/len(Xgt))

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
    #print(tensor)
    max_len = max(list(map(len, tensor.tensors)))
    window = tuple([tensor.tensors.numpy()[0][offset:offset+window_size]])
    padded_window, mask = pad_to_batch_max(window, max_len)
    nested_tensor = NestedTensor(torch.Tensor(padded_window).to(torch.device(params.training.device)),
                                 torch.Tensor(mask).bool().to(torch.device(params.training.device)))
    
    return nested_tensor


    
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


    params.data_generation.seed = 2335718734 #params.general.pytorch_and_numpy_seed #random.randint(0, 9999999999999999999) # Tune this to get different runs
    do_plot_preds = True
    do_plot_ellipse = False
    do_plot_vel = True
    do_plot_missed_predictions = False

    @torch.no_grad()
    def output_truth_plot(ax, prediction, labels, matched_idx, batch, params, training_example_to_plot=0):

        assert hasattr(prediction, 'positions'), 'Positions should have been predicted for plotting.'
        assert hasattr(prediction, 'logits'), 'Logits should have been predicted for plotting.'
        if params.data_generation.prediction_target == 'position_and_velocity':
            assert hasattr(prediction, 'velocities'), 'Velocities should have been predicted for plotting.'

        bs, num_queries = prediction.positions.shape[:2]
        assert training_example_to_plot <= bs, "'training_example_to_plot' should be less than batch_size"

        if params.data_generation.prediction_target == 'position_and_shape':
            raise NotImplementedError('Plotting not working yet for shape predictions.')

        # Get ground-truth, predicted state, and logits for chosen training example
        truth = labels[training_example_to_plot].cpu().numpy()
        indices = tuple([t.cpu().detach().numpy() for t in matched_idx[training_example_to_plot]])
        if params.data_generation.prediction_target == 'position':
            out = prediction.positions[training_example_to_plot].cpu().detach().numpy()
        elif params.data_generation.prediction_target == 'position_and_velocity':
            pos = prediction.positions[training_example_to_plot]
            vel = prediction.velocities[training_example_to_plot]
            out = torch.cat((pos, vel), dim=1).cpu().detach().numpy()
        out_prob = prediction.logits[training_example_to_plot].cpu().sigmoid().detach().numpy().flatten()

        # Optionally get uncertainties for chosen training example
        if hasattr(prediction, 'uncertainties'):
            uncertainties = prediction.uncertainties[training_example_to_plot].cpu().detach().numpy()
        else:
            uncertainties = None

        once = True
        for i, posvel in enumerate(out):
            pos_x = posvel[0]
            pos_y = posvel[1]
            vel_x = posvel[2]
            vel_y = posvel[3]

            if i in indices[0]:

                gt_idx = indices[1][np.where(indices[training_example_to_plot] == i)[0][0]]
                gt_x = truth[gt_idx][0]
                gt_y = truth[gt_idx][1]
                gt_vx = truth[gt_idx][2]
                gt_vy = truth[gt_idx][3]

                if do_plot_preds:
                    if once:
                        p = ax.plot(pos_x, pos_y, marker='o', color='r', markersize=5, label="Latest pred")
                        color = p[0].get_color()
                        # Plot ground-truth
                        ax.plot(gt_x, gt_y, marker='D', color='g', markersize=5, label="Latest gt")
                        once = False
                    else:
                    
                        # Plot predicted positions
                        p = ax.plot(pos_x, pos_y, marker='o', color='r', markersize=5)
                        color = p[0].get_color()
                        # Plot ground-truth
                        ax.plot(gt_x, gt_y, marker='D', color='g', markersize=5)

                # Plot arrows that indicate velocities for each object
                if params.data_generation.prediction_target == 'position_and_velocity' and do_plot_vel:
                    ax.arrow(pos_x, pos_y, vel_x, vel_y, color=color, head_width=0.2, linestyle='--', length_includes_head=True)
                    ax.arrow(gt_x, gt_y, gt_vx, gt_vy, color=p[0].get_color(), head_width=0.2, length_includes_head=True)

                # Plot uncertainties (2-sigma ellipse)
                if uncertainties is not None and do_plot_ellipse:
                    ell_position = Ellipse(xy=(pos_x, pos_y), width=uncertainties[i, 0]*4, height=uncertainties[i, 1]*4,
                                        color=color, alpha=0.4)
                    ell_velocity = Ellipse(xy=(pos_x + vel_x, pos_y + vel_y), width=uncertainties[i, 2]*4,
                                        height=uncertainties[i, 3]*4, edgecolor=color, linestyle='--', facecolor='none')
                    ax.add_patch(ell_position)
                    ax.add_patch(ell_velocity)
            else:
                if do_plot_missed_predictions:
                    if once:
                        p = ax.plot(pos_x, pos_y, marker='*', color='k', label='Unmatched Predicted Object', markersize=5)
                        once = False
                    else:
                        p = ax.plot(pos_x, pos_y, marker='*', color='k', markersize=5)

                    if params.data_generation.prediction_target == 'position_and_velocity' and do_plot_vel:
                        ax.arrow(pos_x, pos_y, vel_x, vel_y, color='k', head_width=0.2, linestyle='--', length_includes_head=True)

    data_generator = DataGenerator(params)
    last_filename = "C:/Users/chris/MT3v2/task1/checkpoints/" + "checkpoint_gradient_step_999999"

    # Load model weights and pass model to correct device
    prev = params.data_generation.n_timesteps
    params.data_generation.n_timesteps = 20
    model = MT3V2(params)
    params.data_generation.n_timesteps = prev

    checkpoint = torch.load(last_filename, map_location=params.training.device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(torch.device(params.training.device))

    mot_loss = MotLoss(params)
    mot_loss.to(params.training.device)

    # Calculate all values used for plotting
    batch, labels, unique_ids, _, trajectories, true_measurements, false_measurements = data_generator.get_batch()

    r, theta = get_rtheta_from_data(true_measurements[0])
    Xt, Yt = get_xy_from_rtheta(*get_rtheta_from_data(true_measurements[0]))
    Xf, Yf = get_xy_from_rtheta(*get_rtheta_from_data(false_measurements[0]))

    #batch = sliding_window(batch, 0, 20) # Inflates covariances A LOT, why?? Seems like n_timesteps is not the same as the sequence length. Window should thus be set adaptively for 20 elements

    prediction, intermediate_predictions, encoder_prediction, aux_classifications, _ = model.forward(batch)
    loss_dict, indices = mot_loss.forward(labels, prediction, intermediate_predictions, encoder_prediction, loss_type=params.loss.type)


    fig, output_ax = plt.subplots() 
    color = [(1, 0, 0, max(a / len(Xt), 0.01)) for a in range(len(Xt))]
    output_ax.scatter(Xt, Yt, marker='x', c=color, zorder=np.inf, label="True Measurements")
    color = [(0, 0, 0, max(a / len(Xf), 0.01)) for a in range(len(Xf))]
    output_ax.scatter(Xf, Yf, marker='x', c=color, zorder=np.inf, label="False Measurements")

    plot_ground_truth(trajectories)

    output_ax.set_ylabel('North')
    output_ax.set_xlabel('East')
    output_ax.grid('on')
    output_ax.set_aspect('equal', 'box')

    output_truth_plot(output_ax, prediction, labels, indices, batch, params, 0)

    leg = output_ax.legend()
    for lh in leg.legendHandles: 
        lh.set_alpha(1)


    # Note: The ground truth tracks may have come from very early timesteps. As
    # such, when plotting the latest tracking estimates, only the active tracks
    # will get estimates. The green GT points overlaid GT tracks will show which
    # tracks were active at the last timestep.
    plt.show()


    # Plot all measurements together
    # # Plot xy position of measurements, alpha-coded by time
    # measurements = batch.tensors[0][~batch.mask[0]]
    # colors = np.zeros((measurements.shape[0], 4))
    # unique_time_values = np.array(sorted(list(set(measurements[:, 3].tolist()))))
    # def f(t):
    #     """Exponential decay for alpha in time"""
    #     idx = (np.abs(unique_time_values - t)).argmin()
    #     return 1/1.2**(len(unique_time_values)-idx)

    # colors[:, 3] = [f(t) for t in measurements[:, 3].tolist()]
    
    # measurements_cpu = measurements.cpu()
    # meas_r = measurements_cpu[:, 0]
    # meas_theta = measurements_cpu[:, 2]
    # meas_x = meas_r * np.cos(meas_theta)
    # meas_y = meas_r * np.sin(meas_theta)

    # output_ax.scatter(meas_x, meas_y, marker='x', c=colors, zorder=np.inf)
