import os
import time
import argparse
import random
import copy

import torch

import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple

from data_generation.data_generator import DataGenerator
from util.load_config_files import load_yaml_into_dotdict
from modules.loss import MotLoss

from modules.models.mt3v2.mt3v2 import MT3V2
from util.misc import NestedTensor

from matplotlib.patches import Ellipse



# TODO: Create interface for user-generated measurement generation
# TODO: Test specific scenario with different levels of measurement noise

# TODO: Should I focus on object complexity or noise complexity? (or both..)


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

def clip_batch(tensor: NestedTensor, N: int, offset=0) -> NestedTensor:
    measurements = tuple([tensor.tensors.numpy()[0]])

    clipped_measurements = clip_measurements(measurements, N, offset)
    max_len = max(list(map(len, clipped_measurements)))

    # if len(clipped_measurements[0]) == 0:
    #     return None

    padded_window, mask = pad_to_batch_max(clipped_measurements, max_len)
    nested_tensor = NestedTensor(torch.Tensor(padded_window).to(torch.device(params.training.device)),
                                 torch.Tensor(mask).bool().to(torch.device(params.training.device)))
    return nested_tensor

def clip_trajectories(trajectories, N, offset=0):
    """
    Clip ground truth data to the time step range (offset, N + offset)
    """
    dt = 0.1
    trajectories = trajectories[0]

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

def get_xy_from_data(data):
    Xgt = []
    Ygt = []
    for point in data:
        Xgt.append(point[0])
        Ygt.append(point[1])

    return Xgt, Ygt

def get_vxvy_from_data(data):
    VXgt = []
    VYgt = []
    for point in data:
        VXgt.append(point[2])
        VYgt.append(point[3])

    return VXgt, VYgt

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

def plot_ground_truth(trajectories, prior_lengths, n_historical_steps):
    cmap = plt.cm.get_cmap('nipy_spectral', len(trajectories[0])) 

    posterior_lengths = prior_lengths
    # TODO: Check if the current track has the current timestamp in it. If not, do not plot
    for color_idx, (track_id, track) in enumerate(trajectories[0].items()):
        Xgt, Ygt = get_xy_from_data(track)

        Xgt_size = posterior_lengths.get(track_id)
        if Xgt_size != Xgt[-1] and Xgt_size != -1: # Hack to ensure that only active tracks are drawn
            posterior_lengths[track_id] = Xgt[-1]
            n_hist = min(len(Xgt), n_historical_steps)
            for i in range(n_hist):
                if i == 0:
                    plt.plot(Xgt[-i-3:-i-1], Ygt[-i-3:-i-1], 'o-', c=cmap(color_idx), alpha = 1 - i/n_hist, label=f"Track #{track_id}")
                else:
                    plt.plot(Xgt[-i-3:-i-1], Ygt[-i-3:-i-1], 'o-', c=cmap(color_idx), alpha = 1 - i/n_hist)

            VXgt, VYgt = get_vxvy_from_data(track)

            # Plot GT diamond and velocity arrow
            plt.arrow(Xgt[-1], Ygt[-1], VXgt[-1], VYgt[-1], color='g', head_width=0.2, length_includes_head=True)
            plt.plot(Xgt[-1], Ygt[-1], marker='D', color='g', markersize=5, label="Latest gt")

        else:
            posterior_lengths[track_id] = -1

        # n_hist = min(len(Xgt), n_historical_steps)
        # for i in range(n_hist):
        #     plt.plot(Xgt[-i-3:-i-1], Ygt[-i-3:-i-1], 'o-', c=cmap(color_idx), alpha = 1 - i/n_hist)

        # VXgt, VYgt = get_vxvy_from_data(track)

        # # Plot GT diamond and velocity arrow
        # plt.arrow(Xgt[-1], Ygt[-1], VXgt[-1], VYgt[-1], color='g', head_width=0.2, length_includes_head=True)
        # plt.plot(Xgt[-1], Ygt[-1], marker='D', color='g', markersize=5, label="Latest gt")
        
    return posterior_lengths
        
def plot_measurements(true_measurements, false_measurements):
    r, theta = get_rtheta_from_data(true_measurements[0])
    Xm, Ym = get_xy_from_rtheta(r, theta)
    plt.scatter(Xm, Ym, c='r', marker='x', alpha = 0.75, label="true measurements")

    r, theta = get_rtheta_from_data(false_measurements[0])
    Xf, Yf = get_xy_from_rtheta(r, theta)
    plt.scatter(Xf, Yf, c='k', marker='x', alpha = 0.75, label="false measurements")

def pad_to_batch_max(training_data, max_len):
    batch_size = len(training_data)
    d_meas = training_data[0].shape[1]
    training_data_padded = np.zeros((batch_size, max_len, d_meas))
    mask = np.ones((batch_size, max_len))
    for i, ex in enumerate(training_data):
        training_data_padded[i,:len(ex),:] = ex
        mask[i,:len(ex)] = 0
    return training_data_padded, mask

# This should not index directly based on window_size, instead find all measurements with timestamp inside range, and plot.
def sliding_window(tensor: NestedTensor, offset:int, window_size=20) -> NestedTensor:
    #print(tensor)
    max_len = max(list(map(len, tensor.tensors)))
    window = tuple([tensor.tensors.numpy()[0][offset:offset+window_size]])
    padded_window, mask = pad_to_batch_max(window, max_len)
    nested_tensor = NestedTensor(torch.Tensor(padded_window).to(torch.device(params.training.device)),
                                 torch.Tensor(mask).bool().to(torch.device(params.training.device)))
    
    return nested_tensor

@torch.no_grad()
def output_truth_plot(ax, prediction, labels, indices, batch, params):

    assert hasattr(prediction, 'positions'), 'Positions should have been predicted for plotting.'
    assert hasattr(prediction, 'logits'), 'Logits should have been predicted for plotting.'
    if params.data_generation.prediction_target == 'position_and_velocity':
        assert hasattr(prediction, 'velocities'), 'Velocities should have been predicted for plotting.'

    bs, num_queries = prediction.positions.shape[:2]

    if params.data_generation.prediction_target == 'position_and_shape':
        raise NotImplementedError('Plotting not working yet for shape predictions.')

    # Get ground-truth, predicted state, and logits for chosen training example
    #print(matched_idx)
    truth = labels[0].cpu().numpy()
    #indices = matched_idx[0] #tuple([t.cpu().detach().numpy() for t in matched_idx[0]])

    if params.data_generation.prediction_target == 'position':
        out = prediction.positions[0].cpu().detach().numpy()
    elif params.data_generation.prediction_target == 'position_and_velocity':
        pos = prediction.positions[0]
        vel = prediction.velocities[0]
        out = torch.cat((pos, vel), dim=1).cpu().detach().numpy()
    out_prob = prediction.logits[0].cpu().sigmoid().detach().numpy().flatten()

    # Optionally get uncertainties for chosen training example
    if hasattr(prediction, 'uncertainties'):
        uncertainties = prediction.uncertainties[0].cpu().detach().numpy()
    else:
        uncertainties = None

    once = True
    for i, posvel in enumerate(out):
        pos_x = posvel[0]
        pos_y = posvel[1]
        vel_x = posvel[2]
        vel_y = posvel[3]

        

        if i in indices:
            if do_plot_preds:
                if once:
                    p = ax.plot(pos_x, pos_y, marker='o', color='r', markersize=5, label="Latest pred")
                    once = False
                else:                
                    # Plot predicted positions
                    p = ax.plot(pos_x, pos_y, marker='o', color='r', markersize=5)
                
                color = p[0].get_color()

            # Plot arrows that indicate velocities for each object
            if params.data_generation.prediction_target == 'position_and_velocity' and do_plot_vel:
                ax.arrow(pos_x, pos_y, vel_x, vel_y, color=color, head_width=0.2, linestyle='--', length_includes_head=True)

            # Plot uncertainties (2-sigma ellipse)
            if uncertainties is not None and do_plot_ellipse:
                ell_position = Ellipse(xy=(pos_x, pos_y), width=uncertainties[i, 0]*4, height=uncertainties[i, 1]*4,
                                    color=color, alpha=0.4)
                ell_velocity = Ellipse(xy=(pos_x + vel_x, pos_y + vel_y), width=uncertainties[i, 2]*4,
                                    height=uncertainties[i, 3]*4, edgecolor=color, linestyle='--', facecolor='none')
                ax.add_patch(ell_position)
                ax.add_patch(ell_velocity)
        else:
            if do_plot_unmatched_hypotheses:
                if once:
                    p = ax.plot(pos_x, pos_y, marker='*', color='k', label='Unmatched Predicted Object', markersize=5)
                    once = False
                else:
                    p = ax.plot(pos_x, pos_y, marker='*', color='k', markersize=5)

                if params.data_generation.prediction_target == 'position_and_velocity' and do_plot_vel:
                    ax.arrow(pos_x, pos_y, vel_x, vel_y, color='k', head_width=0.2, linestyle='--', length_includes_head=True)


def step_once(event):
    global timestep
    global prior_lengths
    global N
    global K
    global M
    global gospa_total
    global gospa_loc
    global gospa_miss
    global gospa_false

    # Clip all data to a sliding window
    clipped_trajectories = clip_trajectories(trajectories, N, timestep)
    clipped_measurements = clip_batch(measurements, N, timestep)

    prediction, intermediate_predictions, encoder_prediction, aux_classifications, _ = model.forward(clipped_measurements, timestep)
    output_existence_probabilities = prediction.logits.sigmoid().detach().cpu().numpy().flatten()

    existence_threshold = 0.75 # Hyperparameter
    alive_indices = np.where(output_existence_probabilities > existence_threshold)[0]

    # Plot results
    clipped_true_measurements = clip_measurements(true_measurements, M, timestep + N - M + 1)
    clipped_false_measurements = clip_measurements(false_measurements, M, timestep + N - M + 1)

    plot_measurements(clipped_true_measurements, clipped_false_measurements)
    prior_lengths = plot_ground_truth(clipped_trajectories, prior_lengths, 10)
    output_truth_plot(output_ax, prediction, labels, alive_indices, clipped_measurements, params)

    # prediction_in_format_for_loss = {'state': torch.cat((prediction.positions, prediction.velocities), dim=2),
    #                                     'logits': prediction.logits,
    #                                     'state_covariances': prediction.uncertainties ** 2}
    # loss, indices, decomposition = mot_loss.compute_orig_gospa_matching(prediction_in_format_for_loss, labels, existence_threshold=0.0)

    # gospa_total.append(loss.item())
    # gospa_loc.append(decomposition['localization'])
    # # gospa_norm_loc.append(decomposition['localization'] / decomposition['n_matched_objs'] if \
    # #     decomposition['n_matched_objs'] != 0 else 0.0)
    # gospa_miss.append(decomposition['missed'])
    # gospa_false.append(decomposition['false'])

    output_ax.set_aspect('equal', 'box')
    output_ax.set_ylabel('North')
    output_ax.set_xlabel('East')
    output_ax.grid('on')
    output_ax.set_xlim([-10, 20]) 
    output_ax.set_ylim([-15, 15]) 

    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.1)
    output_ax.clear()
    timestep += 1


def update_labels(trajectories, timestep):
    dt = 0.1
    t = timestep * dt
    
    # TODO: LABELS MUST BE UPDATED FOR EVERY TIME STEP
    #       Labels take the form of 
    #[tensor([[x1, y1, vx1, vy1],
    #              ...
    #        [ xn, yn, vxn, vyn]])]
    # For all ground truth objects AT THE CURRENT TIME STEP (1.9s )

    # Q: Can the labels be circumvented by simply plotting all predictions with a certain existence probability?
    # A: Yes! but..
        # Current plotting just takes all measurements above a certain existence threshold and 
        # displays them. However, GOSPA depends on labels, and to assess performance, the labels
        # need to be updated.











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



    # 2015188077: Single crossing
    # 1782001962: Chaos but good
    # 1261187305: Performance in the horizontal direction
    
    params.general.pytorch_and_numpy_seed = 1798125999 #int.from_bytes(random_data, byteorder="big")
    print(f'Using seed: {params.general.pytorch_and_numpy_seed}')

    # Seed pytorch and numpy for reproducibility
    torch.manual_seed(params.general.pytorch_and_numpy_seed)
    torch.cuda.manual_seed_all(params.general.pytorch_and_numpy_seed)
    np.random.seed(params.general.pytorch_and_numpy_seed)

    if params.training.device == 'auto':
        params.training.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if eval_params.training.device == 'auto':
        eval_params.training.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    params.data_generation.seed = params.general.pytorch_and_numpy_seed #2335718734 #random.randint(0, 9999999999999999999) # Tune this to get different runs
    do_plot_preds = True
    do_plot_ellipse = True
    do_plot_vel = True
    do_plot_unmatched_hypotheses = True

    data_generator = DataGenerator(params)
    last_filename = "C:/Users/chris/MT3v2/task1/checkpoints/" + "checkpoint_gradient_step_999999"

    # Load model weights and pass model to correct device
    prev = params.data_generation.n_timesteps
    params.data_generation.n_timesteps = 20 # Hack to load the pretrained model and still generate more than 20 timesteps
    model = MT3V2(params)
    
    checkpoint = torch.load(last_filename, map_location=params.training.device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(torch.device(params.training.device))

    mot_loss = MotLoss(eval_params)
    mot_loss.to(params.training.device)
    model.eval()
    #mot_loss = MotLoss(eval_params)

    params.data_generation.n_timesteps = prev
    # Calculate all values used for plotting

    measurements, labels, unique_ids, _, trajectories, true_measurements, false_measurements = data_generator.get_batch()
    print(labels)
    #print(trajectories)
    #print(labels)

    do_interactive = False
    if not do_interactive:
        plt.ion() # Required for dynamic plot updates
    fig, output_ax = plt.subplots() 

    K = 10 # amount of gt in the past to plot.
    N = 20 # Track length to consider. This cannot be longer than 20 since the learned position encoding is fixed in size
    M = 1 # amount of measurements in the past to plot.

    prior_lengths = {}
    timestep = 0
    gospa_total = []
    gospa_loc = []
    gospa_miss = []
    gospa_false = []

    # Batch includes all MEASUREMENTS, and thus the window should gate measurements on their timestep, not the amount of measurements.
 
    # Note: The ground truth tracks may have come from very early timesteps. As
    # such, when plotting the latest tracking estimates, only the active tracks
    # will get estimates. The green GT points overlaid GT tracks will show which
    # tracks were active at the last timestep.

    # TODO: How to test for active hypotheses for each time step. I.e. is there a probability array that can be printed?


    if do_interactive:
        try:
            fig.canvas.mpl_connect('key_press_event', step_once)
            leg = output_ax.legend()
            for lh in leg.legendHandles: 
                lh.set_alpha(1)

            plt.show()
        except RuntimeError as e:
            exit()
    else:
        for _ in range(30):
            # TODO: Add pause function
            
            step_once(None)
            time.sleep(0.5)

        # Turn off interactive and keep plot open
        plt.ioff()
        #plt.show()

    # total_gospa = np.array(gospa_total)
    # total_loc = np.array(gospa_loc)
    # total_miss = np.array(gospa_miss)
    # total_false = np.array(gospa_false)

    # import scipy.stats as st

    # print(f"GOSPA: {total_gospa.mean()} +/- {max(st.norm.interval(0.95, loc=0, scale=st.sem(total_gospa)))}")
    # print(f"LOC: {total_loc.mean()} +/- {max(st.norm.interval(0.95, loc=0, scale=st.sem(total_loc)))}")
    # print(f"MISS: {total_miss.mean()} +/- {max(st.norm.interval(0.95, loc=0, scale=st.sem(total_miss)))}")
    # print(f"FALSE: {total_false.mean()} +/- {max(st.norm.interval(0.95, loc=0, scale=st.sem(total_false)))}")
