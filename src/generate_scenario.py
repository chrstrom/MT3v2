import os
import time
import argparse
import random
import copy
from tqdm import tqdm

import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Tuple
from operator import add

from data_generation.data_generator import DataGenerator
from util.load_config_files import load_yaml_into_dotdict
from modules.loss import MotLoss

from modules.models.mt3v2.mt3v2 import MT3V2
from util.misc import NestedTensor, nested_tensor_from_tensor_list
from matplotlib.lines import Line2D

from matplotlib.patches import Ellipse

from data_loader import tensor_from_gt, label_at_step_from_gt

def clip_measurements(measurements, N, offset=0):
    """
    Clip measurements on the form [r, rdot, theta, t] to the time step
    range (offset, N + offset)
    """
    dt = 0.1
    measurements = measurements[0]
    clipped_measurements = []
    for measurement in measurements:

        if offset * dt < measurement[3] <= (N + offset) * dt:
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

def plot_ground_truth(trajectories, prior_lengths, n_historical_steps, at_time):
    cmap = plt.cm.get_cmap('brg', len(trajectories[0])) 

    posterior_lengths = prior_lengths
    times = []
    for color_idx, (track_id, track) in enumerate(trajectories[0].items()):
        times.append(track[-1][4])
        
    max_time = np.round(max(times)-1.9, 3)
    # TODO: Check if the current track has the current timestamp in it. If not, do not plot
    for color_idx, (track_id, track) in enumerate(trajectories[0].items()):
        Xgt, Ygt = get_xy_from_data(track)
        current_time = np.round(track[-1][4] - 1.9, 3)
        

        Xgt_size = posterior_lengths.get(track_id)

        #if Xgt_size != Xgt[-1] and Xgt_size != -1: # Hack to ensure that only active tracks are drawn
        if current_time == max_time:
            posterior_lengths[track_id] = Xgt[-1]
            n_hist = min(len(Xgt), n_historical_steps)

            alphas = np.exp(5*np.linspace(0.8, 1, n_hist)) # The constant multiplied here is the rate of decay (higher = more decay)
            alphas = alphas / np.max(alphas)
            alphas = np.flip(alphas)

            for i in range(n_hist):
                    plt.plot(Xgt[-i-3:-i-1], Ygt[-i-3:-i-1], '-', c=cmap(color_idx),  alpha = alphas[i], markersize=5)

            VXgt, VYgt = get_vxvy_from_data(track)

            # Plot GT diamond and velocity arrow
            try:
                plt.plot(Xgt[-2], Ygt[-2], marker='D', color='g', markersize=5) #, label="Latest gt")
            except Exception as e:
                plt.plot(Xgt[-1], Ygt[-1], marker='D', color='g', markersize=5)#, label="Latest gt")
            if do_plot_vel:
                plt.arrow(Xgt[-1], Ygt[-1], VXgt[-1], VYgt[-1], color='g', head_width=0.2, length_includes_head=True)


        else:
            posterior_lengths[track_id] = -1

        
    return posterior_lengths

def scatter_and_decay(measurements, color="k", marker=".", label="", exponential=True):
    r, theta = get_rtheta_from_data(measurements[0])
    X, Y = get_xy_from_rtheta(r, theta)
    t = np.array((measurements[0]))[:, 3]

    timesteps = len(np.unique(t))
    starting_time = np.min(t)
    if exponential:
        alphas = np.exp(3*np.linspace(0.1, 1, timesteps)) # The constant multiplied here is the rate of decay (higher = more decay)
        alphas = alphas / np.max(alphas)

    else:
        alphas = np.linspace(0.1, 1, timesteps)


    for i in range(timesteps):
        index_for_time = np.where(t == np.round(starting_time + i * 0.1, 3))[0]
        if i == 0:
            plt.scatter(X[index_for_time], Y[index_for_time], c=color, marker=marker, alpha = alphas[i], s=20)# label=label, s=20)
        else:
            plt.scatter(X[index_for_time], Y[index_for_time], c=color, marker=marker, alpha = alphas[i], s=20)

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
    max_len = max(list(map(len, tensor.tensors)))
    window = tuple([tensor.tensors.numpy()[0][offset:offset+window_size]])
    padded_window, mask = pad_to_batch_max(window, max_len)
    nested_tensor = NestedTensor(torch.Tensor(padded_window).to(torch.device(params.training.device)),
                                 torch.Tensor(mask).bool().to(torch.device(params.training.device)))
    
    return nested_tensor

@torch.no_grad()
def output_truth_plot(ax, prediction, indices, batch, params):

    assert hasattr(prediction, 'positions'), 'Positions should have been predicted for plotting.'
    assert hasattr(prediction, 'logits'), 'Logits should have been predicted for plotting.'
    if params.data_generation.prediction_target == 'position_and_velocity':
        assert hasattr(prediction, 'velocities'), 'Velocities should have been predicted for plotting.'

    bs, num_queries = prediction.positions.shape[:2]

    if params.data_generation.prediction_target == 'position_and_shape':
        raise NotImplementedError('Plotting not working yet for shape predictions.')

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
                    p = ax.plot(pos_x, pos_y, marker='o', color='r', markersize=5)#, label="Model prediction")
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

                ax.add_patch(ell_position)
                # if do_plot_vel:
                #     ell_velocity = Ellipse(xy=(pos_x + vel_x, pos_y + vel_y), width=uncertainties[i, 2]*4,
                #                         height=uncertainties[i, 3]*4, edgecolor=color, linestyle='--', facecolor='none')
                #     ax.add_patch(ell_velocity)



        else:
            if do_plot_unmatched_hypotheses:
                if once:
                    p = ax.plot(pos_x, pos_y, marker='*', color='m',markersize=5 )# label='Unmatched hypothesis')
                    once = False
                else:
                    p = ax.plot(pos_x, pos_y, marker='*', color='m', markersize=5)

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
    clipped_measurements = clip_batch(measurements, N, timestep)
    label = label_at_step_from_gt(gt, timestep)

    prediction, intermediate_predictions, encoder_prediction, aux_classifications, _ = model.forward(clipped_measurements, timestep)
    output_existence_probabilities = prediction.logits.sigmoid().detach().cpu().numpy().flatten()
    existence_threshold = 0.75 # Hyperparameter
    alive_indices = np.where(output_existence_probabilities > existence_threshold)[0]
    prediction_in_format_for_loss = {'state': torch.cat((prediction.positions, prediction.velocities), dim=2),
                                       'logits': prediction.logits,
                                       'state_covariances': prediction.uncertainties ** 2}
    loss, gospa_alive_indices, decomposition = mot_loss.gospa_forward(
        prediction_in_format_for_loss, label, probabilistic=False,
        existence_threshold=existence_threshold)


    gospa_total.append(loss.item())
    gospa_loc.append(decomposition['localization'])
    gospa_loc_norm.append(decomposition['localization'] / decomposition['n_matched_objs'] if \
        decomposition['n_matched_objs'] != 0 else 0.0)
    gospa_miss.append(decomposition['missed'])
    gospa_false.append(decomposition['false'])


    # Plot results
    # TODO: Figure out what to do with the offset, as it depends on M
    if do_plot:
        clipped_true_measurements = clip_measurements(true_measurements, M, timestep)
        clipped_false_measurements = clip_measurements(false_measurements, M, timestep)
        clipped_trajectories = clip_trajectories(trajectories, N, timestep)


        scatter_and_decay(clipped_true_measurements, color="k",  marker="x" )#, label= "True measurements")
        scatter_and_decay(clipped_false_measurements, color="k", marker=".")#, label="False measurements")

        prior_lengths = plot_ground_truth(clipped_trajectories, prior_lengths, M, np.round(timestep * 0.1, 3))
        output_truth_plot(output_ax, prediction, gospa_alive_indices[0][0], clipped_measurements, params)

        output_ax.set_aspect('equal', 'box')
        output_ax.set_ylabel('North')
        output_ax.set_xlabel('East')
        output_ax.grid('on')
        mlim = 20
        output_ax.set_xlim([-mlim, mlim]) 
        output_ax.set_ylim([-mlim, mlim]) 

        legend_handles = []

        legend_handles.append(Line2D([0], [0], marker='o', color='w', label='MT3 estimates', markerfacecolor='r', markersize=7))
        legend_handles.append(Line2D([0], [0], marker='*', color='w', label='Dead MB components', markerfacecolor='m', markersize=10))
        legend_handles.append(Line2D([0], [0], label='Ground truth tracks', color='k'))
        legend_handles.append(Line2D([0], [0], marker='X', color='w', label='True measurements', markerfacecolor='k', markersize=7))
        legend_handles.append(Line2D([0], [0], marker='.', color='w', label='False measurements', markerfacecolor='k', markersize=10))
        legend_handles.append(Line2D([0], [0], marker='D', color='w', label='Current targets', markerfacecolor='g', markersize=7))
        plt.legend(handles=legend_handles)


        fig.canvas.draw()

        #fig.canvas.flush_events()
        #time.sleep(0.1)
        #output_ax.clear()

    timestep += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-tp', '--task_params', help='filepath to configuration yaml file defining the task', required=True)
    parser.add_argument('-mp', '--model_params', help='filepath to configuration yaml file defining the model', required=True)
    parser.add_argument("-s", "--scenario")
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
    # 1798125999
    
    params.general.pytorch_and_numpy_seed = int.from_bytes(random_data, byteorder="big")
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

    data_generator = DataGenerator(params)
    last_filename = "C:/Users/chris/MT3v2/task2/checkpoints/" + "checkpoint_gradient_step_699999"

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

    params.data_generation.n_timesteps = prev
    # Calculate all values used for plotting


    gt = pd.read_csv(args.scenario, delimiter=',') # assume: [id, x, y, vx, vy, t]
    tensor, true_measurements, false_measurements, trajectories = tensor_from_gt(gt, params)
    mask = torch.tensor([False]).repeat(tensor.size()[1])
    mask = mask[None, :]
    measurements = NestedTensor(tensor, mask)

    # For "normal" scenarios
    #measurements, labels, unique_ids, _, trajectories, true_measurements, false_measurements = data_generator.get_batch()

    do_plot = True
    do_plot_preds = True
    do_plot_ellipse = True
    do_plot_vel = False
    do_plot_unmatched_hypotheses = True
    do_interactive = True

    if not do_interactive and do_plot:
        plt.ion() # Required for dynamic plot updates
    if do_plot:
        fig, output_ax = plt.subplots() 

    K = 10 # amount of gt in the past to plot.
    N = 20 # Track length to consider. This cannot be longer than 20 since the learned position encoding is fixed in size
    M = 20 # amount of measurements in the past to plot.

    prior_lengths = {}
    timestep = 20 # Have to start at 20 to avoid problems with the network 
    gospa_total = []
    gospa_loc = []
    gospa_loc_norm = []
    gospa_miss = []
    gospa_false = []

    all_gospa_total = None
    all_gospa_loc = None
    all_gospa_loc_norm = None
    all_gospa_miss = None
    all_gospa_false = None

    # Batch includes all MEASUREMENTS, and thus the window should gate measurements on their timestep, not the amount of measurements.
 
    # Note: The ground truth tracks may have come from very early timesteps. As
    # such, when plotting the latest tracking estimates, only the active tracks
    # will get estimates. The green GT points overlaid GT tracks will show which
    # tracks were active at the last timestep.

    # TODO: How to test for active hypotheses for each time step. I.e. is there a probability array that can be printed?
    from matplotlib.widgets import Button

    def next(event):
        fig.canvas.flush_events()
        time.sleep(0.1)
        output_ax.clear()

    b_ax = plt.axes([0.8, 0.0, 0.1, 0.05])
    bclear = Button(b_ax, 'Clear')
    plt.sca(output_ax)

    b_ax = plt.axes([0.6, 0.0, 0.1, 0.05])
    bnext = Button(b_ax, 'Next')
    plt.sca(output_ax)


    bclear.on_clicked(next)
    bnext.on_clicked(step_once)
    plt.sca(output_ax)


    n_mc = 1
    n_ts = 100

    if do_interactive:
        try:
            fig.canvas.mpl_connect('key_press_event', step_once)
            # leg = output_ax.legend()
            # for lh in leg.legendHandles: 
            #     lh.set_alpha(1)

            #ground_truths = mpatches.Patch(color='red', label='The red data')
            #blue_patch = mpatches.Patch(color='blue', label='The blue data')
            #output_ax.legend()

            plt.show()
        except RuntimeError as e:
            exit()
    else:
        for i in range(n_mc):
            print(f"Monte carlo run {i + 1}/{n_mc}")
            prior_lengths = {}
            timestep = 0 # Have to start at 20 to avoid problems with the network 
            gospa_total = []
            gospa_loc = []
            gospa_loc_norm = []
            gospa_miss = []
            gospa_false = []
            for _ in tqdm(range(n_ts)):
                # TODO: Add pause function
                
                step_once(None)
                #time.sleep(0.1)
            if all_gospa_total is None:
                all_gospa_total = gospa_total
            else:
                all_gospa_total = list(map(add, all_gospa_total, gospa_total))

            if all_gospa_loc is None:
                all_gospa_loc = gospa_loc
            else:
                all_gospa_loc = list(map(add, all_gospa_loc, gospa_loc))

            if all_gospa_loc_norm is None:
                all_gospa_loc_norm = gospa_loc_norm
            else:
                all_gospa_loc_norm = list(map(add, all_gospa_loc_norm, gospa_loc_norm))

            if all_gospa_miss is None:
                all_gospa_miss = gospa_miss
            else:
                all_gospa_miss = list(map(add, all_gospa_miss, gospa_miss))

            if all_gospa_false is None:
                all_gospa_false = gospa_false
            else:
                all_gospa_false = list(map(add, all_gospa_false, gospa_false))

        # Turn off interactive and keep plot open
        plt.ioff()
        #plt.show()

    total_gospa = np.array(all_gospa_total) / n_mc
    rms_gospa = np.sqrt(total_gospa**2)
    total_loc = np.array(all_gospa_loc) / n_mc
    total_loc_norm = np.array(all_gospa_loc_norm) / n_mc
    total_miss = np.array(all_gospa_miss) / n_mc
    total_false = np.array(all_gospa_false) / n_mc

    fig, ax = plt.subplots(5, 1)
    fig.tight_layout()
    
    offset = 20
    end = 91 + offset
    ax[0].plot(np.arange(len(rms_gospa[offset:end])), rms_gospa[offset:end], "k")
    ax[0].set_title("RMS GOSPA error")
    ax[0].axvline(x=50, color="r", linestyle="--")
    ax[1].plot(np.arange(len(total_loc[offset:end])), total_loc[offset:end], "k")
    ax[1].set_title("GOSPA: Localization error")
    ax[1].axvline(x=50, color="r", linestyle="--")
    ax[2].plot(np.arange(len(total_loc_norm[offset:end])), total_loc_norm[offset:end], "k")
    ax[2].set_title("GOSPA: Localization error (normalized for # objects)")
    ax[2].axvline(x=50, color="r", linestyle="--")
    ax[3].plot(np.arange(len(total_miss[offset:end])), total_miss[offset:end], "k")
    ax[3].set_title("GOSPA: Missed objects")
    ax[3].axvline(x=50, color="r", linestyle="--")
    ax[4].plot(np.arange(len(total_false[offset:end])), total_false[offset:end], "k")
    ax[4].set_title("GOSPA: False detections")
    ax[4].axvline(x=50, color="r", linestyle="--")

    #fig.suptitle("GOSPA over time for scenario with 6 targets. Targets collide at step 32")


    import scipy.stats as st

    print(f"RMS GOSPA: {rms_gospa.mean()} +/- {max(st.norm.interval(0.95, loc=0, scale=st.sem(rms_gospa)))}")
    print(f"LOC: {total_loc.mean()} +/- {max(st.norm.interval(0.95, loc=0, scale=st.sem(total_loc)))}")
    print(f"MISS: {total_miss.mean()} +/- {max(st.norm.interval(0.95, loc=0, scale=st.sem(total_miss)))}")
    if total_false.sum() == 0:
        print(f"No false detections...")
    else:
        print(f"FALSE: {total_false.mean()} +/- {max(st.norm.interval(0.95, loc=0, scale=st.sem(total_false)))}")

    plt.show()


"""
Notes: The tracks are the same before and after crossing, so we should expect the GOSPA loss to be the same before and after

However: this symmetry only occurs from timestep 20 onwards. This is likely because the sliding window is max 20 timesteps, and
such, the information available to the tracker is skewed from timestep 0 to 19


Make sure to up the Radar max range for scenarios that start far away!


Changing radar parameters: Considerations?

"""
