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


from matplotlib.patches import Ellipse

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


params.data_generation.seed = params.general.pytorch_and_numpy_seed #random.randint(0, 9999999999999999999) # Tune this to get different runs
do_plot_preds = True
do_plot_ellipse = True
do_plot_vel = True
do_plot_missed_predictions = True

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




    # Plot xy position of measurements, alpha-coded by time
    measurements = batch.tensors[training_example_to_plot][~batch.mask[training_example_to_plot]]
    colors = np.zeros((measurements.shape[0], 4))
    unique_time_values = np.array(sorted(list(set(measurements[:, 3].tolist()))))
    def f(t):
        """Exponential decay for alpha in time"""
        idx = (np.abs(unique_time_values - t)).argmin()
        return 1/1.2**(len(unique_time_values)-idx)

    colors[:, 3] = [f(t) for t in measurements[:, 3].tolist()]
    
    measurements_cpu = measurements.cpu()
    meas_r = measurements_cpu[:, 0]
    meas_theta = measurements_cpu[:, 2]
    meas_x = meas_r * np.cos(meas_theta)
    meas_y = meas_r * np.sin(meas_theta)

    ax.scatter(meas_x, meas_y, marker='x', c=colors, zorder=np.inf)
    # ax.scatter(X, Y, marker='x', color="k", alpha=0.8, label="True Measurements")
    # ax.scatter(Xf, Yf, marker='x', color="r", alpha=0.8, label="False Measurements")

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
                # Plot predicted positions
                p = ax.plot(pos_x, pos_y, marker='o', label='Object Pred', markersize=5)
                color = p[0].get_color()

                # Plot ground-truth
                ax.plot(gt_x, gt_y, marker='D', color=color, label='Object GT', markersize=5)

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


    #     label = "{:.2f}".format(out_prob[i])
    #     ax.annotate(label, # this is the text
    #                 (out[i,0], out[i,1]), # this is the point to label
    #                 textcoords="offset points", # how to position the text
    #                 xytext=(0,10), # distance from text to points (x,y)
    #                 ha='center',
    #                 color=p[0].get_color())

        ax.legend()


data_generator = DataGenerator(params)
last_filename = "/home/strom/models/task1/checkpoints/" + "checkpoint_gradient_step_999999"

# Load model weights and pass model to correct device
model = MT3V2(params)

checkpoint = torch.load(last_filename, map_location=params.training.device)
model.load_state_dict(checkpoint['model_state_dict'])

model.to(torch.device(params.training.device))

mot_loss = MotLoss(params)
mot_loss.to(params.training.device)



# Calculate all values used for plotting
batch, labels, unique_ids, _, trajectories, true_measurements, false_measurements = data_generator.get_batch()

gt = true_measurements[0]
ft = false_measurements[0]
#print(gt_one)

X = []
Y = []
#for gt in true_measurements:
for sample in gt:
    h = sample[3]
    r = sample[0]
    theta = sample[2]
    X.append(r * np.cos(theta))
    Y.append(r * np.sin(theta))


Xf= []
Yf = []
for sample in ft:
    h = sample[3]
    r = sample[0]
    theta = sample[2]
    Xf.append(r * np.cos(theta))
    Yf.append(r * np.sin(theta))

prediction, intermediate_predictions, encoder_prediction, aux_classifications, _ = model.forward(batch)
loss_dict, indices = mot_loss.forward(labels, prediction, intermediate_predictions, encoder_prediction, loss_type=params.loss.type)

fig = plt.figure(constrained_layout=True)

gs = GridSpec(2, 2, figure=fig)

output_ax = fig.add_subplot(111)
output_ax.set_ylabel('Y')
output_ax.set_xlabel('X')
output_ax.set_aspect('equal', 'box')


output_ax.cla()
output_ax.grid('on')
output_truth_plot(output_ax, prediction, labels, indices, batch, params, 0)

#output_ax.set_xlim([-params.data_generation.field_of_view.max_range*0.2, params.data_generation.field_of_view.max_range*1.2])
#output_ax.set_ylim([-params.data_generation.field_of_view.max_range, params.data_generation.field_of_view.max_range])

plt.show()

