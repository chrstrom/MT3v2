import os
import time
import datetime
import re
import shutil
from collections import deque
import argparse

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from data_generation.data_generator import DataGenerator
from util.misc import save_checkpoint, update_logs
from util.load_config_files import load_yaml_into_dotdict
from util.plotting import output_truth_plot, compute_avg_certainty, get_constrastive_ax, get_false_ax, \
    get_total_loss_ax, get_state_uncertainties_ax
from util.logger import Logger
from modules.loss import MotLoss
from modules.contrastive_loss import ContrastiveLoss
from modules import evaluator
from modules.models.mt3v2.mt3v2 import MT3V2


import time


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

#!!!!!!!!!! IMPORTANT TO LOAD AND UPDATE PARAMS FOR EVAL
eval_params = load_yaml_into_dotdict('configs/eval/default.yaml')
params.recursive_update(eval_params)
eval_params.update(load_yaml_into_dotdict(args.model_params))
eval_params.recursive_update(load_yaml_into_dotdict('configs/eval/default.yaml'))
eval_params.data_generation.seed += 1  # make sure we don't evaluate with same seed as final evaluation after training

# Generate 32-bit random seed, or use user-specified one
if params.general.pytorch_and_numpy_seed is None:
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


last_filename = "/home/strom/pinto/MT3v2/task4/checkpoints/" + "checkpoint_gradient_step_399999"

# Load model weights and pass model to correct device
data_generator = DataGenerator(params)  
model = MT3V2(params)

checkpoint = torch.load(last_filename, map_location=params.training.device)
model.load_state_dict(checkpoint['model_state_dict'])

model.to(torch.device(params.training.device))
optimizer = AdamW(model.parameters(), lr=params.training.learning_rate, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer,
                                patience=params.training.reduce_lr_patience,
                                factor=params.training.reduce_lr_factor,
                                verbose=params.debug.print_reduce_lr_messages)


#print(model)
import scipy.stats as st
N = 100 # Note that this is multiplied by n_samples in eval params!!!!
total_gospa = []
total_loc = []
total_miss = []
total_false = []

start = time.time()
for i in range(N):
    gospa_total, gospa_loc, gospa_norm_loc, gospa_miss, gospa_false = evaluator.evaluate_gospa(data_generator, model, eval_params)

    total_gospa.append(gospa_total)
    total_loc.append(gospa_loc)
    total_miss.append(gospa_miss)
    total_false.append(gospa_false)

    print(f"{i} / {N}")

print(f"Time taken: {time.time() - start} seconds")

total_gospa = np.array(total_gospa)
total_loc = np.array(total_loc)
total_miss = np.array(total_miss)
total_false = np.array(total_false)

with open('total_gospa.npy', 'wb') as f:
    np.save(f, total_gospa)
with open('total_loc.npy', 'wb') as f:
    np.save(f, total_loc)
with open('total_miss.npy', 'wb') as f:
    np.save(f, total_miss)
with open('total_false.npy', 'wb') as f:
    np.save(f, total_false)



"""
Time taken: 285.16901659965515 seconds
GOSPA: 3.3874949262566862 +/- 1.82120102201727
LOC: 2.322494924592227 +/- 1.1180683024625047
MISS: 0.9165 +/- 1.3693469246323227
FALSE: 0.1485 +/- 0.5373927800036022

"""