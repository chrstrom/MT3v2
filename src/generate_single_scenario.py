import sys
import argparse
import os

import torch
import numpy as np

from data_generation.data_generator import DataGenerator
from util.load_config_files import load_yaml_into_dotdict, dotdict

import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('-fp', '--filepath', help='filepath to configuration yaml file', required=True)
args = parser.parse_args()

seed = int.from_bytes(os.urandom(4), byteorder="big")
print(f'Using seed: {seed}')
n_samples = 1
print_n_avg_objects = True

# Create data generator
params = load_yaml_into_dotdict(args.filepath)
params.training = dotdict()
params.training.batch_size = 1
params.training.device = 'cuda' if torch.cuda.is_available() else 'cpu'
params.data_generation.seed = seed
data_gen = DataGenerator(params)

def get_xy_from_data(data):
    Xgt = []
    Ygt = []
    for point in data:
        Xgt.append(point[0])
        Ygt.append(point[1])

    return Xgt, Ygt

def get_rtheta_from_data(data):
    r = []
    theta = []
    for point in data:
        r.append(point[0])
        theta.append(point[2])

    return r, theta




training_tensor, labels, unique_ids, unique_measurement_ids, trajectories, true_measurements, false_measurements = data_gen.get_batch()

cmap = plt.cm.get_cmap('nipy_spectral', len(trajectories[0])) 

for color_idx, (track_id, track) in enumerate(trajectories[0].items()):

    Xgt, Ygt = get_xy_from_data(track)
    
    for i in range(len(Xgt)):
        plt.plot(Xgt[i:i+2], Ygt[i:i+2], 'o-',c=cmap(color_idx), alpha = i/len(Xgt))

r, theta = get_rtheta_from_data(true_measurements[0])
Xm = r*np.cos(theta)
Ym = r*np.sin(theta)
plt.scatter(Xm, Ym, c='r', marker='x', alpha = 0.75)

r, theta = get_rtheta_from_data(false_measurements[0])
Xf = r*np.cos(theta)
Yf = r*np.sin(theta)
plt.scatter(Xf, Yf, c='k', marker='x', alpha = 0.1)
plt.show()



# savedict = \
#    {
#        'measurements': measurements,
#        'ground_truths': gt_datas,
#        'hyperparam_n_timesteps': params.data_generation.n_timesteps,
#        'p_add': params.data_generation.p_add,
#        'p_remove': params.data_generation.p_remove,
#        'p_meas': params.data_generation.p_meas,
#        'sigma_q': params.data_generation.sigma_q,
#        'sigma_y': params.data_generation.sigma_y,
#        'n_avg_false_measurments': params.data_generation.n_avg_false_measurements,
#        'n_avg_starting_objects': params.data_generation.n_avg_starting_objects,
#        'field_of_view_lb': params.data_generation.field_of_view_lb,
#        'field_of_view_ub': params.data_generation.field_of_view_ub,
#        'mu_x0': params.data_generation.mu_x0,
#        'std_x0': params.data_generation.std_x0,
#        'mu_v0': params.data_generation.mu_v0,
#        'std_v0': params.data_generation.std_v0,
#    }

# task_name = sys.argv[-1].split('/')[-1].split('.')[0]
# print(task_name)
# print(n_samples)
# print(seed)
# #print(savedict)

# ground_truths = savedict['ground_truths']

# Xgt = []
# Ygt = []
# for gt in ground_truths:
#     gt = gt[0]
#     x = gt[0]
#     y = gt[1]
#     vx = gt[2]
#     vy = gt[3]

#     Xgt.append(x)
#     Ygt.append(y)



# # import numpy as np
# import matplotlib.pyplot as plt

# plt.scatter(Xgt, Ygt)
# plt.show()


#for meas in savedict['measurements']:
#    print(meas[0])

#savemat(f'{task_name}-{n_samples}samples-seed{seed}.mat', savedict)