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


    n_samples = 1
    print_n_avg_objects = True

    # Create data generator
    # params = load_yaml_into_dotdict(args.task_params)
    # params.training = dotdict()
    params.training.batch_size = 1

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



    # Training tensor has rows that contains [r, rdot, theta, time]
    training_tensor, labels, unique_ids, unique_measurement_ids, trajectories, true_measurements, false_measurements = data_gen.get_batch()

    cmap = plt.cm.get_cmap('nipy_spectral', len(trajectories[0])) 

    for color_idx, (track_id, track) in enumerate(trajectories[0].items()):

        Xgt, Ygt = get_xy_from_data(track)
        
        for i in range(len(Xgt)):
            if i < len(Xgt) - 1:
                plt.plot(Xgt[i:i+2], Ygt[i:i+2], 'o-',c=cmap(color_idx), alpha = i/len(Xgt))
            else: 
                plt.plot(Xgt[i:i+2], Ygt[i:i+2], 'o-',c=cmap(color_idx), alpha = i/len(Xgt), label=f"Track {track_id}")

    r, theta = get_rtheta_from_data(true_measurements[0])
    Xm = r*np.cos(theta)
    Ym = r*np.sin(theta)
    plt.scatter(Xm, Ym, c='r', marker='x', alpha = 0.75, label="true measurements")

    r, theta = get_rtheta_from_data(false_measurements[0])
    Xf = r*np.cos(theta)
    Yf = r*np.sin(theta)
    plt.scatter(Xf, Yf, c='k', marker='x', alpha = 0.1, label="false measurements")

    params.data_generation.n_timesteps = 20
    model = MT3V2(params)
    mot_loss = MotLoss(params)
    mot_loss.to(params.training.device)

    last_filename = "C:/Users/chris/MT3v2/task1/checkpoints/" + "checkpoint_gradient_step_999999"
    checkpoint = torch.load(last_filename, map_location=params.training.device)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.to(torch.device(params.training.device))

    params.data_generation.n_timesteps = 40
    def pad_to_batch_max(training_data, max_len):
        batch_size = len(training_data)
        d_meas = training_data[0].shape[1]
        training_data_padded = np.zeros((batch_size, max_len, d_meas))
        mask = np.ones((batch_size, max_len))
        for i, ex in enumerate(training_data):
            training_data_padded[i,:len(ex),:] = ex
            mask[i,:len(ex)] = 0

        return training_data_padded, mask


    for k in range(params.data_generation.n_timesteps-20):

        z_k = tuple([training_tensor.tensors.numpy()[0][k:k+20]]) # Sliding window of 20 elements

        #print(f"{k}: {len(z_k[0])}")

        # Pad training data
        max_len = max(list(map(len, z_k)))
        partial_tensor, mask = pad_to_batch_max(z_k, max_len)

        training_nested_tensor = NestedTensor(torch.Tensor(partial_tensor).to(torch.device(params.training.device)),
                                                torch.Tensor(mask).bool().to(torch.device(params.training.device)))
        
        #print(training_nested_tensor)

        prediction, intermediate_predictions, encoder_prediction, aux_classifications, _ = model.forward(training_nested_tensor)
        pred_np = prediction.positions.detach().numpy()
        #print(f"{k}: {pred_np}")
        x, y = get_xy_from_data(pred_np)
        plt.scatter(x, y, marker='o')

        #print(f"{k}: {pred_track_zero}, length:{len(z_k[0])}")


    #prediction, intermediate_predictions, encoder_prediction, aux_classifications, _ = model.forward(training_tensor)
    #loss_dict, matched_idx = mot_loss.forward(labels, prediction, intermediate_predictions, encoder_prediction, loss_type=params.loss.type)


    #0 = last
    # ex_to_plot = 0
    # pos = prediction.positions[ex_to_plot]
    # vel = prediction.velocities[ex_to_plot]
    # out = torch.cat((pos, vel), dim=1).cpu().detach().numpy()

    # indices = tuple([t.cpu().detach().numpy() for t in matched_idx[ex_to_plot]])


    # for i, posvel in enumerate(out):
    #     pos_x = posvel[0]
    #     pos_y = posvel[1]
    #     vel_x = posvel[2]
    #     vel_y = posvel[3]

    #     if i in indices[0]:
    #         plt.plot(pos_x, pos_y, marker='o', label='Object Pred', markersize=5)

    plt.xlabel("East [m]")
    plt.ylabel("North [m]")
    plt.legend()
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