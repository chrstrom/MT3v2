from doctest import set_unittest_reportflags
import multiprocessing

import numpy as np
from numpy.random import SeedSequence, default_rng
import torch
from torch import Tensor

from data_generation.mot_data_generation import MotDataGenerator
from util.misc import NestedTensor


"""
Goal: Create an interface that generates a scenario on the format that the MT3v2
can use from a predefined list of trajectories.

This will let me manually design a ground truth scenario, and generate measurements
that can directly be fed to the generate_scenario code.

The first scenario to be created is the "converge at one point" scenario.
After this, look into the 9 ravens example.

The current code that exists here is the one from data_generator. Will have to
be changed.
"""

class DataGenerator:
    def __init__(self, params, rngs=None):
        self.params = params
        assert 0 <= params.data_generation.n_prediction_lag <= params.data_generation.n_timesteps, "Prediction lag has to be smaller than the total number of time-steps."
        self.device = params.training.device
        self.n_timesteps = params.data_generation.n_timesteps

        self.pool = multiprocessing.Pool()

        # Create `batch_size` data generators, each with its own independent (to a high probability) RNG
        ss = SeedSequence(params.data_generation.seed)
        if rngs is None:
            rngs = [default_rng(s) for s in ss.spawn(params.training.batch_size)]
        else:
            assert len(rngs) == params.training.batch_size, 'The number of provided RNGs must match the desired batch size'
        self.datagens = [MotDataGenerator(params, rng=rng) for rng in rngs]

    def get_batch(self):
        """
        This is the function that is called which returns measurements and gt

        The manually constructed trajectories should replace the "results" variable,
        which means that the "get_single_training_example" must be altered.
        """
        if len(self.datagens) != 1:
            results = self.pool.starmap(get_single_training_example, zip(self.datagens, [self.n_timesteps]*len(self.datagens)))
        else:
            results = [get_single_training_example(self.datagens[0], self.n_timesteps)]

        # Unpack results

        training_data, labels, unique_measurement_ids, unique_label_ids, trajectories, new_rngs, true_measurements, false_measurements = tuple(zip(*results))

        labels = [Tensor(l).to(torch.device(self.device)) for l in labels]
        trajectories = list(trajectories)
        unique_measurement_ids = [list(u) for u in unique_measurement_ids]
        unique_label_ids = list(unique_label_ids)

        # Update the RNGs of all the datagens for next call
        for datagen, new_rng in zip(self.datagens, new_rngs):
            datagen.rng = new_rng

        # Pad training data
        max_len = max(list(map(len, training_data)))
        training_data, mask = pad_to_batch_max(training_data, max_len)

        # Pad unique ids
        for i in range(len(unique_measurement_ids)):
            unique_id = unique_measurement_ids[i]
            n_items_to_add = max_len - len(unique_id)
            unique_measurement_ids[i] = np.concatenate([unique_id, [-2] * n_items_to_add])[None, :]
        unique_measurement_ids = np.concatenate(unique_measurement_ids)

        training_nested_tensor = NestedTensor(Tensor(training_data).to(torch.device(self.device)),
                                              Tensor(mask).bool().to(torch.device(self.device)))
        unique_measurement_ids = Tensor(unique_measurement_ids).to(self.device)

        return training_nested_tensor, labels, unique_measurement_ids, unique_label_ids, trajectories, true_measurements, false_measurements

    def __del__(self):
        self.pool.close()


def pad_to_batch_max(training_data, max_len):
    batch_size = len(training_data)
    d_meas = training_data[0].shape[1]
    training_data_padded = np.zeros((batch_size, max_len, d_meas))
    mask = np.ones((batch_size, max_len))
    for i, ex in enumerate(training_data):
        training_data_padded[i,:len(ex),:] = ex
        mask[i,:len(ex)] = 0

    return training_data_padded, mask


def get_single_training_example(data_generator, n_timesteps):
    """Generates a single training example

    Returns:
        training_data   : A single training example
        label_data       : Ground truth for example
    """

    """
    This code should be replaced to instead use the manually defined targets.

    Targets, timestamps and GT states can be defined in a .csv or the likes

    True measurements should then be generated according to the measurement model
    (see data_generator)

    False measurements should also be generated according to task parameters
    (also see data_generator)

    The rest of the values required in the output should also be generated
    according to data_generator.

    It could be an idea to instead just edit the data_generator code? I.e. any
    step becomes a step forward in the .csv containing the trajectories etc.
    """

    data_generator.reset()
    label_data = []

    while len(label_data) == 0 or len(data_generator.measurements) == 0:
        # Generate n_timesteps of data, from scratch
        data_generator.reset()
        for i in range(n_timesteps - 1):
            # The default workings of step() is to propagate an objects forward, n timesteps. The latest step is then reported. 
            # To get Ground truths then we would need to get the output from each step here.
            data_generator.step()

        training_data, label_data, unique_measurement_ids, unique_label_ids, true_measurements, false_measurements = data_generator.finish()
    new_rng = data_generator.rng
    return training_data, np.array(label_data).reshape(len(label_data),-1), unique_measurement_ids, unique_label_ids, \
           data_generator.trajectories.copy(), new_rng, true_measurements, false_measurements
