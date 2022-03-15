# credit:
# https://stackoverflow.com/questions/57893415/pytorch-dataloader-for-time-series-task
# https://discuss.pytorch.org/t/tensorflow-esque-bucket-by-sequence-length/41284/13

import pickle
import time
import scipy.io
import numpy as np
import pandas as pd
from torch.utils.data import Sampler, Dataset
from collections import OrderedDict
from random import shuffle
from joblib import Parallel, delayed


def from_mat_to_ready():
    """
    Loads raw .mat data file and outputs a data dictionary with data
    and variables needed for training loop
    """

    mat = scipy.io.loadmat('data/data_PPMI.mat')

    # Make data arrays based on .mat file
    Y = mat['Y']
    X = mat['X']
    pat_idx = mat['deidentified_patno'] - 1  # subtract one from all to make things work out nicer later

    # Make time deltas
    t = mat['time']
    time_pd = pd.DataFrame(pd.to_datetime(t.squeeze(), format='%Y%m'))  # convert to pandas df to ease datetime operation
    time_pd.index = pat_idx.squeeze()
    time_pd['Delta_t'] = time_pd.groupby(by=time_pd.index).diff().astype('timedelta64[D]').fillna(0)

    # Make needed variables for data lists
    unique_patients = np.unique(pat_idx)
    Y_list = [None] * len(unique_patients)
    X_list = [None] * len(unique_patients)
    td_list = [None] * len(unique_patients)

    start = time.time()
    # Make each patient a specific index
    for patient in unique_patients:
        which_obs = (pat_idx == patient).squeeze()  # sum of this is the number of observations (sequence length) for that patient
        X_list[patient] = X[which_obs, :].astype(np.float32)
        Y_list[patient] = Y[which_obs, :].astype(np.float32)
        td_list[patient] = time_pd.loc[which_obs, 'Delta_t'].to_numpy().astype(np.float32)

    X_array = np.array(X_list, dtype=object)
    Y_array = np.array(Y_list, dtype=object)
    td_array = np.array(td_list, dtype=object)

    data_dict = {'X': X_array,
                 'Y': Y_array,
                 'time_delta': td_array,
                 'pat_idx': pat_idx}  # pat_idx also implicitly contained in X_array and Y_array

    stop = time.time()
    print('Elapsed time for the entire processing: {:.2f} s'
          .format(stop - start))

    with open('data/data_dict.pkl', 'wb') as handle:
        pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return data_dict


def data_dict_loader(prep_data=True):

    if prep_data:
        data_dict = from_mat_to_ready()
    else:
        with open('data/data_dict.pkl', 'rb') as handle:
            data_dict = pickle.load(handle)

    return data_dict


class BucketDataset(Dataset):
    """
    PyTorch Dataset class to ease training loop
    """

    def __init__(self, inputs, targets, time_deltas):
        self.inputs = inputs
        self.targets = targets
        self.time_deltas = time_deltas

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        if self.targets is None:
            return self.inputs[index], self.time_deltas[index]
        else:
            return self.inputs[index], self.targets[index], self.time_deltas[index]


class BucketBatchSampler(Sampler):
    """
    Sampler which ensures:
        - Sequences of equal length end up in the same batches
        - Shuffle batches such that random which sequence length model is trained on next iteration
    """
    def __init__(self, inputs, batch_size):
        self.batch_size = batch_size
        ind_n_len = []
        for i, p in enumerate(inputs):
            ind_n_len.append((i, p.shape[0]))  # find sequence length of input i
        self.ind_n_len = ind_n_len
        self.batch_list = self._generate_batch_map()
        self.num_batches = len(self.batch_list)

    def _generate_batch_map(self):
        # shuffle all of the indices first so they are put into buckets differently
        shuffle(self.ind_n_len)
        # Organize lengths, e.g., batch_map[10] = [30, 124, 203, ...] <= indices of sequences of length 10
        batch_map = OrderedDict()
        for idx, length in self.ind_n_len:
            if length not in batch_map:
                batch_map[length] = [idx]
            else:
                batch_map[length].append(idx)
        # Use batch_map to split indices into batches of equal size
        # e.g., for batch_size=3, batch_list = [[23,45,47], [49,50,62], [63,65,66], ...]
        batch_list = []
        for length, indices in batch_map.items():
            for group in [indices[i:(i + self.batch_size)] for i in range(0, len(indices), self.batch_size)]:
                batch_list.append(group)
        return batch_list

    def batch_count(self):
        return self.num_batches

    def __len__(self):
        return len(self.ind_n_len)

    def __iter__(self):
        self.batch_list = self._generate_batch_map()
        # shuffle all the batches so they are not ordered by bucket size
        shuffle(self.batch_list)
        for i in self.batch_list:
            yield i
