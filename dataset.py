#!/usr/bin/env python
# coding=utf-8
# wujian@17.11.8

import json
import os
import pickle
import numpy as np
import torch as th
import torch.utils.data as data

class MaskDataset(data.Dataset):
    def __init__(self, root, num_jobs, training=True):
        self.mask_list = list()
        self.root_dir  = root
        for index in range(1, 1 + num_jobs):
            json_path = os.path.join(root, 'flist_{}_{}.json'.format('tr' if training else 'dt', index))
            with open(json_path, 'r') as f:
                arr = json.load(f)
                self.mask_list.extend(arr)
        print('load {} items in total, model = {}'.format(len(self.mask_list), \
                "training" if training else "evaluate"))

    def __len__(self):
        return len(self.mask_list)

    def __getitem__(self, index):
        load_path = os.path.join(self.root_dir, self.mask_list[index])
        with open(load_path, 'rb') as f:
            mask_dict = pickle.load(f)
        return mask_dict['Y_abs'], mask_dict['IBM_X'], mask_dict['IBM_N']

def collate_func(batch):
    assert len(batch) == 1, "batch_size should be set to 1"
    tensor_list = []
    for index, array in enumerate(batch[0]):
        assert type(array).__name__ == 'ndarray', "items in the dataset is expected to be numpy type(ndarray)"
        if index >= 1:
            array = array.reshape(array.shape[0] * array.shape[1], array.shape[2])
        tensor_list.append(th.from_numpy(array))
    return tensor_list


def test():
    dataset = MaskDataset('masks', 15, training=False)
    test_loader = data.DataLoader(dataset=dataset, collate_fn=collate_func, shuffle=False)
    for yabs, ibm_x, ibm_n in test_loader:
        print(yabs.shape)
        break

if __name__ == '__main__':
    test()
