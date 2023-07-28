from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
import pickle
import os

class PickleLatentDatasetLoader(Dataset):
    def __init__(self, annotations_file, dir):
        self.dir_all = []
        self.file_lists_all = []
        for idx, i in enumerate(annotations_file):
            self.dir_all.append(dir[idx])
            self.file_lists_all.append(pd.read_csv(i, header=None))

    def __len__(self):
        return len(self.file_lists_all[0])

    def __getitem__(self, idx):
        results_all_0 = []
        results_all_1 = []
        label_all = []
        for idx2, i in enumerate(self.file_lists_all):
            file_path = os.path.join(self.dir_all[idx2], i.iloc[idx, 0].replace('/','\\'))
            label = i.iloc[idx, 1]

            pkl_file = open(file_path, 'rb')
            results = pickle.load(pkl_file)
            results = np.array(results)
            pkl_file.close()

            results = torch.tensor(results)
            label = torch.tensor(label).type(torch.FloatTensor)

            results_all_0.append(results[0].squeeze())
            results_all_1.append(results[1].squeeze())
            label_all.append(label)

        return results_all_0, results_all_1, label_all
