from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
import pickle
import os

class PickleLatentDatasetLoader(Dataset):
    def __init__(self, annotations_file, dir):
        self.file_lists = pd.read_csv(annotations_file, header=None)
        self.dir = dir

    def __len__(self):
        return len(self.file_lists)

    def __getitem__(self, idx):
        file_path = os.path.join(self.dir, self.file_lists.iloc[idx, 0].replace('/','\\'))
        label = self.file_lists.iloc[idx, 1]
        labels = np.array([1,0])
        if label == 1:
            labels = np.array([0,1])
        pkl_file = open(file_path, 'rb')
        results = pickle.load(pkl_file)
        results = np.array(results)
        pkl_file.close()

        results = torch.tensor(results).type(torch.FloatTensor)
        labels = torch.tensor(labels).type(torch.FloatTensor)
        return (results[0].squeeze(), results[1].squeeze(), labels)
        # if results.shape[0] > 1:
        #     return (results[0].squeeze(), results[1].squeeze(), labels)
        # else:
        #     return (results[0].squeeze(), results[0].squeeze(), labels)