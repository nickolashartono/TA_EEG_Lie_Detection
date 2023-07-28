from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
import cv2
import pickle
import os

class ImgDatasetLoader(Dataset):
    def __init__(self, annotations_file, img_dir):
        self.file_lists = pd.read_csv(annotations_file, header=None)
        self.img_dir = img_dir
        self.eeg_names = ['Fp1', 'Fp2', 'F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4', 'Oz', 'VEOG', 'HEOG']

    def __len__(self):
        return len(self.file_lists)

    def __getitem__(self, idx):
        control = self.file_lists.iloc[idx, 0]
        relevant = self.file_lists.iloc[idx, 0]
        label = self.file_lists.iloc[idx, 1]

        control_img = []
        relevant_img = []

        for j in self.eeg_names:
            temp_control = cv2.imread(control + j + '.png', cv2.IMREAD_GRAYSCALE)
            temp_relevant = cv2.imread(relevant + j + '.png', cv2.IMREAD_GRAYSCALE)
            control_img.append(temp_control)
            relevant_img.append(temp_relevant)

        control_img = torch.tensor(np.array(control_img)).type(torch.FloatTensor)
        relevant_img = torch.tensor(np.array(relevant_img)).type(torch.FloatTensor)

        return (control_img, relevant_img, label)