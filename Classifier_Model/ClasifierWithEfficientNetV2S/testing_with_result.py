import pandas as pd
import numpy as np
import os
import torch
import pytorch_lightning as pl
from torchvision import models
from pytorch_lightning.loggers import WandbLogger
import wandb
from callback import LoggingCallback
import data_setup
from model_builder import LitClassifierWithEfficientNetV2S
from torchmetrics.classification import BinaryF1Score, BinaryAccuracy
import argparse
import wandb
import os

parser = argparse.ArgumentParser()
parser.add_argument('--run', type=int, required=True)
args = parser.parse_args()

# DIRECTORY
SAVE_DIR = 'D:\\Nicko\\TUGAS_AKHIR\\Classifier\\ClasifierWithEfficientNetV2S\\Logs\\'
RUN_NAME = 'Run_1_allfreeze'
PROJECT_NAME = 'classifier_efficientnetv2s'
# PKL_DIR = 'D:\\Nicko\\TUGAS_AKHIR\\Dataset_new\\Dataset_TA_pkl\\LATENT_32640_2CHANNEL\\'
PKL_DIR = 'D:\\Nicko\\TUGAS_AKHIR\\Dataset_new_no_norm\\Dataset_TA_pkl\\LATENT_32640_2CHANNEL\\'
TRAIN_DIR = PKL_DIR + 'train_set.csv'
VALID_DIR = PKL_DIR + 'valid_set.csv'
TEST_DIR = PKL_DIR + 'test_set.csv'
RUN_NUMBER = 'Run_' + str(args.run) + '_'
ACC_FUNC = BinaryAccuracy()

# PARAMS
NUM_EPOCHS = 2000
BATCH_SIZE = 30
LEARNING_RATE = 1e-4

# DATALOADER
train_dataloader, valid_dataloader, test_dataloader = data_setup.create_dataloaders(
    train_dir=TRAIN_DIR,
    valid_dir=VALID_DIR,
    test_dir=TEST_DIR,
    pkl_dir=PKL_DIR,
    batch_size=BATCH_SIZE
)

# MODEL
ckpt_dir = 'D:\\Nicko\\TUGAS_AKHIR\\Classifier\\ClasifierWithEfficientNetV2S\\Logs\\runs\\'
# ckpt_dir = 'D:\\Nicko\\TUGAS_AKHIR\\Classifier\\ClasifierWithEfficientNetV2S\\Logs\\runs_no_norm\\'
for idx, dir in enumerate(os.listdir(ckpt_dir)):
    if RUN_NUMBER in dir:
        ckpt_dir += dir + '\\'
        break

for ckpt_name in os.listdir(ckpt_dir):
    ckpt_path = ckpt_dir + ckpt_name
    classifier_loaded = LitClassifierWithEfficientNetV2S(learning_rate=LEARNING_RATE, test_dataloader=test_dataloader).load_from_checkpoint(checkpoint_path=ckpt_path)

    classifier_loaded.eval()
    targets = np.array([])
    preds = np.array([])
    with torch.no_grad():
        for x1, x2, y in test_dataloader:
            y_logits = classifier_loaded(x1, x2)
            y_logits = y_logits.squeeze()
            y_preds = torch.round(torch.sigmoid(y_logits.cpu())).detach().numpy()
            preds = np.append(preds, y_preds, axis=0)
            targets = np.append(targets, y.cpu(), axis=0)
    print('Checkpoint', ckpt_name)
    print('Targets    :', targets)
    print('Prediction :', preds)
    print('Accuration :', ACC_FUNC(torch.tensor(preds), torch.tensor(targets)))
    print('===============================================')