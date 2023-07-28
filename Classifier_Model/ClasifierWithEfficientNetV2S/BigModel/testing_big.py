import pandas as pd
import numpy as np
import os
import torch
import pytorch_lightning as pl
from torchvision import models
from pytorch_lightning.loggers import WandbLogger
import wandb
from callback import LoggingCallback
import data_setup_big
import argparse
import wandb
import os
from model_builder_big import LitClassifierWithEfficientNetV2S as classifier_big

parser = argparse.ArgumentParser()
parser.add_argument('--run', type=int, required=True)
parser.add_argument('--setup', type=str, required=True)
args = parser.parse_args()

# DIRECTORY
# SAVE_DIR = 'D:\\Nicko\\TUGAS_AKHIR\\Classifier\\ClasifierWithEfficientNetV2S\\Logs\\'
SAVE_DIR = 'E:\\Nicko\\TUGAS_AKHIR\\ClasifierWithEfficientNetV2S\\Logs\\'
PROJECT_NAME = 'classifier_bigmodel'
DATA_LIST = [12,12,12]
PKL_DIR_LIST = []
for i in DATA_LIST:
    PKL_DIR_LIST.append('D:\\Nicko\\TUGAS_AKHIR\\Dataset_new_' + str(i) + '\\Dataset_TA_pkl\\LATENT_32640_2CHANNEL\\')

TRAIN_DIR = []
VALID_DIR = []
TEST_DIR = []
for i in PKL_DIR_LIST:
    if args.setup == 'yes':
        TRAIN_DIR.append(i + 'train_set_setup.csv')
        VALID_DIR.append(i + 'valid_set_setup.csv')
        TEST_DIR.append(i + 'test_set_setup.csv')
    else:
        TRAIN_DIR.append(i + 'train_set.csv')
        VALID_DIR.append(i + 'valid_set.csv')
        # TEST_DIR.append(i + 'test_set.csv')
        TEST_DIR.append(i + 'test_out3.csv')
RUN_NUMBER = 'Run_' + str(args.run)

# PARAMS
NUM_EPOCHS = 2000
BATCH_SIZE = 30
LEARNING_RATE = 1e-4

train_dataloader, valid_dataloader, test_dataloader = data_setup_big.create_dataloaders(
    train_dir=TRAIN_DIR,
    valid_dir=VALID_DIR,
    test_dir=TEST_DIR,
    pkl_dir=PKL_DIR_LIST,
    batch_size=BATCH_SIZE
)

# MODEL
pl.seed_everything(42, workers=True)
ckpt_dir = 'E:\\Nicko\\TUGAS_AKHIR\\ClasifierBigModel\\Logs\\runs' +'\\'
for idx, dir in enumerate(os.listdir(ckpt_dir)):
    if RUN_NUMBER in dir:
        ckpt_dir += dir + '\\'
        break

for ckpt_name in os.listdir(ckpt_dir):
    ckpt_path = ckpt_dir + ckpt_name
    big_classifier = classifier_big(
        learning_rate=LEARNING_RATE
    ).load_from_checkpoint(checkpoint_path=ckpt_path)

    # TRAINER
    trainer = pl.Trainer(max_epochs=NUM_EPOCHS, devices=1, accelerator='gpu', log_every_n_steps=9, logger=False)
    print(ckpt_name)
    trainer.test(big_classifier, test_dataloader)
