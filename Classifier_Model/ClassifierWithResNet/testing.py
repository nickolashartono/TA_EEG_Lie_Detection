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
from model_builder_128 import LitClassifierWithResNet50 as classifier_128
import argparse
import wandb
import os

parser = argparse.ArgumentParser()
parser.add_argument('--run', type=int, required=True)
parser.add_argument('--with35', type=str, required=True)
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--latent', type=str, required=True)
args = parser.parse_args()

# DIRECTORY
# SAVE_DIR = 'D:\\Nicko\\TUGAS_AKHIR\\Classifier\\ClasifierWithEfficientNetV2S\\Logs\\'
SAVE_DIR = 'E:\\Nicko\\TUGAS_AKHIR\\ClasifierWithEfficientNetV2S\\Logs\\'
RUN_NAME = 'Run_1_allfreeze'
DATA_NAME = args.data
# PROJECT_NAME = 'classifier_efficientnetv2s'
PROJECT_NAME = 'classifier_efficientnetv2s_nonorm_nodivide_microvolt'
# PKL_DIR = 'D:\\Nicko\\TUGAS_AKHIR\\Dataset_new\\Dataset_TA_pkl\\LATENT_32640_2CHANNEL\\'
# PKL_DIR = 'D:\\Nicko\\TUGAS_AKHIR\\Dataset_new_no_norm\\Dataset_TA_pkl\\LATENT_32640_2CHANNEL\\'
PKL_DIR = 'D:\\Nicko\\TUGAS_AKHIR\\Dataset_new_' + DATA_NAME +'\\Dataset_TA_pkl\\LATENT_32640_2CHANNEL\\'
TRAIN_DIR = PKL_DIR + 'train_set.csv'
VALID_DIR = PKL_DIR + 'valid_set.csv'
TEST_DIR = PKL_DIR + 'test_set.csv'
TEST_DIR_35 = PKL_DIR + 'test_set_with_35.csv'
RUN_NUMBER = 'Run_' + str(args.run) + '_'

# PARAMS
NUM_EPOCHS = 2000
BATCH_SIZE = 50
LEARNING_RATE = 1e-4

if args.with35 == 'yes':
    # DATALOADER
    train_dataloader, valid_dataloader, test_dataloader = data_setup.create_dataloaders(
        train_dir=TRAIN_DIR,
        valid_dir=VALID_DIR,
        test_dir=TEST_DIR_35,
        pkl_dir=PKL_DIR,
        batch_size=BATCH_SIZE
    )
else:
    train_dataloader, valid_dataloader, test_dataloader = data_setup.create_dataloaders(
        train_dir=TRAIN_DIR,
        valid_dir=VALID_DIR,
        test_dir=TEST_DIR,
        pkl_dir=PKL_DIR,
        batch_size=BATCH_SIZE
    )

# MODEL
pl.seed_everything(42, workers=True)
# ckpt_dir = 'D:\\Nicko\\TUGAS_AKHIR\\Classifier\\ClasifierWithEfficientNetV2S\\Logs\\runs\\'
# ckpt_dir = 'D:\\Nicko\\TUGAS_AKHIR\\Classifier\\ClasifierWithEfficientNetV2S\\Logs\\runs_no_norm\\'
# ckpt_dir = 'D:\\Nicko\\TUGAS_AKHIR\\Classifier\\ClasifierWithEfficientNetV2S\\Logs\\runs_' + DATA_NAME +'\\'
ckpt_dir = 'E:\\Nicko\\TUGAS_AKHIR\\ClasifierWithResnet50\\Logs\\runs_' + DATA_NAME +'\\'
for idx, dir in enumerate(os.listdir(ckpt_dir)):
    if RUN_NUMBER in dir:
        ckpt_dir += dir + '\\'
        break

for ckpt_name in os.listdir(ckpt_dir):
    ckpt_path = ckpt_dir + ckpt_name
    if args.latent == 'sub':
        pass
    elif args.latent == 'cat':
        classifier_loaded = classifier_128(learning_rate=LEARNING_RATE, test_dataloader=test_dataloader).load_from_checkpoint(checkpoint_path=ckpt_path)
    elif args.latent == 'attlin':
        pass
    elif args.latent == 'attconv':
        pass
    elif args.latent == 'attconv2':
        pass
    else:
        pass

    # TRAINER
    trainer = pl.Trainer(max_epochs=NUM_EPOCHS, devices=1, accelerator='gpu', log_every_n_steps=9, logger=False)
    print(ckpt_name)
    trainer.test(classifier_loaded, test_dataloader)
