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
from model_builder import LitClassifierWithEfficientNetV2S as classifier_64
from model_builder_128 import LitClassifierWithEfficientNetV2S as classifier_128
from model_builder_single import LitClassifierWithEfficientNetV2S as classifier_single
from model_builder_mulhead_linear import LitClassifierWithEfficientNetV2S as classifier_att_linear
from model_builder_mulhead_conv import LitClassifierWithEfficientNetV2S as classifier_att_conv
from model_builder_mulhead_conv_2 import LitClassifierWithEfficientNetV2S as classifier_att_conv_2
from model_builder_mulhead_conv_3 import LitClassifierWithEfficientNetV2S as classifier_att_conv_3
from model_builder_mulhead_conv_5 import LitClassifierWithEfficientNetV2S as classifier_att_conv_5
from model_builder_mulhead_conv_6 import LitClassifierWithEfficientNetV2S as classifier_att_conv_6
from model_builder_baghel import LitClassifierWithEfficientNetV2S as classifier_baghel
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
SAVE_DIR = 'E:\\Nicko\\TUGAS_AKHIR\\ClasifierWithEfficientNetV2S\\Logs\\'
RUN_NAME = 'Run_1_allfreeze'
DATA_NAME = args.data
PROJECT_NAME = 'classifier_baghel'
# PKL_DIR = 'D:\\Nicko\\TUGAS_AKHIR\\Dataset_new_' + DATA_NAME +'\\Dataset_TA_pkl\\LATENT_32640_2CHANNEL\\'
# PKL_DIR = 'D:\\Nicko\\TUGAS_AKHIR\\Dataset_Dryad\\dataset_pkl\\'
PKL_DIR = 'D:\\Nicko\\TUGAS_AKHIR\\Dataset_new_' + DATA_NAME + '\\Dataset_TA_raw_pkl\\4_channel\\'
# PKL_DIR = 'D:\\Nicko\\TUGAS_AKHIR\\Dataset_new_' + DATA_NAME + '\\Dataset_TA_raw_pkl\\4_channel\\'
TRAIN_DIR = PKL_DIR + 'train_set.csv'
VALID_DIR = PKL_DIR + 'val_set.csv'
TEST_DIR = PKL_DIR + 'testing_set.csv'
# TEST_DIR_35 = PKL_DIR + 'test_set_with_35.csv'
# IMG_DIR = 'D:\\Nicko\\TUGAS_AKHIR\\Dataset_new_' + DATA_NAME +'\\Dataset_TA_img\\'
# IMG_DIR = 'D:\\Nicko\\TUGAS_AKHIR\\Dataset_Dryad\\dataset_img\\'
# TRAIN_DIR = IMG_DIR + 'img_train_set.csv'
# VALID_DIR = IMG_DIR + 'img_val_set.csv'
# TEST_DIR = IMG_DIR + 'img_test_set.csv'
RUN_NUMBER = 'Run_' + str(args.run) + '_'

# PARAMS
NUM_EPOCHS = 2000
BATCH_SIZE = 50
LEARNING_RATE = 1e-4

if args.with35 == 'yes':
    # DATALOADER
    # train_dataloader, valid_dataloader, test_dataloader = data_setup.create_dataloaders(
    #     train_dir=TRAIN_DIR,
    #     valid_dir=VALID_DIR,
    #     test_dir=TEST_DIR_35,
    #     pkl_dir=PKL_DIR,
    #     batch_size=BATCH_SIZE
    # )
    pass
else:
    train_dataloader, valid_dataloader, test_dataloader = data_setup.create_dataloaders(
        train_dir=TRAIN_DIR,
        valid_dir=VALID_DIR,
        test_dir=TEST_DIR,
        pkl_dir=PKL_DIR,
        batch_size=BATCH_SIZE
    )
    # train_dataloader, valid_dataloader, test_dataloader = data_setup.create_dataloaders(
    #     train_dir=TRAIN_DIR,
    #     valid_dir=VALID_DIR,
    #     test_dir=TEST_DIR,
    #     pkl_dir=IMG_DIR,
    #     batch_size=BATCH_SIZE
    # )

# MODEL
pl.seed_everything(42, workers=True)
ckpt_dir = 'E:\\Nicko\\TUGAS_AKHIR\\ClasifierWithEfficientNetV2S\\Logs\\runs_' + DATA_NAME +'_baghel\\'
for idx, dir in enumerate(os.listdir(ckpt_dir)):
    if RUN_NUMBER in dir:
        ckpt_dir += dir + '\\'
        break

for ckpt_name in os.listdir(ckpt_dir):
    ckpt_path = ckpt_dir + ckpt_name
    print(ckpt_path)
    if args.latent == 'sub':
        classifier_loaded = classifier_64(learning_rate=LEARNING_RATE, test_dataloader=test_dataloader).load_from_checkpoint(checkpoint_path=ckpt_path)
    elif args.latent == 'cat':
        classifier_loaded = classifier_128(learning_rate=LEARNING_RATE, test_dataloader=test_dataloader).load_from_checkpoint(checkpoint_path=ckpt_path)
    elif args.latent == 'attlin':
        classifier_loaded = classifier_att_linear(learning_rate=LEARNING_RATE, test_dataloader=test_dataloader).load_from_checkpoint(checkpoint_path=ckpt_path)
    elif args.latent == 'attconv':
        classifier_loaded = classifier_att_conv(learning_rate=LEARNING_RATE, test_dataloader=test_dataloader).load_from_checkpoint(checkpoint_path=ckpt_path)
    elif args.latent == 'attconv2':
        classifier_loaded = classifier_att_conv_2(learning_rate=LEARNING_RATE, test_dataloader=test_dataloader).load_from_checkpoint(checkpoint_path=ckpt_path)
    elif args.latent == 'attconv3':
        classifier_loaded = classifier_att_conv_3(learning_rate=LEARNING_RATE, test_dataloader=test_dataloader).load_from_checkpoint(checkpoint_path=ckpt_path)
    elif args.latent == 'attconv5':
        classifier_loaded = classifier_att_conv_5(learning_rate=LEARNING_RATE, test_dataloader=test_dataloader).load_from_checkpoint(checkpoint_path=ckpt_path)
    elif args.latent == 'attconv6':
        classifier_loaded = classifier_att_conv_6(learning_rate=LEARNING_RATE, test_dataloader=test_dataloader).load_from_checkpoint(checkpoint_path=ckpt_path)
    elif args.latent == 'baghel':
        classifier_loaded = classifier_baghel(learning_rate=LEARNING_RATE, test_dataloader=test_dataloader).load_from_checkpoint(checkpoint_path=ckpt_path)
    else:
        classifier_loaded = classifier_single(learning_rate=LEARNING_RATE, test_dataloader=test_dataloader).load_from_checkpoint(checkpoint_path=ckpt_path)

    # TRAINER
    trainer = pl.Trainer(max_epochs=NUM_EPOCHS, devices=1, accelerator='gpu', log_every_n_steps=9, logger=False)
    print(ckpt_name)
    trainer.test(classifier_loaded, test_dataloader)
