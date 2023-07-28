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
from model_builder_big import LitClassifierWithEfficientNetV2S as classifier_big

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, required=True)
parser.add_argument('--epoch', type=int, required=True)
parser.add_argument('--dropout', type=float, required=False)
parser.add_argument('--setup', type=str, required=True)
args = parser.parse_args()

# DIRECTORY
SAVE_DIR = 'E:\\Nicko\\TUGAS_AKHIR\\ClasifierBigModel\\Logs\\'
RUN_NAME = 'Run_' + str((len(os.listdir(SAVE_DIR+'runs'))+1))
PROJECT_NAME = 'classifier_bigmodel'
# DATA_LIST = [6,9,12]
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
        TRAIN_DIR.append(i + 'stack\\meta_set.csv')
        VALID_DIR.append(i + 'stack\\meta_val_set.csv')
        TEST_DIR.append(i + 'test_set.csv')
print(RUN_NAME)

# PARAMS
NUM_EPOCHS = args.epoch
BATCH_SIZE = 50
LEARNING_RATE = args.lr if args.lr else 1e-4
DROPOUT_RATE = args.dropout if args.dropout else 0.2
print(DROPOUT_RATE)

# # DATALOADER
train_dataloader, valid_dataloader, test_dataloader = data_setup_big.create_dataloaders(
    train_dir=TRAIN_DIR,
    valid_dir=VALID_DIR,
    test_dir=TEST_DIR,
    pkl_dir=PKL_DIR_LIST,
    batch_size=BATCH_SIZE
)

# LOGGER
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor="F1_Score_avg",
    mode="max",
    dirpath=SAVE_DIR+"runs"+"\\"+RUN_NAME,
    filename="classifier-{epoch:02d}-{F1_Score_avg:.2f}",
    save_top_k=5,
    save_last=True
)
logging_callback = LoggingCallback()
wandb_logger = WandbLogger(project=PROJECT_NAME, name=RUN_NAME, save_dir=SAVE_DIR)

# MODEL
pl.seed_everything(42, workers=True)
big_classifier = classifier_big(
        learning_rate=LEARNING_RATE,
        dropout=DROPOUT_RATE
)

# # TRAINER
trainer = pl.Trainer(
    max_epochs=NUM_EPOCHS,
    devices=1,
    accelerator='gpu',
    logger=wandb_logger,
    log_every_n_steps=3,
    callbacks=[checkpoint_callback, logging_callback]
)
trainer.fit(big_classifier, train_dataloader, valid_dataloader)
# trainer.test(big_classifier, test_dataloader)
wandb.finish()
