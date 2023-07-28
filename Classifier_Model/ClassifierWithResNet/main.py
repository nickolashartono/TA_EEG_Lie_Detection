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

parser = argparse.ArgumentParser()
parser.add_argument('--unfreeze', type=str, required=False)
parser.add_argument('--lr', type=float, required=True)
parser.add_argument('--epoch', type=int, required=True)
parser.add_argument('--dropout', type=float, required=False)
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--latent', type=str, required=True)
args = parser.parse_args()

# DIRECTORY
SAVE_DIR = 'E:\\Nicko\\TUGAS_AKHIR\\ClasifierWithResnet50\\Logs\\'
DATA_NAME = args.data
RUN_NAME = 'Run_' + str((len(os.listdir(SAVE_DIR+'runs_'+DATA_NAME))+1))
RUN_NAME = RUN_NAME+('_layer_' + args.unfreeze + '_train') if args.unfreeze else  RUN_NAME+'_allfreeze'
PROJECT_NAME = 'classifier_resnet50_' + DATA_NAME
PKL_DIR = 'D:\\Nicko\\TUGAS_AKHIR\\Dataset_new_' + DATA_NAME + '\\Dataset_TA_pkl\\LATENT_32640_2CHANNEL\\'
TRAIN_DIR = PKL_DIR + 'train_set.csv'
VALID_DIR = PKL_DIR + 'valid_set.csv'
TEST_DIR = PKL_DIR + 'test_set.csv'
print(RUN_NAME)

# PARAMS
NUM_EPOCHS = args.epoch
BATCH_SIZE = 50
LEARNING_RATE = args.lr if args.lr else 1e-4
UNFREEZE_LAYER = args.unfreeze.split(',') if args.unfreeze else None
DROPOUT_RATE = args.dropout if args.dropout else 0.2
print(DROPOUT_RATE)

# # DATALOADER
train_dataloader, valid_dataloader, test_dataloader = data_setup.create_dataloaders(
    train_dir=TRAIN_DIR,
    valid_dir=VALID_DIR,
    test_dir=TEST_DIR,
    pkl_dir=PKL_DIR,
    batch_size=BATCH_SIZE
)

# LOGGER
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor="F1_Score_val",
    mode="max",
    dirpath=SAVE_DIR+"runs_"+DATA_NAME+"\\"+RUN_NAME,
    filename="classifier-{epoch:02d}-{F1_Score_val:.2f}",
    save_top_k=5,
    save_last=True
)
logging_callback = LoggingCallback()
wandb_logger = WandbLogger(project=PROJECT_NAME, name=RUN_NAME, save_dir=SAVE_DIR)

# MODEL
pl.seed_everything(42, workers=True)
if args.latent == 'sub':
    pass
elif args.latent == 'cat':
    print('CONCAT LATENT')
    classifier = classifier_128(
        learning_rate=LEARNING_RATE,
        unfreeze_layer=UNFREEZE_LAYER,
        test_dataloader=test_dataloader,
        dropout=DROPOUT_RATE
    )
elif args.latent == 'attlin':
    print('ATTENTION LINEAR LATENT')
    pass
elif args.latent == 'attconv':
    print('ATTENTION CONV LATENT')
    pass
elif args.latent == 'attconv2':
    print('ATTENTION CONV 2 LATENT')
    pass
else:
    print('SINGLE LATENT')
    pass

# TRAINER
trainer = pl.Trainer(
    max_epochs=NUM_EPOCHS,
    devices=1,
    accelerator='gpu',
    logger=wandb_logger,
    log_every_n_steps=3,
    callbacks=[checkpoint_callback, logging_callback]
)
trainer.fit(classifier, train_dataloader, valid_dataloader)
trainer.test(classifier, test_dataloader)
wandb.finish()
