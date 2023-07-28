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
from model_builder_mulhead_linear_2 import LitClassifierWithEfficientNetV2S as classifier_att_linear_2
from model_builder_mulhead_conv_4 import LitClassifierWithEfficientNetV2S as classifier_att_conv_4
from model_builder_mulhead_conv_5 import LitClassifierWithEfficientNetV2S as classifier_att_conv_5
from model_builder_mulhead_conv_6 import LitClassifierWithEfficientNetV2S as classifier_att_conv_6
from model_builder_baghel import LitClassifierWithEfficientNetV2S as classifier_baghel
import argparse
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--unfreeze', type=str, required=False)
parser.add_argument('--lr', type=float, required=True)
parser.add_argument('--epoch', type=int, required=True)
parser.add_argument('--dropout', type=float, required=False)
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--latent', type=str, required=True)
parser.add_argument('--setup', type=str, required=True)
args = parser.parse_args()

# DIRECTORY
SAVE_DIR = 'E:\\Nicko\\TUGAS_AKHIR\\ClasifierWithEfficientNetV2S\\Logs\\'
DATA_NAME = args.data
# RUN_NAME = 'Run_' + str((len(os.listdir(SAVE_DIR+'runs_'+DATA_NAME))+1))
RUN_NAME = 'Run_' + str((len(os.listdir(SAVE_DIR+'runs_'+DATA_NAME+'_baghel'))+1))
RUN_NAME = RUN_NAME+('_feature_' + args.unfreeze + '_train') if args.unfreeze else  RUN_NAME+'_allfreeze'
# PROJECT_NAME = 'classifier_efficientnetv2s_' + DATA_NAME
PROJECT_NAME = 'classifier_baghel'
# PKL_DIR = 'D:\\Nicko\\TUGAS_AKHIR\\Dataset_new_' + DATA_NAME + '\\Dataset_TA_pkl\\LATENT_32640_2CHANNEL\\'
PKL_DIR = 'D:\\Nicko\\TUGAS_AKHIR\\Dataset_new_' + DATA_NAME + '\\Dataset_TA_raw_pkl\\4_channel\\'
# PKL_DIR = 'D:\\Nicko\\TUGAS_AKHIR\\Dataset_new_' + DATA_NAME + '\\Dataset_TA_raw_pkl\\4_channel\\'
# PKL_DIR = 'D:\\Nicko\\TUGAS_AKHIR\\Dataset_Dryad\\dataset_pkl\\'
# IMG_DIR = 'D:\\Nicko\\TUGAS_AKHIR\\Dataset_new_' + DATA_NAME +'\\Dataset_TA_img\\'
# IMG_DIR = 'D:\\Nicko\\TUGAS_AKHIR\\Dataset_Dryad\\dataset_img\\'
if args.setup == 'yes':
    TRAIN_DIR = PKL_DIR + 'train_set_setup.csv'
    VALID_DIR = PKL_DIR + 'valid_set_setup.csv'
    TEST_DIR = PKL_DIR + 'test_set_setup.csv'
else:
    TRAIN_DIR = PKL_DIR + 'train_set.csv'
    VALID_DIR = PKL_DIR + 'val_set.csv'
    TEST_DIR = PKL_DIR + 'testing_set.csv'
    # TRAIN_DIR = IMG_DIR + 'img_train_set.csv'
    # VALID_DIR = IMG_DIR + 'img_val_set.csv'
    # TEST_DIR = IMG_DIR + 'img_test_set.csv'
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
    # pkl_dir=IMG_DIR,
    batch_size=BATCH_SIZE
)

# LOGGER
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    # monitor="F1_Score_val",
    monitor="F1_Score_avg",
    mode="max",
    # monitor="val_loss",
    # mode="min",
    dirpath=SAVE_DIR+"runs_"+DATA_NAME+"_baghel"+"\\"+RUN_NAME,
    filename="classifier-{epoch:02d}-{F1_Score_avg:.2f}",
    save_top_k=5,
    save_last=True
)
logging_callback = LoggingCallback()
wandb_logger = WandbLogger(project=PROJECT_NAME, name=RUN_NAME, save_dir=SAVE_DIR)

# MODEL
pl.seed_everything(42, workers=True)
if args.latent == 'sub':
    print('SUBTRACT LATENT')
    classifier = classifier_64(
        learning_rate=LEARNING_RATE,
        unfreeze_layer=UNFREEZE_LAYER,
        test_dataloader=test_dataloader,
        dropout=DROPOUT_RATE
    )
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
    classifier = classifier_att_linear(
        learning_rate=LEARNING_RATE,
        unfreeze_layer=UNFREEZE_LAYER,
        test_dataloader=test_dataloader,
        dropout=DROPOUT_RATE
    )
elif args.latent == 'attconv':
    print('ATTENTION CONV LATENT')
    classifier = classifier_att_conv(
        learning_rate=LEARNING_RATE,
        unfreeze_layer=UNFREEZE_LAYER,
        test_dataloader=test_dataloader,
        dropout=DROPOUT_RATE
    )
elif args.latent == 'attconv2':
    print('ATTENTION CONV 2 LATENT')
    classifier = classifier_att_conv_2(
        learning_rate=LEARNING_RATE,
        unfreeze_layer=UNFREEZE_LAYER,
        test_dataloader=test_dataloader,
        dropout=DROPOUT_RATE
    )
elif args.latent == 'attconv3':
    print('ATTENTION CONV 3 LATENT')
    classifier = classifier_att_conv_3(
        learning_rate=LEARNING_RATE,
        unfreeze_layer=UNFREEZE_LAYER,
        test_dataloader=test_dataloader,
        dropout=DROPOUT_RATE
    )
elif args.latent == 'attlin2':
    print('ATTENTION LINEAR 2 LATENT')
    classifier = classifier_att_linear_2(
        learning_rate=LEARNING_RATE,
        unfreeze_layer=UNFREEZE_LAYER,
        test_dataloader=test_dataloader,
        dropout=DROPOUT_RATE
    )
elif args.latent == 'attconv4':
    print('ATTENTION CONV 4 LATENT')
    classifier = classifier_att_conv_4(
        learning_rate=LEARNING_RATE,
        unfreeze_layer=UNFREEZE_LAYER,
        test_dataloader=test_dataloader,
        dropout=DROPOUT_RATE
    )
elif args.latent == 'attconv5':
    print('ATTENTION CONV 5 LATENT')
    classifier = classifier_att_conv_5(
        learning_rate=LEARNING_RATE,
        unfreeze_layer=UNFREEZE_LAYER,
        test_dataloader=test_dataloader,
        dropout=DROPOUT_RATE
    )
elif args.latent == 'attconv6':
    print('ATTENTION CONV 6 LATENT')
    classifier = classifier_att_conv_6(
        learning_rate=LEARNING_RATE,
        unfreeze_layer=UNFREEZE_LAYER,
        test_dataloader=test_dataloader,
        dropout=DROPOUT_RATE
    )
elif args.latent == 'baghel':
    print('BAGHEL METHOD')
    classifier = classifier_baghel(
        learning_rate=LEARNING_RATE,
        unfreeze_layer=UNFREEZE_LAYER,
        test_dataloader=test_dataloader,
        dropout=DROPOUT_RATE
    )
else:
    print('SINGLE LATENT')
    classifier = classifier_single(
        learning_rate=LEARNING_RATE,
        unfreeze_layer=UNFREEZE_LAYER,
        test_dataloader=test_dataloader,
        dropout=DROPOUT_RATE
    )


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
