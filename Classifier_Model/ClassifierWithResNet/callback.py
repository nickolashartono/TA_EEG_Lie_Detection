from pytorch_lightning.callbacks import Callback
from torchmetrics.classification import BinaryF1Score, BinaryAccuracy
import numpy as np
import torch


class LoggingCallback(Callback):
    def __init__(self):
        super().__init__()
        self.f1 = BinaryF1Score()
        self.acc = BinaryAccuracy()

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pl_module.predictions = np.array([])
        pl_module.targets = np.array([])

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pl_module.f1_train = pl_module.f1_fn(torch.tensor(pl_module.predictions.squeeze()), torch.tensor(pl_module.targets.squeeze()))
        # pl_module.acc_train = pl_module.acc_fn(torch.tensor(pl_module.predictions.squeeze()), torch.tensor(pl_module.targets.squeeze()))
        pl_module.log('F1_Score_train', pl_module.f1_train)
        # pl_module.log('Accuracy_train', pl_module.acc_train)

    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pl_module.predictions_val = np.array([])
        pl_module.targets_val = np.array([])

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pl_module.f1_val = pl_module.f1_fn(torch.tensor(pl_module.predictions_val.squeeze()), torch.tensor(pl_module.targets_val.squeeze()))
        # pl_module.acc_val = pl_module.acc_fn(torch.tensor(pl_module.predictions_val.squeeze()), torch.tensor(pl_module.targets_val.squeeze()))
        pl_module.log('F1_Score_val', pl_module.f1_val)
        # pl_module.log('Accuracy_val', pl_module.acc_val)
        # pl_module.test_each_epoch()
        # f1_avg = torch.mean(torch.tensor([pl_module.f1_train,pl_module.f1_val,pl_module.f1_test]))
        # pl_module.log('F1_Score_avg', f1_avg)