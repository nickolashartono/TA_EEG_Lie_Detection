import torchinfo
import pytorch_lightning as pl
from torchmetrics.classification import BinaryF1Score, BinaryAccuracy
from torchmetrics.classification import MulticlassF1Score, MulticlassAccuracy
from sklearn import metrics
import matplotlib.pyplot as plt
import torch
import numpy as np
from torchvision import models,ops

class LitClassifierWithEfficientNetV2S(pl.LightningModule):
    def __init__(self, unfreeze_layer=None, test_dataloader=None, learning_rate=1e-3, dropout=0.2):
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.dropout = dropout

        # variable
        self.predictions = np.array([])
        self.targets = np.array([])
        self.predictions_val = np.array([])
        self.targets_val = np.array([])
        self.predictions_test = np.array([])
        self.targets_test = np.array([])
        self.f1_fn = BinaryF1Score()
        self.acc_fn = BinaryAccuracy()
        self.f1_train = 0
        self.f1_val = 0
        self.f1_test = 0
        self.acc_train = 0
        self.acc_val = 0
        self.acc_test = 0

        # model
        self.model = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=4,out_channels=32, kernel_size=3, stride=3),
            torch.nn.ReLU(True),
            torch.nn.Conv1d(in_channels=32,out_channels=64, kernel_size=3, stride=2),
            torch.nn.ReLU(True),
            torch.nn.MaxPool1d(kernel_size=3),
            torch.nn.Conv1d(in_channels=64, out_channels=96, kernel_size=3, stride=2),
            torch.nn.ReLU(True),
            torch.nn.MaxPool1d(kernel_size=3),
            torch.nn.Conv1d(in_channels=96, out_channels=128, kernel_size=3, stride=2),
            torch.nn.ReLU(True),
            torch.nn.MaxPool1d(kernel_size=3),
            torch.nn.Dropout(p=0.25),
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(in_features=256, out_features=150),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(in_features=150, out_features=2),
            torch.nn.Softmax(dim=1)
        )


    def forward(self, x1, x2):
        x_combined = torch.cat((x1, x2), 1)
        y_logits = self.model(x_combined)
        y_pred = torch.argmax(y_logits, dim=1)
        return y_pred

    def training_step(self, batch, batch_idx):
        x1, x2, y = batch
        x_combined = torch.cat((x1, x2), 1)
        y_logits = self.model(x2)
        y_logits = y_logits.squeeze()
        y_preds = torch.argmax(y_logits, dim=1)
        loss = self.loss_fn(y_logits, y)
        acc = self.acc_fn(y_preds, torch.argmax(y, dim=1))
        y_preds = torch.argmax(y_logits.cpu(), dim=1).detach().numpy()
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        self.predictions = np.append(self.predictions, y_preds, axis=0)
        self.targets = np.append(self.targets,  torch.argmax(y, dim=1).cpu(), axis=0)
        return {"loss": loss, "acc": acc}

    def validation_step(self, val_batch, batch_idx):
        x1, x2, y = val_batch
        x_combined = torch.cat((x1, x2), 1)
        y_logits = self.model(x2)
        y_logits = y_logits.squeeze()
        y_preds = torch.argmax(y_logits, dim=1)
        val_loss = self.loss_fn(y_logits, y)
        val_acc = self.acc_fn(y_preds, torch.argmax(y, dim=1))
        self.log("val_loss", val_loss)
        self.log("val_acc", val_acc)
        y_preds = torch.argmax(y_logits.cpu(), dim=1).detach().numpy()
        self.predictions_val = np.append(self.predictions_val, y_preds, axis=0)
        self.targets_val = np.append(self.targets_val, torch.argmax(y, dim=1).cpu(), axis=0)
        return {"val_loss": val_loss, "val_acc": val_acc}

    def test_step(self, test_batch, batch_idx):
        x1, x2, y = test_batch
        x_combined = torch.cat((x1, x2), 1)
        y_logits = self.model(x2)
        y_logits = y_logits.squeeze()
        y_preds = torch.argmax(y_logits, dim=1)
        test_loss = self.loss_fn(y_logits, y)
        test_acc = self.acc_fn(y_preds, torch.argmax(y, dim=1))
        self.log("test_loss", test_loss)
        self.log("test_acc", test_acc)
        # y_preds = torch.argmax(y_logits.cpu(), dim=1).detach().numpy()
        # conf_matrix = metrics.confusion_matrix(torch.argmax(y.cpu(), dim=1), y_preds)
        # conf_matrix = np.flip(conf_matrix).T
        # legend = ['Lie','Truth']
        # legend2 = [['(TP)','(FP)'],['(FN)','(TN)']]
        # fig, ax = plt.subplots(figsize=(5, 5))
        # ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
        # ax.set_xticklabels([''] + legend)
        # ax.set_yticklabels([''] + legend)
        # for i in range(conf_matrix.shape[0]):
        #     for j in range(conf_matrix.shape[1]):
        #         ax.text(x=j, y=i, s=(str(conf_matrix[i, j]) + ' ' + legend2[i][j]), va='center', ha='center', size='xx-large')
        #
        # plt.ylabel('Predictions', fontsize=20)
        # plt.title('Actual', fontsize=20)
        # plt.xticks(fontsize=18)
        # plt.yticks(fontsize=18)
        # plt.tight_layout(pad=1)
        # y_preds = torch.round(torch.sigmoid(y_logits))
        # f1score = self.f1_fn(y_preds, y)
        # print('F1-score:', f1score)
        # plt.show()
        return {"test_loss": test_loss, "test_acc": test_acc}

    def configure_optimizers(self):
        optimizer = torch.optim.Adadelta(self.parameters(), lr=self.learning_rate)
        return optimizer
