import torchinfo
import pytorch_lightning as pl
from torchmetrics.classification import BinaryF1Score, BinaryAccuracy
from sklearn import metrics
import matplotlib.pyplot as plt
import torch
import numpy as np
from torchvision import models,ops
from torch import nn
import torch.nn.functional as F
from model_builder_128 import LitClassifierWithEfficientNetV2S as classifier_128
from model_builder_128_ext2 import LitClassifierWithEfficientNetV2S as classifier_128_ext2
from model_builder_128_ext3_norelu import LitClassifierWithEfficientNetV2S as classifier_128_ext3_norelu
from model_builder import LitClassifierWithEfficientNetV2S as classifier_64
from model_builder_single import LitClassifierWithEfficientNetV2S as classifier_single

class LitClassifierWithEfficientNetV2S(pl.LightningModule):
    def __init__(self, learning_rate=1e-3, dropout=0.2):
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
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

        # models
        # self.model1 = classifier_128()
        # self.model2 = classifier_64()
        # self.model3 = classifier_single()
        # self.model1 = classifier_128()
        # self.model2 = classifier_128_ext2()
        # self.model3 = classifier_128_ext3_norelu()
        # self.model4 = classifier_single()
        self.model1 = classifier_128()
        self.model2 = classifier_128()
        self.model3 = classifier_128()
        self._change_layer()
        #run1,6,9
        # self.meta_classifier = torch.nn.Sequential(
        #     torch.nn.Linear(in_features=4, out_features=16, bias=True),
        #     torch.nn.Linear(in_features=16, out_features=8, bias=True),
        #     torch.nn.Dropout(p=self.dropout, inplace=True),
        #     torch.nn.Linear(in_features=8, out_features=1, bias=True)
        # )

        #run2
        # self.meta_classifier = torch.nn.Sequential(
        #     torch.nn.Linear(in_features=3, out_features=3, bias=True),
        #     torch.nn.ReLU(True),
        #     torch.nn.Linear(in_features=3, out_features=3, bias=True),
        #     torch.nn.Dropout(p=self.dropout, inplace=True),
        #     torch.nn.Linear(in_features=3, out_features=1, bias=True)
        # )

        # run3,7
        self.meta_classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=3, out_features=64, bias=True),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm1d(64),
            torch.nn.Linear(in_features=64, out_features=32, bias=True),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm1d(32),
            torch.nn.Linear(in_features=32, out_features=16, bias=True),
            torch.nn.Dropout(p=self.dropout, inplace=True),
            torch.nn.Linear(in_features=16, out_features=1, bias=True)
        )

        # run4
        # self.meta_classifier = torch.nn.Sequential(
        #     torch.nn.Linear(in_features=3, out_features=32, bias=True),
        #     torch.nn.ReLU(True),
        #     torch.nn.Linear(in_features=32, out_features=16, bias=True),
        #     torch.nn.BatchNorm1d(16),
        #     torch.nn.Linear(in_features=16, out_features=8, bias=True),
        #     torch.nn.Dropout(p=self.dropout, inplace=True),
        #     torch.nn.Linear(in_features=8, out_features=1, bias=True)
        # )

        # run5
        # self.meta_classifier = torch.nn.Sequential(
        #     torch.nn.Linear(in_features=3, out_features=128, bias=True),
        #     torch.nn.ReLU(True),
        #     torch.nn.Linear(in_features=128, out_features=64, bias=True),
        #     torch.nn.BatchNorm1d(64),
        #     torch.nn.Linear(in_features=64, out_features=32, bias=True),
        #     torch.nn.Dropout(p=self.dropout, inplace=True),
        #     torch.nn.Linear(in_features=32, out_features=1, bias=True)
        # )

    def _change_layer(self):
        #freeze param
        for i in self.model1.parameters():
            i.requires_grad = False
        for i in self.model2.parameters():
            i.requires_grad = False
        for i in self.model3.parameters():
            i.requires_grad = False
        # for i in self.model4.parameters():
        #     i.requires_grad = False

        #load weight
        self.model1.load_state_dict(torch.load('E:\\Nicko\\TUGAS_AKHIR\\ClasifierWithEfficientNetV2S\\Saved_weigths\\model1_3.pt'))
        self.model2.load_state_dict(torch.load('E:\\Nicko\\TUGAS_AKHIR\\ClasifierWithEfficientNetV2S\\Saved_weigths\\model2_3.pt'))
        self.model3.load_state_dict(torch.load('E:\\Nicko\\TUGAS_AKHIR\\ClasifierWithEfficientNetV2S\\Saved_weigths\\model3_3.pt'))
        # self.model4.load_state_dict(torch.load('E:\\Nicko\\TUGAS_AKHIR\\ClasifierWithEfficientNetV2S\\Saved_weigths\\model4_2.pt'))


    def forward(self, x1, x2):
        prob1 = self.model1(x1[0], x2[0])
        prob2 = self.model2(x1[1], x2[1])
        prob3 = self.model3(x1[2], x2[2])
        # prob4 = self.model4(x1[3], x2[3])
        meta_input = torch.cat([prob1,prob2,prob3], dim=1)
        y_logits = self.meta_classifier(meta_input)
        y_pred = torch.round(torch.sigmoid(y_logits))
        return y_pred

    def training_step(self, batch, batch_idx):
        x1, x2, y = batch
        prob1 = self.model1(x1[0], x2[0])
        prob2 = self.model2(x1[1], x2[1])
        prob3 = self.model3(x1[2], x2[2])
        # prob4 = self.model4(x1[3], x2[3])
        meta_input = torch.cat([prob1,prob2,prob3], dim=1)
        y_logits = self.meta_classifier(meta_input)
        y_logits = y_logits.squeeze()
        loss = self.loss_fn(y_logits, y[0])
        acc = self.acc_fn(y_logits, y[0])
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        y_preds = torch.round(torch.sigmoid(y_logits.cpu())).detach().numpy()
        self.predictions = np.append(self.predictions, y_preds, axis=0)
        self.targets = np.append(self.targets, y[0].cpu(), axis=0)
        return {"loss": loss, "acc": acc}

    def validation_step(self, val_batch, batch_idx):
        x1, x2, y = val_batch
        prob1 = self.model1(x1[0], x2[0])
        prob2 = self.model2(x1[1], x2[1])
        prob3 = self.model3(x1[2], x2[2])
        # prob4 = self.model4(x1[3], x2[3])
        meta_input = torch.cat([prob1,prob2,prob3], dim=1)
        y_logits = self.meta_classifier(meta_input)
        y_logits = y_logits.squeeze()
        val_loss = self.loss_fn(y_logits, y[0])
        val_acc = self.acc_fn(y_logits, y[0])
        self.log("val_loss", val_loss)
        self.log("val_acc", val_acc)
        y_preds = torch.round(torch.sigmoid(y_logits.cpu())).detach().numpy()
        self.predictions_val = np.append(self.predictions_val, y_preds, axis=0)
        self.targets_val = np.append(self.targets_val, y[0].cpu(), axis=0)
        return {"val_loss": val_loss, "val_acc": val_acc}

    def test_step(self, test_batch, batch_idx):
        x1, x2, y = test_batch
        prob1 = self.model1(x1[0], x2[0])
        prob2 = self.model2(x1[1], x2[1])
        prob3 = self.model3(x1[2], x2[2])
        # prob4 = self.model4(x1[3], x2[3])
        meta_input = torch.cat([prob1,prob2,prob3], dim=1)
        y_logits = self.meta_classifier(meta_input)
        y_logits = y_logits.squeeze()
        test_loss = self.loss_fn(y_logits, y[0])
        test_acc = self.acc_fn(y_logits, y[0])
        self.log("test_loss", test_loss)
        self.log("test_acc", test_acc)
        y_preds = torch.round(torch.sigmoid(y_logits.cpu())).detach().numpy()
        conf_matrix = metrics.confusion_matrix(y[0].cpu(), y_preds)
        conf_matrix = np.flip(conf_matrix).T
        legend = ['Lie','Truth']
        legend2 = [['(TP)','(FP)'],['(FN)','(TN)']]
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
        ax.set_xticklabels([''] + legend)
        ax.set_yticklabels([''] + legend)
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                ax.text(x=j, y=i, s=(str(conf_matrix[i, j]) + ' ' + legend2[i][j]), va='center', ha='center', size='xx-large')

        plt.ylabel('Predictions', fontsize=20)
        plt.title('Actual', fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout(pad=1)
        y_preds = torch.round(torch.sigmoid(y_logits))
        print(y[0])
        f1score = self.f1_fn(y_preds, y[0])
        print('F1-score:', f1score)
        plt.show()
        return {"test_loss": test_loss, "test_acc": test_acc}

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
