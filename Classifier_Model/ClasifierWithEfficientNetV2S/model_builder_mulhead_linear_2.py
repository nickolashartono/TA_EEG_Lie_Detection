import torchinfo
import pytorch_lightning as pl
from torchmetrics.classification import BinaryF1Score, BinaryAccuracy
import torch
import numpy as np
from torchvision import models,ops
from torch import nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        # Define linear transformations for query, key, and value projections
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)

        # Define linear transformation for output projection
        self.output_linear = nn.Linear(d_model, d_model)

    def forward(self, query, key, value):
        batch_size = query.size(0)

        # Apply linear transformations to obtain query, key, and value projections
        Q = self.query_linear(query)
        K = self.key_linear(key)
        V = self.value_linear(value)

        # Split the projected tensors into multiple heads
        Q = Q.view(batch_size, -1, self.num_heads, self.d_model // self.num_heads).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_model // self.num_heads).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_model // self.num_heads).transpose(1, 2)

        # Compute the attention scores using the dot product of query and key
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.d_model / self.num_heads, dtype=torch.float32))

        # Apply softmax to obtain attention weights
        weights = F.softmax(scores, dim=-1)

        # Apply attention weights to the value projections
        attention = torch.matmul(weights, V)

        # Concatenate the multiple heads and apply the output projection
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1,
                                                                self.num_heads * (self.d_model // self.num_heads))
        output = self.output_linear(attention)

        return output


class LitClassifierWithEfficientNetV2S(pl.LightningModule):
    def __init__(self, unfreeze_layer=None, test_dataloader=None, learning_rate=1e-3, dropout=0.2):
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.dropout = dropout
        # self.mulhead_att = MultiHeadAttention(d_model=4080, num_heads=16)

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

        # backbone
        self.weights = models.EfficientNet_V2_S_Weights.DEFAULT
        self.backbone = models.efficientnet_v2_s(weights=self.weights)
        self.unfreeze_layer = unfreeze_layer
        self.test_dataloader = test_dataloader
        self._change_layer()

    def _change_layer(self):
        # UNFREEZE LAYER IF MEET NAME BELOW
        # FEATURE 1-7
        number = self.unfreeze_layer
        number = list(map(int, number)) if number else None
        for i, feature in enumerate(self.backbone.features):
            trainable = False
            if number:
                if i in number:
                    trainable = True
                feature.requires_grad_(trainable)
            else:
                feature.requires_grad_(False)

        # change input layer
        self.backbone.features[0] = ops.Conv2dNormActivation(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                                                             bias=False)

        # # change last layer
        self.backbone.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=self.dropout, inplace=True),
            torch.nn.Linear(in_features=1280, out_features=1, bias=True)
        )

    def forward(self, x1, x2):
        x_att_1 = self.mulhead_att(x1,x1,x1)
        x_att_2 = self.mulhead_att(x2,x2,x2)
        # x1 = torch.unsqueeze(x1, 1)
        # x2 = torch.unsqueeze(x2, 1)
        att_cat = torch.cat([x_att_1, x_att_2], dim=1)
        att_cat = torch.unsqueeze(att_cat, 1)
        y_logits = self.backbone(att_cat)
        y_pred = torch.round(torch.sigmoid(y_logits))
        return y_pred

    def training_step(self, batch, batch_idx):
        x1, x2, y = batch
        # x_att_1 = self.mulhead_att(x1,x1,x1)
        # x_att_2 = self.mulhead_att(x2,x2,x2)
        # x1 = torch.unsqueeze(x1, 1)
        # x2 = torch.unsqueeze(x2, 1)
        att_cat = torch.cat([x1, x2], dim=1)
        att_cat = torch.unsqueeze(att_cat, 1)
        y_logits = self.backbone(att_cat)
        y_logits = y_logits.squeeze()
        loss = self.loss_fn(y_logits, y)
        acc = self.acc_fn(y_logits, y)
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        y_preds = torch.round(torch.sigmoid(y_logits.cpu())).detach().numpy()
        self.predictions = np.append(self.predictions, y_preds, axis=0)
        self.targets = np.append(self.targets, y.cpu(), axis=0)
        return {"loss": loss, "acc": acc}

    def validation_step(self, val_batch, batch_idx):
        x1, x2, y = val_batch
        # x_att_1 = self.mulhead_att(x1,x1,x1)
        # x_att_2 = self.mulhead_att(x2,x2,x2)
        # x1 = torch.unsqueeze(x1, 1)
        # x2 = torch.unsqueeze(x2, 1)
        att_cat = torch.cat([x1, x2], dim=1)
        att_cat = torch.unsqueeze(att_cat, 1)
        y_logits = self.backbone(att_cat)
        y_logits = y_logits.squeeze()
        val_loss = self.loss_fn(y_logits, y)
        val_acc = self.acc_fn(y_logits, y)
        self.log("val_loss", val_loss)
        self.log("val_acc", val_acc)
        y_preds = torch.round(torch.sigmoid(y_logits.cpu())).detach().numpy()
        self.predictions_val = np.append(self.predictions_val, y_preds, axis=0)
        self.targets_val = np.append(self.targets_val, y.cpu(), axis=0)
        return {"val_loss": val_loss, "val_acc": val_acc}

    def test_step(self, test_batch, batch_idx):
        x1, x2, y = test_batch
        # x_att_1 = self.mulhead_att(x1,x1,x1)
        # x_att_2 = self.mulhead_att(x2,x2,x2)
        # x1 = torch.unsqueeze(x1, 1)
        # x2 = torch.unsqueeze(x2, 1)
        att_cat = torch.cat([x1, x2], dim=1)
        att_cat = torch.unsqueeze(att_cat, 1)
        y_logits = self.backbone(att_cat)
        y_logits = y_logits.squeeze()
        test_loss = self.loss_fn(y_logits, y)
        test_acc = self.acc_fn(y_logits, y)
        self.log("test_loss", test_loss)
        self.log("test_acc", test_acc)
        return {"test_loss": test_loss, "test_acc": test_acc}

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
