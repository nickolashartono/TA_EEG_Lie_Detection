import torchinfo
import pytorch_lightning as pl
from torchmetrics.classification import BinaryF1Score, BinaryAccuracy
import torch
import numpy as np
from torchvision import models,ops
from torch import nn
import torch.nn.functional as F


class MultiheadAttention(nn.Module):
    def __init__(self, in_channels, head_channels, num_heads):
        super(MultiheadAttention, self).__init__()
        self.in_channels = in_channels
        self.head_channels = head_channels
        self.num_heads = num_heads

        # Define the query, key, and value convolutional layers for each head
        self.query_convs = nn.ModuleList(
            [nn.Conv2d(in_channels, head_channels, kernel_size=1, bias=False) for i in range(num_heads)])
        self.key_convs = nn.ModuleList(
            [nn.Conv2d(in_channels, head_channels, kernel_size=1, bias=False) for i in range(num_heads)])
        self.value_convs = nn.ModuleList(
            [nn.Conv2d(in_channels, head_channels, kernel_size=1, bias=False) for i in range(num_heads)])

        # Define the output projection convolutional layer
        self.output_conv = nn.Conv2d(head_channels * num_heads, in_channels, kernel_size=1, bias=False)

        # Define the softmax layer
        self.softmax = nn.Softmax(dim=-1)

        # Define the scaling factor for dot product attention
        self.scale_factor = 1 / (head_channels ** 0.5)

    def forward(self, x):
        batch_size, in_channels, height, width = x.size()

        # Compute query, key, and value tensors for each head by passing input through convolutional layers
        queries = [self.query_convs[i](x) for i in range(self.num_heads)]
        keys = [self.key_convs[i](x) for i in range(self.num_heads)]
        values = [self.value_convs[i](x) for i in range(self.num_heads)]

        # Concatenate the query, key, and value tensors for each head along the batch dimension
        queries = torch.cat(queries, dim=0)
        keys = torch.cat(keys, dim=0)
        values = torch.cat(values, dim=0)

        # Transpose and reshape the query, key, and value tensors for batch matrix multiplication
        queries = queries.view(self.num_heads, batch_size, self.head_channels, height * width)
        queries = queries.transpose(2, 3)
        keys = keys.view(self.num_heads, batch_size, self.head_channels, height * width)
        keys = keys.transpose(2, 3)
        values = values.view(self.num_heads, batch_size, self.head_channels, height * width)
        values = values.transpose(2, 3)

        # Compute the attention scores using batch matrix multiplication
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale_factor
        attention_probs = self.softmax(attention_scores)

        # Apply attention to the value tensor
        weighted_sum = torch.matmul(attention_probs, values)

        # Transpose and reshape the weighted sum tensor
        weighted_sum = weighted_sum.transpose(1, 2).reshape(batch_size, self.head_channels*self.num_heads, height,
                                                            width)

        # Apply the output projection convolutional layer
        output = self.output_conv(weighted_sum)

        return output


class LitClassifierWithEfficientNetV2S(pl.LightningModule):
    def __init__(self, unfreeze_layer=None, test_dataloader=None, learning_rate=1e-3, dropout=0.2):
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.dropout = dropout
        self.mulhead_att = MultiheadAttention(in_channels=32, num_heads=8, head_channels=4)

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
        self.backbone.features[0] = ops.Conv2dNormActivation(64, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                                                             bias=False)

        # # change last layer
        self.backbone.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=self.dropout, inplace=True),
            torch.nn.Linear(in_features=1280, out_features=640, bias=True),
            torch.nn.ReLU(True),
            torch.nn.Linear(in_features=640, out_features=1, bias=True)
        )

    def forward(self, x1, x2):
        x_att_1 = self.mulhead_att(x1)
        x_att_2 = self.mulhead_att(x2)
        att_cat = torch.cat([x_att_1, x_att_2], dim=1)
        y_logits = self.backbone(att_cat)
        y_pred = torch.round(torch.sigmoid(y_logits))
        return y_pred

    def training_step(self, batch, batch_idx):
        x1, x2, y = batch
        x_att_1 = self.mulhead_att(x1)
        x_att_2 = self.mulhead_att(x2)
        att_cat = torch.cat([x_att_1, x_att_2], dim=1)
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
        x_att_1 = self.mulhead_att(x1)
        x_att_2 = self.mulhead_att(x2)
        att_cat = torch.cat([x_att_1, x_att_2], dim=1)
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
        x_att_1 = self.mulhead_att(x1)
        x_att_2 = self.mulhead_att(x2)
        att_cat = torch.cat([x_att_1, x_att_2], dim=1)
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
