import torchinfo
import pytorch_lightning as pl
from torchmetrics.classification import BinaryF1Score, BinaryAccuracy
import torch
import numpy as np
from torchvision import models,ops
from torch import nn
import torch.nn.functional as F

class MultiheadAttention(nn.Module):
    def __init__(self, in_channels, num_heads, head_channels, kernel_size=1, padding=0, stride=1, dropout=0.1):
        super(MultiheadAttention, self).__init__()

        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_channels = head_channels

        # Convolutional layers for query, key, and value projections for each head
        self.query_projections = nn.ModuleList([nn.Conv2d(in_channels, head_channels, kernel_size=kernel_size, padding=padding, stride=stride) for _ in range(num_heads)])
        self.key_projections = nn.ModuleList([nn.Conv2d(in_channels, head_channels, kernel_size=kernel_size, padding=padding, stride=stride) for _ in range(num_heads)])
        self.value_projections = nn.ModuleList([nn.Conv2d(in_channels, head_channels, kernel_size=kernel_size, padding=padding, stride=stride) for _ in range(num_heads)])

        # Convolutional layer for the output projection
        self.output_projection = nn.Conv2d(num_heads * head_channels, in_channels, kernel_size=kernel_size, padding=padding, stride=stride)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_list):
        # x_list is a list of tensors with shape (batch_size, in_channels, height, width)

        batch_size, in_channels, height, width = x_list[0].size()

        # Concatenate tensors along the batch dimension
        x = torch.cat(x_list, dim=0)

        # Compute the query, key, and value projections for each head
        queries = [self.query_projections[i](x) for i in range(self.num_heads)]
        keys = [self.key_projections[i](x) for i in range(self.num_heads)]
        values = [self.value_projections[i](x) for i in range(self.num_heads)]

        # Split the queries, keys, and values back into batches
        queries = [q.view(batch_size, self.head_channels, -1) for q in queries]
        keys = [k.view(batch_size, self.head_channels, -1) for k in keys]
        values = [v.view(batch_size, self.head_channels, -1) for v in values]

        # Compute the dot product attention scores
        attention_scores = [torch.bmm(queries[i].transpose(1, 2), keys[i]) for i in range(self.num_heads)]

        # Scale the attention scores by the square root of the head dimension
        attention_scores = [score / (self.head_channels ** 0.5) for score in attention_scores]


        # Apply the softmax function to obtain the attention weights
        attention_weights = [F.softmax(score, dim=2) for score in attention_scores]

        # Apply dropout to the attention weights
        attention_weights = [self.dropout(weight) for weight in attention_weights]

        # Compute the weighted sum of the values using the attention weights
        weighted_sum = [torch.bmm(values[i], attention_weights[i].transpose(1, 2)) for i in range(self.num_heads)]

        # Concatenate the weighted sums along the head dimension
        weighted_sum = torch.cat(weighted_sum, dim=1)

        # Reshape the weighted sum tensor to (batch_size * num_heads, head_channels, height, width)
        weighted_sum = weighted_sum.view(batch_size * self.num_heads, self.head_channels * self.num_heads, height, width)

        # Apply the output projection
        output = self.output_projection(weighted_sum)
        output_reshape = output.view(batch_size, self.head_channels*self.num_heads, height, width)

        return output_reshape


class LitClassifierWithEfficientNetV2S(pl.LightningModule):
    def __init__(self, unfreeze_layer=None, test_dataloader=None, learning_rate=1e-3, dropout=0.2):
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.dropout = dropout
        self.mulhead_att = MultiheadAttention(in_channels=64, num_heads=2, head_channels=64)

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
        self.backbone.features[0] = ops.Conv2dNormActivation(128, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                                                             bias=False)

        # # change last layer
        self.backbone.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=self.dropout, inplace=True),
            torch.nn.Linear(in_features=1280, out_features=1, bias=True)
        )

    def forward(self, x1, x2):
        x_combined = [x1,x2]
        x_combined_att = self.mulhead_att(x_combined)
        y_logits = self.backbone(x_combined_att)
        y_pred = torch.round(torch.sigmoid(y_logits))
        return y_pred

    def training_step(self, batch, batch_idx):
        x1, x2, y = batch
        x_combined = [x1,x2]
        x_combined_att = self.mulhead_att(x_combined)
        y_logits = self.backbone(x_combined_att)
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
        x_combined = [x1,x2]
        x_combined_att = self.mulhead_att(x_combined)
        y_logits = self.backbone(x_combined_att)
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
        x_combined = [x1,x2]
        x_combined_att = self.mulhead_att(x_combined)
        y_logits = self.backbone(x_combined_att)
        y_logits = y_logits.squeeze()
        test_loss = self.loss_fn(y_logits, y)
        test_acc = self.acc_fn(y_logits, y)
        self.log("test_loss", test_loss)
        self.log("test_acc", test_acc)
        return {"test_loss": test_loss, "test_acc": test_acc}

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        return optimizer
