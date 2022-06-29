import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from flexnets.nn.activation import GeneralizedLehmerSoftMax, GeneralizedPowerSoftMax

from flexnets.nn.layers import GeneralizedLehmerLayer
from flexnets.training.utils import clip_poolings


class GeneralizedLehmerNetwork(pl.LightningModule):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.accuracy = torchmetrics.Accuracy()
        self.layers = nn.Sequential(
            GeneralizedLehmerLayer(in_features, 50),
            nn.Sigmoid(),
            GeneralizedLehmerLayer(50, 50),
            nn.Sigmoid(),
            GeneralizedLehmerLayer(50, out_features)
        )
        # self.ce = nn.CrossEntropyLoss()
        self.ce = nn.NLLLoss()
        self.activation_fn = GeneralizedPowerSoftMax(2, 2, 0.5, 0.5)

    def forward(self, x):
        x = self.layers(x)
        x = self.activation_fn(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self.layers(x)
        loss = self.ce(y_hat, y)
        self.log('train_loss', float("{:.3f}".format(loss)))

        acc = self.accuracy(y_hat, y)
        self.log('train_accuracy', float("{:.2f}".format(acc * 100)), on_step=False, on_epoch=True)

        for name, p in self.layers.named_parameters():
            if ("alpha" in name) or ("beta" in name):
                self.log(name, p.data)
        
        # clip_poolings(self.layers)
        return { 'loss': loss, 'accuracy': acc }
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self.layers(x)
        loss = self.ce(y_hat, y)
        self.log('val_loss', float("{:.3f}".format(loss)))

        acc = self.accuracy(y_hat, y)
        self.log('val_accuracy', float("{:.2f}".format(acc * 100)), on_step=False, on_epoch=True)

        # clip_poolings(self.layers)
        return { 'loss': loss, 'accuracy': acc }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer