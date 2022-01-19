import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics

from flexnets.nn.gln import GeneralizedLehmerLayer
from flexnets.training.utils import clip_poolings

class MLP(pl.LightningModule):
    def __init__(self, in_features):
        super().__init__()
        self.accuracy = torchmetrics.Accuracy()
        self.layers = nn.Sequential(
            # nn.Linear(in_features, 100),
            # nn.Linear(100, 100),
            # nn.Linear(100, 3)
            GeneralizedLehmerLayer(in_features, 100),
            nn.ReLU(),
            GeneralizedLehmerLayer(100, 100),
            nn.ReLU(),
            GeneralizedLehmerLayer(100, 3)
        )
        # self.softmax = nn.Softmax(dim=1)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.layers(x)
        # return self.softmax(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self.layers(x)
        loss = self.ce(y_hat, y)
        self.log('train_loss', loss)

        acc = self.accuracy(y_hat, y)
        self.log('train_accuracy', acc, on_step=True, on_epoch=False)
        
        return { 'loss': loss, 'accuracy': acc }
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self.layers(x)
        loss = self.ce(y_hat, y)
        self.log('val_loss', loss)

        acc = self.accuracy(y_hat, y)
        self.log('val_accuracy', acc, on_step=True, on_epoch=False)

        return { 'loss': loss, 'accuracy': acc }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer