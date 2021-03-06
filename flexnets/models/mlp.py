import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics

from flexnets.nn.activation import GeneralizedLehmerSoftMax


class MLP(pl.LightningModule):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.accuracy = torchmetrics.Accuracy()
        self.layers = nn.Sequential(
            nn.Linear(in_features, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, out_features)
        )
        self.ce = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.layers(x)
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
        
        return { 'loss': loss, 'accuracy': acc }
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self.layers(x)
        loss = self.ce(y_hat, y)
        self.log('val_loss', float("{:.3f}".format(loss)))

        acc = self.accuracy(y_hat, y)
        self.log('val_accuracy', float("{:.2f}".format(acc * 100)), on_step=False, on_epoch=True)

        return { 'loss': loss, 'accuracy': acc }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer