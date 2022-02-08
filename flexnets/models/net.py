from typing import List
import torch.nn as nn
from flexnets.nn import GeneralizedLehmerConvolution as Conv2d # FIXME


class Net(nn.Module):
    def __init__(self, pool: List):
        super(Net, self).__init__()
        # FIXME if you call the same layer, the same underlying parameters
        # (weights and bias) will be used for the computation.
        # https://discuss.pytorch.org/t/calling-a-layer-multiple-times-will-produce-the-same-weights/28951
        self.pool1 = pool[0](**pool[1])
        self.pool2 = pool[0](**pool[1])
        self.pool3 = pool[0](**pool[1])

        self.block1 = nn.Sequential(
            Conv2d(in_channels=3, out_channels=32,
                      kernel_size=3, padding=1),

            nn.BatchNorm2d(32),
            nn.ReLU(),
            Conv2d(in_channels=32, out_channels=64,
                      kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            Conv2d(in_channels=64, out_channels=128,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            Conv2d(in_channels=128, out_channels=128,
                      kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.drop2 = nn.Dropout2d(p=0.05)
        self.block3 = nn.Sequential(
            Conv2d(in_channels=128, out_channels=256,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            Conv2d(in_channels=256, out_channels=256,
                      kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.pool1(self.block1(x))
        x = self.drop2(self.pool2(self.block2(x)))
        x = self.pool3(self.block3(x))
        x = x.reshape(x.shape[0], -1)
        x = self.fc_layer(x)
        return x
