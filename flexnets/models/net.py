from typing import List
import torch.nn as nn
from flexnets.nn.pooling import GeneralizedLehmerPool2d, GeneralizedPowerMeanPool2d, LPPool2d
from flexnets.nn.convolution import GeneralizedLehmerConvolution, GeneralizedPowerConvolution


cfg = {
    'NN': [32, 'BN', 64, 'M', 128, 'BN', 128, 'M', 'D', 256, 'BN', 256, 'M'],
}


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()

        self.features = make_layers(cfg['NN'], args)
        self.fc_layer = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Dropout2d(p=0.15),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout2d(p=0.15),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc_layer(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.modules.conv._ConvNd):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, args):
    layers = []
    in_channels = 3
    batch_norm = False
    for out in cfg:
        if out == 'M':
            get_pooling(args, layers)
        elif out == 'D':
            layers += [nn.Dropout2d(p=0.15)]
        elif out == 'BN':
            batch_norm = True
        else:
            conv2d = get_conv(args, in_channels, out)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(out), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = out
            batch_norm = False
    return nn.Sequential(*layers)


def get_conv(args, in_channels, out):
    if args.conv_type == 'generalized_lehmer_conv':
        return GeneralizedLehmerConvolution(in_channels, out,
                                            kernel_size=3, padding=1,
                                            alpha=float(args.alpha), beta=float(args.beta))
    elif args.conv_type == 'generalized_power_conv':
        return GeneralizedPowerConvolution(in_channels, out,
                                           kernel_size=3, padding=1,
                                           gamma=float(args.gamma), delta=float(args.delta))
    else:
        return nn.Conv2d(in_channels, out, kernel_size=3, padding=1)


def get_pooling(args, layers):
    if args.pooling_type == 'generalized_lehmer_pool':
        layers += [GeneralizedLehmerPool2d(float(args.alpha), float(args.beta),
                                           kernel_size=2, stride=2)]
    elif args.pooling_type == 'generalized_power_mean_pool':
        layers += [GeneralizedPowerMeanPool2d(float(args.gamma), float(args.delta),
                                              kernel_size=2, stride=2)]
    elif args.pooling_type == 'lp_pool':
        layers += [LPPool2d(float(args.norm_type), kernel_size=2, stride=2)]
    else:
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
