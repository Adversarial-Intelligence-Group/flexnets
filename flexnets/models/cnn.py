import torch.nn as nn
from flexnets.nn.activation import GeneralizedLehmerSoftMax, GeneralizedPowerSoftMax
from flexnets.nn.pooling import GeneralizedLehmerPool2d, GeneralizedPowerMeanPool2d, LPPool2d
from flexnets.nn.convolution import GeneralizedLehmerConvolution, GeneralizedPowerConvolution
from flexnets.nn.activation import GLSoftMax, GPSoftMax, GReLU
from flexnets.nn.layers import GeneralizedLehmerLayer
import torch

cfg = {
    'NN_mid': [32, 'BN', 64, 'M'],
    'NN_last': [32, 'BN', 64, 'M', 128, 'BN', 128, 'M', 'D'],
    'NN_first': [128, 'BN', 128, 'M', 'D', 256, 'BN', 256, 'M'],
    'NN': [32, 'BN', 64, 'M', 128, 'BN', 128, 'M', 'D', 256, 'BN', 256, 'M'],
}


class Net_MidBlock(nn.Module): #
    def __init__(self, args):
        super(Net, self).__init__()

        self.features = make_layers(cfg['NN_mid'], args)
        self.last_block = nn.Sequential(
            # GeneralizedLehmerConvolution(64, 128, 3, 1, 1, alpha=1.8, beta=1.3),
            GeneralizedPowerConvolution(64, 128,  3, 1, 1, delta=1.5, gamma=-1.3),
            nn.ReLU(),
            # GeneralizedLehmerConvolution(128, 128, 3, 1, 1, alpha=1.8, beta=1.3),
            GeneralizedPowerConvolution(128, 128, 3, 1, 1, delta=1.5, gamma=-1.3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # GeneralizedLehmerPool2d(1.8, 1.3, 2, 2),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Dropout2d(p=0.15),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Dropout2d(p=0.15),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout2d(p=0.15),
            nn.Linear(512, 10),
            # nn.LogSoftmax()
            # GPSoftMax()
        )
        # self.fc_layer = nn.Sequential(
        #     GeneralizedLehmerLayer(4096, 1024),
        #     nn.ReLU(),
        #     # nn.Sigmoid(),
        #     nn.Dropout2d(p=0.15),
        #     GeneralizedLehmerLayer(1024, 512),
        #     nn.ReLU(),
        #     # nn.Sigmoid(),
        #     nn.Dropout2d(p=0.15),
        #     GeneralizedLehmerLayer(512, 10),
        #     # nn.LogSoftmax()
        #     # GPSoftMax()
        # )

    def forward(self, x):
        x = self.features(x)
        x = self.last_block(x)
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


class Net_GLN(nn.Module): #
    def __init__(self, args):
        super(Net, self).__init__()

        self.features = make_layers(cfg['NN'], args)
        # self.fc_layer = nn.Sequential(
        #     nn.Linear(4096, 1024),
        #     nn.ReLU(),
        #     nn.Dropout2d(p=0.15),
        #     nn.Linear(1024, 512),
        #     nn.ReLU(),
        #     nn.Dropout2d(p=0.15),
        #     nn.Linear(512, 10),
        #     # nn.LogSoftmax()
        #     # GPSoftMax()
        # )
        # path = '.assets_conv/checkpoints/ex_1_c_220531-145946942239_max_pool2d/best_model.pth'
        # state = torch.load(path)
        # self.load_state_dict(state)
        # for name, p in self.features.named_parameters():
        #     p.requires_grad = False

        self.fc_layer = nn.Sequential(
            GeneralizedLehmerLayer(4096, 1024),
            nn.ReLU(),
            # nn.Sigmoid(),
            nn.Dropout2d(p=0.15),
            GeneralizedLehmerLayer(1024, 512),
            nn.ReLU(),
            # nn.Sigmoid(),
            nn.Dropout2d(p=0.15),
            GeneralizedLehmerLayer(512, 10),
            # nn.LogSoftmax()
            # GPSoftMax()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc_layer(x)

        if (self.activation_fn):
            return self.activation_fn(x)
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

class Net0(nn.Module):
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
            nn.Linear(512, 10),
            # nn.LogSoftmax()
            # GPSoftMax()
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

class Net_FirstBlock(nn.Module): #
    def __init__(self, args):
        super(Net, self).__init__()

        self.features = make_layers(cfg['NN_first'], args)
        self.first_block = nn.Sequential(
            # GeneralizedLehmerConvolution(3, 32, 3, 1, 1, alpha=2, beta=0.5),
            GeneralizedPowerConvolution(3, 32, 3, 1, 1, delta=2., gamma=0.5),
            # GReLU(1.3, 1.2),
            nn.ReLU(),
            # GeneralizedLehmerConvolution(32, 64, 3, 1, 1, alpha=1.8, beta=0.),
            GeneralizedPowerConvolution(32, 64, 3, 1, 1, delta=2., gamma=0.5),
            # nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            # GReLU(1.3, 1.2),
            nn.ReLU(),
            # GeneralizedLehmerPool2d(1.5, 1.3, 2, 2),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Dropout2d(p=0.15),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout2d(p=0.15),
            nn.Linear(512, 10),
            # nn.LogSoftmax()
            # GPSoftMax()
        )

    def forward(self, x):
        x = self.first_block(x)
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

class Net(nn.Module): #_LastBlock
    def __init__(self, args):
        super(Net, self).__init__()

        self.features = make_layers(cfg['NN_last'], args)
        self.last_block = nn.Sequential(
            GeneralizedLehmerConvolution(128, 256, 3, 1, 1, alpha=1.8, beta=1.3),
            # GeneralizedPowerConvolution(128, 256, 3, 1, 1, delta=1.5, gamma=1.3),
            nn.ReLU(),
            GeneralizedLehmerConvolution(256, 256, 3, 1, 1, alpha=1.8, beta=1.3),
            # GeneralizedPowerConvolution(256, 256, 3, 1, 1, delta=1.5, gamma=1.3),

            # GeneralizedPowerConvolution(256, 256, 3, 1, 1, delta=2.3, gamma=-2.),
            nn.BatchNorm2d(256),
            # GReLU(1.3, 1.2),
            nn.ReLU(),
            GeneralizedLehmerPool2d(1.8, 1.3, 2, 2),
            # nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # self.fc_layer = nn.Sequential(
        #     nn.Linear(4096, 1024),
        #     nn.ReLU(),
        #     nn.Dropout2d(p=0.15),
        #     nn.Linear(1024, 512),
        #     nn.ReLU(),
        #     nn.Dropout2d(p=0.15),
        #     nn.Linear(512, 10),
        #     # nn.LogSoftmax()
        #     # GPSoftMax()
        # )
        self.fc_layer = nn.Sequential(
            GeneralizedLehmerLayer(4096, 1024),
            nn.ReLU(),
            # nn.Sigmoid(),
            nn.Dropout2d(p=0.15),
            GeneralizedLehmerLayer(1024, 512),
            nn.ReLU(),
            # nn.Sigmoid(),
            # nn.Dropout2d(p=0.15),
            GeneralizedLehmerLayer(512, 10),
            # nn.LogSoftmax()
            # GPSoftMax()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.last_block(x)
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
                layers += [conv2d, nn.BatchNorm2d(out), nn.ReLU(inplace=True)]#GReLU()] # ## #
            else:
                layers += [conv2d, nn.ReLU(inplace=True)] #GReLU()] # # # #
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


def get_activation_fn(args):
    if (args.activation_fn_type == 'generalized_lehmer_softmax'):
        return GeneralizedLehmerSoftMax(float(args.alpha_1), float(args.alpha_2),
                                        float(args.beta_1), float(args.beta_2))
    elif (args.activation_fn_type == 'generalized_power_softmax'):
        return GeneralizedPowerSoftMax(float(args.gamma_1), float(args.gamma_2),
                                       float(args.delta_1), float(args.delta_2))
    else:
        return None
