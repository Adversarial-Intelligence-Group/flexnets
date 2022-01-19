import argparse
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.modules.conv import Conv2d
from torch.utils.data import DataLoader
from torch.nn.modules.module import Module
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from data.data import get_datasets
from nn.pooling import GeneralizedLehmerPool2d
from models import Net


from nn.activation import GeneralizedSoftPlus

matplotlib.rcParams.update({'font.size': 5})


def get_grads(act_fn: Module, x: Tensor):
    x = x.clone().requires_grad_()
    out = act_fn(x)
    out.sum().backward()
    return x.grad


def vis_act_fn(act_fn, ax, x):
    # Run activation function
    y = act_fn(x)
    y_grads = get_grads(act_fn, x)
    # Push x, y and gradients back to cpu for plotting
    x, y, y_grads = x.cpu().detach().numpy(), y.cpu(
    ).detach().numpy(), y_grads.cpu().numpy()
    # Plotting
    ax.plot(x, y, linewidth=2, label="ActFn")
    ax.plot(x, y_grads, linewidth=2, label="Gradient")
    ax.set_title(act_fn.__class__.__name__)
    ax.legend()
    ax.set_ylim(-1.5, x.max())


def plot_act_fns():
    act_fn_by_name = {"generalizedsoftplus": GeneralizedSoftPlus,
                      "relu": nn.ReLU, "sigmoid": nn.Sigmoid, "softplus": nn.Softplus}
    act_fns = [act_fn() for act_fn in act_fn_by_name.values()]
    # Range on which we want to visualize the activation functions
    x = torch.linspace(-5, 5, 1000)
    # Plotting
    cols = 2
    rows = math.ceil(len(act_fns) / float(cols))
    fig, ax = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    for i, act_fn in enumerate(act_fns):
        vis_act_fn(act_fn, ax[divmod(i, cols)], x)
    fig.subplots_adjust(hspace=0.3)
    plt.show()


def visualize_gradients(net, device, train_set, color="C0"):
    net.eval()
    small_loader = DataLoader(train_set, batch_size=256, shuffle=False)
    imgs, labels = next(iter(small_loader))
    imgs, labels = imgs.to(device), labels.to(device)

    # Pass one batch through the network, and calculate the gradients for the weights
    net.zero_grad()
    preds = net(imgs)
    loss = F.cross_entropy(preds, labels)
    loss.backward()
    # We limit our visualization to the weight parameters and exclude the bias to reduce the number of plots
    grads = {
        name: params.grad.data.view(-1).cpu().clone().numpy()
        for name, params in net.named_parameters()
        if "weight" in name
    }
    net.zero_grad()

    # Plotting
    columns = len(grads)
    fig, ax = plt.subplots(1, columns, figsize=(columns * 3.5, 2.5))
    fig_index = 0
    for key in grads:
        key_ax = ax[fig_index % columns]
        sns.histplot(data=grads[key], bins=30,
                     ax=key_ax, color=color, kde=True)
        key_ax.set_title(str(key))
        key_ax.set_xlabel("Grad magnitude")
        fig_index += 1
    fig.suptitle(
        f"Gradient magnitude distribution for activation function", fontsize=6
    )
    fig.subplots_adjust(wspace=0.2)
    plt.show()
    plt.close()


def visualize_activations(net, device, train_set, color="C0"):
    activations = {}

    net.eval()
    small_loader = DataLoader(train_set, batch_size=32)
    imgs, labels = next(iter(small_loader))
    with torch.no_grad():
        layer_index = 0
        imgs = imgs.to(device)
        # We need to manually loop through the layers to save all activations
        for layer_index, layer in enumerate(list(net.modules())[:-1]):
            # if isinstance(layer, nn.ReLU):
            #     imgs = imgs.view(imgs.size(0), -1)
            #     imgs = layer(imgs)
            #     activations[layer_index] = imgs.view(-1).cpu().numpy()

            if isinstance(layer, nn.Conv2d):
                imgs = layer(imgs)
                activations[layer_index] = imgs.view(-1).cpu().numpy()

    # Plotting
    columns = 4
    rows = math.ceil(len(activations) / columns)
    fig, ax = plt.subplots(rows, columns, figsize=(columns * 2.7, rows * 2.5))
    fig_index = 0
    for key in activations:
        key_ax = ax[fig_index // columns][fig_index % columns]
        sns.histplot(data=activations[key], bins=50,
                     ax=key_ax, color=color, kde=True, stat="density")
        key_ax.set_title(
            f"Layer {key} - {list(net.modules())[key].__class__.__name__}")
        fig_index += 1
    fig.suptitle(
        f"Activation distribution for activation function", fontsize=6
    )
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.show()
    plt.close()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


set_seed(42)
pool = GeneralizedLehmerPool2d(2.7, 1.5, kernel_size=2, stride=2)
# pool = nn.MaxPool2d(kernel_size=2, stride=2)
net = Net(pool)


args = argparse.Namespace()
args.data_path = '../.assets/data'
args.normalize = False

train, test = get_datasets(args)
visualize_gradients(net, 'cpu', test, color=f"C1")
