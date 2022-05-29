from typing import List, Union
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from flexnets.nn.layers import GeneralizedLehmerLayer
from flexnets.nn.pooling import GeneralizedLehmerPool2d, GeneralizedPowerMeanPool2d
from torch.utils.tensorboard.writer import SummaryWriter


def accuracy(output: torch.Tensor, target: torch.Tensor, top_k=(1,)) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Computes the precision@k for the specified values of k"""
    max_k = max(top_k)
    batch_size = target.size(0)

    _, pred = output.topk(max_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in top_k:
        correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    if len(res) == 1:
        res = res[0]

    return res


def plot_poolings(model: nn.Module, writer: SummaryWriter, tag: str, global_step: int):
    module_types = {key: type(module) for key, module in model.named_modules()}
    for name, p in model.named_parameters():
        # if (module_types[name.split('.')[0]] is GeneralizedLehmerPool2d):
        if (module_types['.'.join(name.split('.')[:-1])] is GeneralizedLehmerPool2d):
            if ("alpha" in name) or ("beta" in name):
                writer.add_scalar(tag+'/'+name, p.data.item(),
                                    global_step=global_step)

        # if (module_types[name.split('.')[0]] is GeneralizedPowerMeanPool2d):
        if (module_types['.'.join(name.split('.')[:-1])] is GeneralizedPowerMeanPool2d):
            if ("gamma" in name) or ("delta" in name):
                writer.add_scalar(tag+'/'+name, p.data.item(),
                                    global_step=global_step)


def freeze_poolings(model: nn.Module):
    module_types = {key: type(module) for key, module in model.named_modules()}
    for name, p in model.named_parameters():
        if (module_types[name.split('.')[0]] is GeneralizedLehmerPool2d):
            if ("alpha" in name) or ("beta" in name):
                p.requires_grad = False

        if (module_types[name.split('.')[0]] is GeneralizedPowerMeanPool2d):
            if "gamma" in name or "delta" in name:
                p.requires_grad = False


def clip_poolings(model: nn.Module):
    module_types = {key: type(module) for key, module in model.named_modules()}
    for name, p in model.named_parameters():
        if (module_types[name.split('.')[0]] is GeneralizedLehmerPool2d or module_types[name.split('.')[0]] is GeneralizedLehmerLayer):
            if "alpha" in name:
                p.data = clip_data(p, 1.00001, 2.71828)
            if "beta" in name:
                p.data = clip_data(p, -2.5, 1.5)

        # if (module_types[name.split('.')[0]] is GeneralizedPowerMeanPool2d):
        if (module_types['.'.join(name.split('.')[:-1])] is GeneralizedPowerMeanPool2d):
            if "gamma" in name:
                p.data = clip_data(p, 1.00001, 2.71828)
            if "delta" in name:
                p.data = clip_data(p, -2.5, 1.5)


def clip_data(parameters: torch.Tensor, min_clip_value: float, max_clip_value: float) -> torch.Tensor:
    device = parameters.data.device

    min_clip_value = float(min_clip_value)
    max_clip_value = float(max_clip_value)

    if float(parameters.item()) > max_clip_value or float(parameters.item()) < min_clip_value:
        print(parameters.item())  # TODO del
        return parameters.data.clamp(
            min=min_clip_value, max=max_clip_value).to(device)
    return parameters


def save_checkpoint(path: str,
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    scheduler: _LRScheduler):
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }, path)


def load_checkpoint(path: str,
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    scheduler: _LRScheduler,
                    cuda: bool = True):
    state = torch.load(path)
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])

    if cuda:
        print('Moving model to cuda')
        model = model.cuda()

    return model
