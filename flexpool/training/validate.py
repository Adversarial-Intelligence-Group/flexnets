from typing import Callable
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
from flexpool.nn.pooling import GeneralizedLehmerPool2d, GeneralizedPowerMeanPool2d

from .utils import accuracy, clip_data


def validate(epoch: int,
             model: nn.Module,
             val_loader: DataLoader,
             loss_func: Callable[[Tensor, Tensor], Tensor],
             writer: SummaryWriter = None):
    model.eval()

    loss_sum, accs_sum = 0, 0

    for idx, (images, targets) in enumerate(tqdm(val_loader)):
        if next(model.parameters()).is_cuda:
            images, targets = images.cuda(), targets.cuda()

        with torch.no_grad():
            if writer is not None and (epoch == 0 and idx == 0):
                writer.add_graph(model, images)

            outputs = model(images)
            acc = accuracy(outputs, targets)
            loss = loss_func(outputs, targets)
            loss_sum += loss.item()
            if isinstance(acc, torch.Tensor):
                accs_sum += acc.item()
            else:
                accs_sum += acc

        global_step = epoch * len(val_loader) + idx

        if writer is not None:
            writer.add_scalar(
                'val/iter_loss', loss.item(), global_step=global_step)

    if writer is not None:
        loss_avg = loss_sum / len(val_loader)
        accs_avg = accs_sum / len(val_loader)

        print(f'Loss = {loss_avg:.4e}, Accuracy = {accs_avg:.4e}')

        writer.add_scalar('val/loss', loss_avg, global_step=epoch)
        writer.add_scalar('val/accuracy', accs_avg, global_step=epoch)

    # FIXME
    if (type(model.pool1) is GeneralizedLehmerPool2d or
            type(model.pool1) is GeneralizedPowerMeanPool2d):
        alpha_pools = ["pool1.alpha", "pool2.alpha", "pool3.alpha"]
        beta_pools = ["pool1.beta", "pool2.beta", "pool3.beta"]
        for name, p in model.named_parameters():
            if name in alpha_pools:
                p.data = clip_data(p, 1.00001, 2.71828)
            if name in beta_pools:
                p.data = clip_data(p, -2.5, 1.5)

    return loss_sum, accs_sum
