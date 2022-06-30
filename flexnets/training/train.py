import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchmetrics import SumMetric
from torchmetrics.functional import accuracy

from typing import Callable, Union
from tqdm import tqdm

from .utils import plot_poolings


def train(
        model: nn.Module,
        device: torch.device,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Union[_LRScheduler, None],
        loss_func: Callable,
        epoch: int,
        writer: Union[SummaryWriter, None] = None):
    
    model.train()

    loss_metric = SumMetric()
    accuracy_metric = SumMetric()

    for idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        outputs = model(data)
        
        loss = loss_func(outputs, target)

        loss.backward()
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()

        # TODO: do we need no_grad?
        with torch.no_grad():
            acc = accuracy(outputs, target, top_k=1)

        loss_metric.update(loss)
        accuracy_metric.update(acc)

        global_step = epoch * len(train_loader) + idx

        if writer is not None:
            writer.add_scalar('train/iter_loss', loss.item(),
                              global_step=global_step)
            plot_poolings(model, writer, 'train_pool', global_step)

    loss_sum = loss_metric.compute()
    accuracy_sum = accuracy_metric.compute()

    if writer is not None:
        loss_avg = loss_sum / len(train_loader)
        accs_avg = accuracy_metric.compute() / len(train_loader)

        print(f'Loss = {loss_avg:.4e}, Accuracy = {accs_avg:.4e}')

        writer.add_scalar('train/loss', loss_avg, global_step=epoch)
        writer.add_scalar('train/accuracy', accs_avg, global_step=epoch)
        if scheduler is not None:
            writer.add_scalar('train/lr', scheduler.get_last_lr()[-1], global_step=epoch)

    return loss_sum, accuracy_sum
