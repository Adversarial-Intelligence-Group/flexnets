from typing import Callable
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from training.utils import accuracy


def train(
        epoch: int,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: _LRScheduler,
        loss_func: Callable,
        writer: SummaryWriter = None):
    model.train()
    
    loss_sum, accs_sum = 0, 0

    for idx, (images, targets) in enumerate(tqdm(train_loader)):
        if next(model.parameters()).is_cuda:
            images, targets = images.cuda(), targets.cuda()

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_func(outputs, targets)
        
        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            acc = accuracy(outputs, targets)

        loss_sum += loss.item()
        if isinstance(acc, torch.Tensor):
            accs_sum += acc.item()

        global_step = epoch * len(train_loader) + idx

        if writer is not None:
            writer.add_scalar('train/iter_loss', loss.item(),
                            global_step=global_step)

    if writer is not None:
        loss_avg = loss_sum / len(train_loader)
        accs_avg = accs_sum / len(train_loader)

        print(f'Loss = {loss_avg:.4e}, Accuracy = {accs_avg:.4e}')

        writer.add_scalar('train/loss', loss_avg, global_step=epoch)
        writer.add_scalar('train/accuracy', accs_avg, global_step=epoch)

    return loss_sum, accs_sum
