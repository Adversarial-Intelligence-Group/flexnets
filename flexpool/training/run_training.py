import os
from argparse import Namespace
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.tensorboard.writer import SummaryWriter
import numpy as np
import pandas as pd
from pathlib import Path

from flexpool.models import Net
from .train import train
from .validate import validate
from flexpool.data import get_dataloaders
from flexpool.nn.pooling import GeneralizedLehmerPool2d, GeneralizedPowerMeanPool2d
from .utils import load_checkpoint, save_checkpoint


def run_training(args: Namespace):
    torch.manual_seed(args.seed)

    device = torch.device(
        'cpu' if not torch.cuda.is_available() else args.device)

    writer = SummaryWriter(args.logs_dir)

    train_loader, val_loader, test_loader = get_dataloaders(args)

    pools: Dict[str, nn.Module] = {
        'max_pool2d': nn.MaxPool2d(kernel_size=2, stride=2),
        'generalized_lehmer_pool': GeneralizedLehmerPool2d(args.alpha, args.beta, kernel_size=2, stride=2),
        'generalized_power_mean_pool': GeneralizedPowerMeanPool2d(args.gamma, args.delta, kernel_size=2, stride=2)
    }

    pool = pools.get(args.pooling_type, nn.MaxPool2d(kernel_size=2, stride=2))
    model = Net(pool)

    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # FIXME gamma
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=len(train_loader), gamma=0.9)

    if args.checkpoint_path is not None:
        print(f'Loading model from {args.checkpoint_path}')
        model = load_checkpoint(
            args.checkpoint_path,
            model,
            optimizer,
            scheduler,
            args.device == 'cuda'
        )

    model.to(device)

    val_loss, val_accuracy = 0, 0
    train_loss, train_accuracy = 0, 0

    for epoch in range(0, args.epochs):
        train_losses, train_accs = train(epoch, model, train_loader, optimizer,
                                         scheduler, loss_func, writer)
        val_losses, val_accs = validate(
            epoch, model, val_loader, loss_func, writer)
        writer.flush()

        train_loss += train_losses
        train_accuracy += train_accs
        val_loss += val_losses
        val_accuracy += val_accs

        if args.save_dir is not None:
            checkpoint_path = os.path.join(
                args.save_dir, str(epoch))
            os.makedirs(checkpoint_path, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_path, 'checkpoint.pth')
            save_checkpoint(checkpoint_path, model, optimizer, scheduler)

    results_root = Path(args.save_dir).parent
    df = pd.DataFrame(data={
        'Device': [args.device],
        'Epochs': [args.epochs],
        'Batch size': [args.batch_size],
        'Pooling type': [args.pooling_type],
        'Alpha': [args.alpha],
        'Beta': [args.beta],
        'Learning rate': [args.lr],
        'Accuracy train': [np.mean(train_accuracy)],
        'Accuracy validation': [np.mean(val_accuracy)],
        'Loss train': [np.mean(train_loss)],
        'Loss validation': [np.mean(val_loss)],
    })

    if (type(model.pool1) is GeneralizedLehmerPool2d or
            type(model.pool1) is GeneralizedPowerMeanPool2d):
        df += pd.DataFrame(data={
            'Pool1 alpha': [model.pool1.alpha.item()],
            'Pool2 alpha': [model.pool2.alpha.item()],
            'Pool3 alpha': [model.pool3.alpha.item()],
            'Pool1 beta': [model.pool1.beta.item()],
            'Pool2 beta': [model.pool2.beta.item()],
            'Pool3 beta': [model.pool3.beta.item()],
        })
    save_path = os.path.join(results_root, args.run_id)
    os.makedirs(save_path, exist_ok=True)
    df.to_csv(os.path.join(save_path, 'final.csv'))
