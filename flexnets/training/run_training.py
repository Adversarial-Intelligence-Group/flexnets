import os
from argparse import Namespace
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.tensorboard.writer import SummaryWriter
import numpy as np
import pandas as pd
from pathlib import Path

from flexnets.models import Net
from .train import train
from .validate import validate
from flexnets.data import get_dataloaders
from flexnets.nn.pooling import GeneralizedLehmerPool2d, GeneralizedPowerMeanPool2d
from .utils import freeze_poolings, load_checkpoint, save_checkpoint


def run_training(args: Namespace):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    import random
    random.seed(args.seed)

    device = torch.device(
        'cpu' if not torch.cuda.is_available() else args.device)

    writer = SummaryWriter(args.logs_dir)

    train_loader, val_loader, test_loader = get_dataloaders(args)

    # FIXME
    pools: Dict[str, List] = {
        'max_pool2d': [nn.MaxPool2d,
                       {'kernel_size': 2, 'stride': 2}],
        'generalized_lehmer_pool': [GeneralizedLehmerPool2d,
                                    {'alpha': float(args.alpha), 'beta': float(args.beta),
                                     'kernel_size': 2, 'stride': 2}],
        'generalized_power_mean_pool': [GeneralizedPowerMeanPool2d,
                                        {'gamma': float(args.gamma), 'delta': float(args.delta),
                                         'kernel_size': 2, 'stride': 2}]
    }

    pool = pools.get(args.pooling_type, [nn.MaxPool2d, {
                     'kernel_size': 2, 'stride': 2}])
    model = Net(pool)
    model.to(device)
    # freeze_poolings(model)

    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # FIXME gamma
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=len(train_loader), gamma=0.9)

    start_epoch = 0
    if args.checkpoint_path is not None:
        start_epoch = 4 # FIXME
        print(f'Loading model from {args.checkpoint_path}')
        model = load_checkpoint(
            args.checkpoint_path,
            model,
            optimizer,
            scheduler,
            args.device == 'cuda'
        )
        # freeze_poolings(model)


    val_loss, val_accuracy = 0, 0
    train_loss, train_accuracy = 0, 0

    for epoch in range(start_epoch, start_epoch+args.epochs):
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

    # results_root = Path(args.save_dir).parent
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

    module_types = {key: type(module) for key, module in model.named_modules()}
    for name, p in model.named_parameters():
        if (module_types[name.split('.')[0]] is GeneralizedLehmerPool2d or
                module_types[name.split('.')[0]] is GeneralizedPowerMeanPool2d):
            df[name] = [p.data.item()]

    # save_path = os.path.join(results_root, args.run_id)
    # os.makedirs(save_path, exist_ok=True)
    save_path = args.save_dir
    df.to_json(os.path.join(save_path, 'final.json'), orient='records')
