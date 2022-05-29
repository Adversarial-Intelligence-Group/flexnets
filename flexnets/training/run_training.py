import os
import random
from argparse import Namespace
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from flexpool.data import get_dataloaders
from flexpool.models import Net
from flexpool.nn.pooling import (GeneralizedLehmerPool2d,
                                 GeneralizedPowerMeanPool2d)
from torch.utils.tensorboard.writer import SummaryWriter

from flexnets.models import Net
from .train import train
from .utils import freeze_poolings, load_checkpoint, plot_poolings, save_checkpoint
from .validate import validate
from flexnets.data import get_dataloaders
from flexnets.nn.pooling import GeneralizedLehmerPool2d, GeneralizedPowerMeanPool2d
from .utils import freeze_other_params, freeze_poolings, load_checkpoint, save_checkpoint, get_parameters


def run_training(args: Namespace):

    device = torch.device(
        'cpu' if not torch.cuda.is_available() else args.device)

    writer = SummaryWriter(args.logs_dir)

    trainset = ImageFolder('./.assets/data/catsdogs/train/', transform=tf.Compose([tf.Resize(256),
                                                                                   tf.RandomCrop(
                                                                                       224, 224),
                                                                                   tf.ToTensor()]))
    valset = ImageFolder('./.assets/data/catsdogs/val/', transform=tf.Compose([tf.Resize(256),
                                                                               tf.RandomCrop(
                                                                                   224, 224),
                                                                               tf.ToTensor()]))
    train_loader = DataLoader(
        trainset, 8, True, drop_last=True, num_workers=6)
    val_loader = DataLoader(valset, 8, False, drop_last=True, num_workers=6)

    # FIXME
    # pools: Dict[str, List] = {
    #     'max_pool2d': [nn.MaxPool2d,
    #                    {'kernel_size': 2, 'stride': 2}],
    #     'generalized_lehmer_pool': [GeneralizedLehmerPool2d,
    #                                 {'alpha': float(args.alpha), 'beta': float(args.beta),
    #                                  'kernel_size': 2, 'stride': 2}],
    #     'generalized_power_mean_pool': [GeneralizedPowerMeanPool2d,
    #                                     {'gamma': float(args.gamma), 'delta': float(args.delta),
    #                                      'kernel_size': 2, 'stride': 2}]
    # }

    # pool = pools.get(args.pooling_type, [nn.MaxPool2d, {
    #                  'kernel_size': 2, 'stride': 2}])
    # model = Net(pool)
    if args.pooling_type == 'max_pool2d':
        from flexpool.models.vgg import vgg11
    else:
        from flexpool.models.vgglhm import vgg11
    model = vgg11(True)
    pools: Dict[str, List] = {
        'max_pool2d': [nn.MaxPool2d,
                       {'kernel_size': 2, 'stride': 2}],
        'generalized_lehmer_pool': [GeneralizedLehmerPool2d,
                                    {'alpha': float(args.alpha), 'beta': float(args.beta),
                                     'kernel_size': 2, 'stride': 2}],
        'generalized_power_mean_pool': [GeneralizedPowerMeanPool2d,
                                        {'gamma': float(args.gamma), 'delta': float(args.delta),
                                         'kernel_size': 2, 'stride': 2}],
        "lp_pool": [LPPool2d,
                       {'norm_type': args.norm_type, 'kernel_size': 2, 'stride': 2}]
    }

    pool = pools.get(args.pooling_type, [nn.MaxPool2d, {
                     'kernel_size': 2, 'stride': 2}])
    model = Net(pool)
    model.to(device)
    # freeze_poolings(model)

    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(get_parameters(model, args.lr), lr=args.lr)
    # optimizer = torch.optim.Adam(get_parameters(model, args.lr), lr=args.lr)
    # optimizer = torch.optim.Adam([{"params": model.pool1.parameters(), "lr": 0.00023},
    #                               {"params": model.pool2.parameters(), "lr": 0.00038},
    #                               {"params": model.pool3.parameters(), "lr": 0.00061},
    #                               {"params": model.block1.parameters()},
    #                               {"params": model.block2.parameters()},
    #                               {"params": model.block3.parameters()},
    #                               {"params": model.drop2.parameters()},
    #                               {"params": model.fc_layer.parameters()},
    #                               ],
    #                              lr=1e-3)
    # FIXME gamma
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=0.9)

    start_epoch = 0
    if args.checkpoint_path is not None:
        start_epoch = 10 # FIXME
        print(f'Loading model from {args.checkpoint_path}')
        model = load_checkpoint(
            args.checkpoint_path,
            model,
            optimizer,
            scheduler,
            args.device == 'cuda'
        )
    freeze_other_params(model)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=len(train_loader), gamma=0.9)

    val_loss, val_accuracy = 0, 0
    train_loss, train_accuracy = 0, 0

    best_val_loss = 100
    best_counter = 0

    for epoch in range(start_epoch, start_epoch+args.epochs):
        train_losses, train_accs = train(epoch, model, train_loader, optimizer,
                                         None, loss_func, writer)
        val_losses, val_accs = validate(
            epoch, model, val_loader, loss_func, writer)
        scheduler.step()

        writer.flush()

        train_loss += train_losses
        train_accuracy += train_accs
        val_loss += val_losses
        val_accuracy += val_accs

        if args.save_dir is not None:
            # checkpoint_path = os.path.join(
            #     args.save_dir, str(epoch))
            checkpoint_path = args.save_dir
            os.makedirs(checkpoint_path, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_path, 'checkpoint.pth')
            save_checkpoint(checkpoint_path, model, optimizer, scheduler)
        
        if best_val_loss >= (val_losses/len(val_loader)):
            best_val_loss = (val_losses/len(val_loader))
            best_counter = 0
        else:
            best_counter += 1
        
        if best_counter == 5:
            break

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
        # if (module_types[name.split('.')[0]] is GeneralizedLehmerPool2d or
        #         module_types[name.split('.')[0]] is GeneralizedPowerMeanPool2d):
        if ((module_types['.'.join(name.split('.')[:-1])] is GeneralizedLehmerPool2d) or
                (module_types['.'.join(name.split('.')[:-1])] is GeneralizedPowerMeanPool2d)):
            df[name] = [p.data.item()]

    # save_path = os.path.join(results_root, args.run_id)
    # os.makedirs(save_path, exist_ok=True)
    save_path = args.save_dir
    df.to_json(os.path.join(save_path, 'final.json'), orient='records')
