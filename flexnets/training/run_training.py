import os
import random
from argparse import Namespace
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.tensorboard.writer import SummaryWriter

from flexnets.models import Net
from flexnets.nn.convolution import GeneralizedLehmerConvolution, GeneralizedPowerConvolution

from .train import train
from .utils import freeze_poolings, load_checkpoint, plot_poolings, save_checkpoint
from .validate import validate
from flexnets.data import get_dataloaders
from .utils import freeze_other_params, freeze_poolings, load_checkpoint, save_checkpoint, get_parameters
from flexnets.nn.pooling import GeneralizedLehmerPool2d, GeneralizedPowerMeanPool2d, LPPool2d


def run_training(args: Namespace):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(
        'cpu' if not torch.cuda.is_available() else args.device)

    writer = SummaryWriter(args.logs_dir)

    train_loader, val_loader, test_loader = get_dataloaders(args)

    # if args.pooling_type == 'max_pool2d':
    #     from flexpool.models.vgg import vgg11
    # else:
    #     from flexnets.models.vgglhm import vgg11
    # model = vgg11(True)

    model = Net(args)
    model.to(device)
    # freeze_poolings(model)
    # freeze_other_params(model)

    loss_func = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(get_parameters(model, args.lr), lr=args.lr)
    optimizer = torch.optim.Adam(get_parameters(model, 1e-2), lr=args.lr)
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

    val_loss, val_accuracy = 0, 0
    train_loss, train_accuracy = 0, 0

    best_val_loss = 100
    best_counter = 0
    best_model = None

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
            best_model = model.state_dict()
            best_counter = 0
        else:
            best_counter += 1
        
        if best_counter == 5:
            checkpoint_path = os.path.join(args.save_dir, 'best_model.pth')
            torch.save(best_model, checkpoint_path)
            break

    # results_root = Path(args.save_dir).parent
    df = pd.DataFrame(data={
        'Device': [args.device],
        'Epochs': [args.epochs],
        'Batch size': [args.batch_size],
        'Pooling type': [args.pooling_type],
        'Conv type': [args.conv_type],
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
                (module_types['.'.join(name.split('.')[:-1])] is GeneralizedPowerMeanPool2d) or
                (module_types['.'.join(name.split('.')[:-1])] is GeneralizedLehmerConvolution) or
                (module_types['.'.join(name.split('.')[:-1])] is GeneralizedPowerConvolution)):
            df[name] = [p.data.item()]

    # save_path = os.path.join(results_root, args.run_id)
    # os.makedirs(save_path, exist_ok=True)
    save_path = args.save_dir
    df.to_json(os.path.join(save_path, 'final.json'), orient='records')
