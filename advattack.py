from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.tensorboard.writer import SummaryWriter
from advertorch.attacks import LinfPGDAttack
from tqdm import tqdm

from flexnets.data import get_dataloaders
from flexnets.models import Net
from flexnets.nn.pooling import (GeneralizedLehmerPool2d,
                                 GeneralizedPowerMeanPool2d,
                                 LPPool2d)
from flexnets.parsing import parse_train_args
from flexnets.training.utils import accuracy
import numpy as np
import random

from sklearn.metrics import f1_score

if __name__ == "__main__":
    args = parse_train_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(
        'cpu' if not torch.cuda.is_available() else 'cuda')

    train_loader, val_loader, test_loader = get_dataloaders(args)

    writer = SummaryWriter(args.logs_dir)

    path = args.checkpoint_path
    
    # epsilons = [0.0001, 0.0005, 0.001, 0.005, 0.008, 0.01, 0.015, 0.02]
    epsilons = np.linspace(0.0001, 0.025, 10)
    print(epsilons)

    model = Net(args)
    model.eval()

    state = torch.load(path)
    # print(state.keys())
    # exit()
    model.load_state_dict(state)
    model.to(device)

    for epoch, eps in enumerate(epsilons):
        writer.add_scalar('pgd/eps', eps, global_step=epoch)
        adversary = LinfPGDAttack(
            model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps,
            nb_iter=5, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0,
            targeted=False)

        clean_accuracy, adv_accuracy = 0, 0
        clean_f1_avg, adv_f1_avg = 0, 0
        pbar = tqdm(test_loader)
        for idx, (cln_data, true_labels) in enumerate(pbar):
            cln_data, true_labels = cln_data.to(device), true_labels.to(device)

            adv_data = adversary.perturb(cln_data, true_labels)

            with torch.no_grad():
                outputs = model(cln_data)
                clean_acc, acc3 = accuracy(outputs, true_labels, (1, 3))

                _, pred = outputs.topk(1, 1, True, True)
                pred = pred.t()[0]
                clean_f1 = f1_score(true_labels.cpu(), pred.cpu(), average='weighted')

                outputs = model(adv_data)
                adv_acc, advacc3 = accuracy(outputs, true_labels, (1, 3))

                _, pred = outputs.topk(1, 1, True, True)
                pred = pred.t()[0]
                adv_f1 = f1_score(true_labels.cpu(), pred.cpu(), average='weighted')

                if isinstance(clean_acc, torch.Tensor):
                    clean_accuracy += clean_acc.item()
                    adv_accuracy += adv_acc.item()
                    clean_f1_avg += clean_f1.item()
                    adv_f1_avg += adv_f1.item()
                else:
                    clean_accuracy += clean_acc
                    adv_accuracy += adv_acc
                    clean_f1_avg += clean_f1
                    adv_f1_avg += adv_f1

                global_step = epoch * len(val_loader) + idx

                writer.add_scalar(
                    'pgd/clean_acc', clean_acc.item(), global_step=global_step)
                writer.add_scalar(
                    'pgd/adv_acc', adv_acc.item(), global_step=global_step)

                writer.add_scalar(
                    'pgd/clean_f1', clean_f1.item(), global_step=global_step)
                writer.add_scalar(
                    'pgd/adv_f1', adv_f1.item(), global_step=global_step)

                writer.add_scalar(
                    'pgd/3_clean_acc', acc3.item(), global_step=global_step)
                writer.add_scalar(
                    'pgd/3_adv_acc', advacc3.item(), global_step=global_step)
                pbar.set_description(
                    f"Acc: {clean_acc.item()}, AdvAcc: {adv_acc.item()}")

        writer.add_scalar(
            'pgd/final_clean', clean_accuracy/len(val_loader), global_step=epoch)
        writer.add_scalar(
            'pgd/final_adv', adv_accuracy/len(val_loader), global_step=epoch)
 
        writer.add_scalar(
            'pgd/final_clean_f1', clean_f1_avg/len(val_loader), global_step=epoch)
        writer.add_scalar(
            'pgd/final_adv_f1', adv_f1_avg/len(val_loader), global_step=epoch)

        writer.flush()
        

        # df[pool_types[pool_id]+"clean_acc"].add(clean_accuracy/len(val_loader))
        # df[pool_types[pool_id]+"adv_acc"].add(adv_accuracy/len(val_loader))

        print("Clean accuracy: ", clean_accuracy/len(val_loader))
        print("adversarial accuracy: ", adv_accuracy/len(val_loader))

        print(epoch)

        # writer.add_images('images/clean', cln_data.detach().cpu().numpy(), epoch)
        # writer.add_images('images/adv', adv_data.detach().cpu().numpy(), epoch)
# lines = df.plot.line()