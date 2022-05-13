import os
from datetime import datetime
from argparse import ArgumentParser, Namespace


def add_train_args(parser: ArgumentParser):
    parser.add_argument('--run_id',
                        type=str,
                        default='ex_1',
                        help='Run ID')
    parser.add_argument('--pooling_type',
                        type=str,
                        default='max_pool2d',
                        choices=['max_pool2d', 'generalized_lehmer_pool', 'generalized_power_mean_pool', 'lp_pool'])
    parser.add_argument('--device',
                        type=str,
                        choices=['cuda', 'cpu'],
                        default='cuda',
                        help='Device for training (default: cuda)')
    parser.add_argument('--data_path',
                        type=str,
                        default='./.assets/data/',
                        help='Path to data')
    parser.add_argument('--checkpoint_path',
                        type=str,
                        default=None,
                        help='Path to model checkpoint (.pt file)')
    parser.add_argument('--save_dir',
                        type=str,
                        default='./.assets/checkpoints',
                        help='Directory where model checkpoints will be saved')
    parser.add_argument('--logs_dir',
                        type=str,
                        default='./.assets/logs',
                        help='Directory where Tensorboard logs will be saved')

    parser.add_argument('--seed',
                        type=int,
                        default=12,
                        help='Random seed to use when splitting data into train/val sets (default: 12)')
    parser.add_argument('--split_sizes',
                        type=float,
                        nargs=2,
                        default=[0.8, 0.2],
                        help='Split proportions for train/validation sets')
    parser.add_argument('--workers',
                        type=int,
                        default=4,
                        help='Number of workers for data loading (default: 4)',
                        )
    parser.add_argument('--normalize',
                        action='store_true',
                        default=False
                        )

    # Training arguments
    parser.add_argument('--epochs',
                        type=int,
                        default=30,
                        help='Number of epochs to run (default: 30)')
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--lr',
                        type=float,
                        default=1e-4,
                        help='Initial learning rate (default: 1e-4)')


def add_glm_args(parser: ArgumentParser):
    parser.add_argument('--alpha',
                        type=float,
                        default=2.5,
                        # min=1.0001,
                        # max=2.718,
                        help='Alpha parameter for GLP')
    parser.add_argument('--beta',
                        type=float,
                        default=1.3,
                        # min=-2.5,
                        # max=1.5,
                        help='Beta parameter for GLP')

    parser.add_argument('--norm_type',
                        type=float,
                        default=2,
                        # min=1,
                        # max=4,
                        help='Norm type parameter for LP')


def add_gpm_args(parser: ArgumentParser):
    parser.add_argument('--gamma',
                        type=float,
                        default=2.3,
                        # min=1.0001,
                        # max=2.718,
                        help='Alpha parameter for GPM')
    parser.add_argument('--delta',
                        type=float,
                        default=0.5,
                        # min=-2.5,
                        # max=1.5,
                        help='Beta parameter for GPM')


def modify_train_args(args: Namespace):
    if args.logs_dir is not None:
        timestamp = datetime.now().strftime('%y%m%d-%H%M%S%f')
        log_path = '{}_{}_{}'.format(args.run_id, timestamp, args.pooling_type)
        args.logs_dir = os.path.join(args.logs_dir, log_path)

        if os.path.exists(args.logs_dir):
            num_ctr = 0
            while (os.path.exists(f'{args.logs_dir}_{num_ctr}')):
                num_ctr += 1
            args.logs_dir = f'{args.logs_dir}_{num_ctr}'

        os.makedirs(args.logs_dir)

    if args.save_dir is not None:
        timestamp = datetime.now().strftime('%y%m%d-%H%M%S%f')
        log_path = '{}_{}_{}'.format(args.run_id, timestamp, args.pooling_type)
        args.save_dir = os.path.join(args.save_dir, log_path)

        if os.path.exists(args.save_dir):
            num_ctr = 0
            while (os.path.exists(f'{args.save_dir}_{num_ctr}')):
                num_ctr += 1
            args.save_dir = f'{args.save_dir}_{num_ctr}'

        os.makedirs(args.save_dir)


def parse_train_args() -> Namespace:
    parser = ArgumentParser()
    add_train_args(parser)
    temp_args, unk_args = parser.parse_known_args()
    # if temp_args.pooling_type == 'generalized_lehmer_pool':
    #     add_glm_args(parser)
    # elif temp_args.pooling_type == 'generalized_power_mean_pool':
    #     add_gpm_args(parser)
    # FIXME needed for init layers in main
    add_glm_args(parser)
    add_gpm_args(parser)
    args = parser.parse_args()
    modify_train_args(args)
    return args
