from flexnets.training import run_training
from flexnets.parsing import parse_train_args

if __name__ == '__main__':
    args = parse_train_args()
    # logger = create_logger(
    #     name='train', save_dir=args.save_dir, quiet=args.quiet)
    run_training(args)
