from flexnets.training import run_training
from flexnets.parsing import parse_train_args

if __name__ == '__main__':
    args = parse_train_args()
    run_training(args)
