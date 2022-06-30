import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from sklearn.datasets import load_iris
from sklearn import preprocessing
from pl_bolts.datamodules.sklearn_datamodule import SklearnDataModule
from flexnets.data.data import get_dataloaders

from flexnets.models.gln import GeneralizedLehmerNetwork
from flexnets.models.mlp import MLP
from flexnets.parsing import parse_train_args


if __name__ == '__main__':
    pl.seed_everything(42)

    # args = parse_train_args()
    # train_loader, val_loader, test_loader = get_dataloaders(args)

    # mlp = GeneralizedLehmerNetwork(32 * 32 * 3, 10)
    # net = MLP(32 * 32 * 3, 10)
    # logger = TensorBoardLogger(args.logs_dir, name=args.run_id)


    X, Y = load_iris(return_X_y=True)
    X_s = preprocessing.scale(X)

    iris_dm = SklearnDataModule(X_s, Y, test_split=0.2, random_state=2, batch_size=15)
    
    # mlp = MLP(4, 3)
    mlp = GeneralizedLehmerNetwork(4, 3)
    trainer = pl.Trainer(auto_scale_batch_size=None,
                         gpus=0, deterministic=True, max_epochs=100, log_every_n_steps=5)
    trainer.fit(mlp, iris_dm.train_dataloader(), iris_dm.val_dataloader())
    # trainer.fit(mlp, train_loader, val_loader)
