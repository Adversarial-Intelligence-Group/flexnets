import pytorch_lightning as pl
from sklearn import preprocessing
from sklearn.datasets import load_iris
from pl_bolts.datamodules.sklearn_datamodule import SklearnDataModule

from flexnets.models.gln import GeneralizedLehmerNetwork


if __name__ == '__main__':
    pl.seed_everything(42)

    X, Y = load_iris(return_X_y=True)
    X_s = preprocessing.scale(X)

    iris_dm = SklearnDataModule(
        X_s, Y, test_split=0.2, random_state=2, batch_size=15)

    mlp = GeneralizedLehmerNetwork(4, 3)
    trainer = pl.Trainer(auto_scale_batch_size=None,
                         gpus=0, deterministic=True, max_epochs=100, log_every_n_steps=5)
    trainer.fit(mlp, iris_dm.train_dataloader(), iris_dm.val_dataloader())
