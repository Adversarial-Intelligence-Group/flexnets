import pytorch_lightning as pl
from sklearn.datasets import load_iris
from sklearn import preprocessing
from pl_bolts.datamodules.sklearn_datamodule import SklearnDataModule

from flexnets.models.mlp import MLP


if __name__ == '__main__':
    pl.seed_everything(42)

    X, Y = load_iris(return_X_y=True)
    X_s = preprocessing.scale(X)

    iris_dm = SklearnDataModule(X_s, Y, val_split=0.1, test_split=0.1, random_state=0, batch_size=15)
    
    mlp = MLP(X.shape[1], 3)
    trainer = pl.Trainer(auto_scale_batch_size=None,
                         gpus=0, deterministic=True, max_epochs=100, log_every_n_steps=5)
    trainer.fit(mlp, iris_dm.train_dataloader(), iris_dm.val_dataloader())
