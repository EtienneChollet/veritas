import os
import torch
from pytorch_lightning import Trainer

from torch.utils.data import DataLoader, random_split
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor

# Custom imports
from veritas.models import Unet
from veritas.data import RealOctDataset
from veritas.synth import RealAug

# Constants
VERSION_N = 11111
MODEL_DIR = 'caroline_models'
PATH = '/autofs/cluster/octdata2/users/epc28/data/caroline_data/unet_training'

N_STEPS = 1e5
N_VOL = 8
TRAIN_TO_VAL_RATIO = 0.5

LEARNING_RATE = 1e-2

total_epochs = int(N_STEPS / (N_VOL*TRAIN_TO_VAL_RATIO))
CHECK_VAL_EVERY_N_EPOCH = int(total_epochs // 100)


def main():
    unet = Unet(
        version_n=VERSION_N,
        synth_params='complex',
        model_dir=MODEL_DIR,
        learning_rate=LEARNING_RATE
        )

    unet.new(
        nb_levels=4,
        nb_features=[32, 64, 128, 256],
        dropout=0,
        augmentation=RealAug()
        )

    dataset = RealOctDataset(PATH)

    seed = torch.Generator().manual_seed(42)
    train_set, val_set = random_split(dataset, [0.5, 0.5], seed)
    logger = TensorBoardLogger(unet.output_path, unet.model_dir, unet.version_n)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_metric_dice", mode="min", every_n_epochs=1,
        save_last=True, filename='{epoch}-{val_loss:.5f}'
        )
    
    trainer_ = Trainer(
        accelerator='gpu',
        check_val_every_n_epoch=CHECK_VAL_EVERY_N_EPOCH,
        logger=logger,
        callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval='step')],
        max_epochs=total_epochs,)

    trainer_.fit(
        unet.trainee,
        DataLoader(train_set, 1, shuffle=True, num_workers=3, persistent_workers=True),
        DataLoader(val_set, 1, shuffle=False)
        )


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')  # allows num_workers > 1
    main()
