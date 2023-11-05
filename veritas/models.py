__all__ = [
    'Unet'
]
# Standard Imports
import torch
from glob import glob
import torch.multiprocessing as mp
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
# Custom Imports
from vesselseg.vesselseg import networks, losses, train
from vesselseg.vesselseg.synth import SynthVesselDataset
from veritas.utils import PathTools, JsonTools, Checkpoint
# Eti's superstition
#torch.no_grad()

class Unet(object):
    """
    Base class for UNet.
    """
    def __init__(
            self, version_n:int, device='cuda'
        ):
        """
        Parameters
        ----------
        version_n : int
            Version of model.
        device : {'cuda', 'cpu'}
            Device to load UNet onto.
        """
        self.version_n=version_n
        self.output_path="output"
        self.version_path=f"{self.output_path}/models/version_{version_n}"
        self.json_path=f"{self.version_path}/json_params.json"
        self.checkpoint_dir = f"{self.version_path}/checkpoints"
        self.device=device
        self.losses={0: losses.DiceLoss(labels=[1], activation='Sigmoid')}
        self.metrics = torch.nn.ModuleDict({'dice': self.losses[0]})


    def load(self, backbone_dict=None, type='best'):
        if backbone_dict is None:
            print(f'Loading backbone params from json...')
            self.backbone_dict = JsonTools(self.json_path).read()
        else:
            self.backbone_dict=backbone_dict

        self.segnet = networks.SegNet(
            3, 1, 1, 3, activation=None, backbone='UNet',
            kwargs_backbone=self.backbone_dict
            )
        #torch.compile(self.segnet)
        
        trainee = train.SupervisedTrainee(
            network=self.segnet,
            loss=self.losses[0],
            metrics=self.metrics,
            #augmentation=self.augmentation
            )

        if backbone_dict is None:
            print("Loading checkpoint...")
            if type == 'best':
                trainee = train.FineTunedTrainee.load_from_checkpoint(
                    checkpoint_path=Checkpoint(self.checkpoint_dir).best(),
                    trainee=trainee,
                    loss=self.losses
                    )
            elif type == 'last':
                trainee = train.FineTunedTrainee.load_from_checkpoint(
                    checkpoint_path=Checkpoint(self.checkpoint_dir).last(),
                    trainee=trainee,
                    loss=self.losses
                    )
            
            trainee.to(self.device)
            print('Checkpoint loaded')
        else:
            trainee = train.FineTunedTrainee(
                trainee=trainee,
                loss=self.losses
            ).to(self.device)
        self.trainee = trainee
        return trainee


    def new(self, nb_levels=4, nb_features=[32,64,128,256], dropout=0, nb_conv=2,
            kernel_size=3, activation='ReLU', norm='instance'):
        """
        nb_levels : int
            Number of convolutional levels for Unet.
        nb_features : list[int]
            Features per layer. len(list) must equal nb_levels.
        dropout : float
            Percent of data to be dropped randomly.
        nb_conv : int
            Number of convolutions per layer.
        kernel_size : int
            Size of convolutional window.
        activation : str
            Activation to be used for all filters.
        norm : str
            How to normalize layers.
        """
        backbone_dict = {
            "nb_levels": nb_levels,
            "nb_features": nb_features,
            "dropout": dropout,
            "nb_conv": nb_conv,
            "kernel_size": kernel_size,
            "activation": activation,
            "norm": norm
            }
        PathTools(self.version_path).makeDir()
        JsonTools(self.json_path).log(backbone_dict)  
        self.trainee = self.load(backbone_dict)
    

    def train_it(self, data_experiment_number, train_to_val:float=0.8, batch_size:int=1,
                 epochs=1000, loader_device='cuda', check_val_every_n_epoch:int=5,
                 accumulate_gradient_n_batches:int=5, subset=-1, n_workers=0):
        """
        Train unet after defining or loading model.
        
        Parameters
        ----------
        data_path : str
            Path to synthetic volumetric data.
        augmentation : nn.Module
            Volume synthesis/augmentation class.
        train_to_val : float
            Ratio of training data to validation data for training loop.
        batch_size : int
            Number of volumes per batch (before optimizer is stepped)
        epochs : int
            Number of epochs in entire trainig loop.
        loader_device : {'cuda', 'cpu'}
            Device that will be used by dataloaders.
        check_val_every_n_epoch : int
            Number of times validation dice score is calculated as expressed in epochs.
        accumulate_gradient_n_batches : int
            Number of batches to compute before stepping optimizer.
        """
        self.epochs = epochs
        #self.augmentation = augmentation
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.accumulate_gradient_n_batches = accumulate_gradient_n_batches
        self.gpus=int(torch.cuda.device_count())

        train_split = train_to_val
        val_split = 1 - train_split
        seed = torch.Generator().manual_seed(42)
        label_paths = glob(f'output/synthetic_data/exp{data_experiment_number:04d}/*label*')

        from veritas.synth import SynthVesselDatasetv2 as sd
        dataset = sd(label_paths, device=loader_device, subset=slice(subset))
        train_set, val_set = random_split(dataset, [train_split, val_split], seed)

        # Instance variables start
        self.train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=n_workers)
        self.val_loader = DataLoader(val_set, 1, shuffle=False, num_workers=n_workers)
        self.logger = TensorBoardLogger(self.output_path, 'models', self.version_n)
        self.checkpoint_callback = ModelCheckpoint(
            monitor="val_metric_dice", mode="min", every_n_epochs=5,
            save_last=True, filename='{epoch}-{val_loss:.5f}'
            )
        # Instance variables stop

        if self.gpus == 1:
            self.sequential_train()
        elif self.gpus >= 2:
            self.distributed_train()

    def sequential_train(self):
        """
        Train on single GPU.
        """
        print('Training on 1 gpu.')
        #import torch
        torch.multiprocessing.set_start_method('spawn')
        trainer_ = Trainer(
            accelerator=self.device,
            check_val_every_n_epoch=self.check_val_every_n_epoch,
            accumulate_grad_batches=self.accumulate_gradient_n_batches,
            devices=1,
            logger=self.logger,
            callbacks=[self.checkpoint_callback],
            max_epochs=self.epochs,
            )
        trainer_.fit(self.trainee, self.train_loader, self.val_loader)

    def distributed_train(self):
        """
        Train on multiple GPU devices.
        """
        print(f"Training on {self.gpus} gpus in cluster.")
        mp.set_start_method('spawn', force=True)
        trainer_ = Trainer(
            accelerator=self.device,
            accumulate_grad_batches=self.accumulate_gradient_n_batches,
            check_val_every_n_epoch=self.check_val_every_n_epoch,
            devices=self.gpus,
            strategy='ddp',
            num_nodes=1,
            callbacks=[self.checkpoint_callback],
            logger=self.logger,
            max_epochs=self.epochs
        )
        n_processes = self.gpus
        self.segnet.share_memory()
        processes = []
        #torch.cuda.empty_cache()
        for rank in range(n_processes):
            process = mp.Process(target=trainer_.fit(
                self.trainee,
                self.train_loader,
                self.val_loader),
                args=(self.segnet,))
            process.start()
            processes.append(process)
        for proc in processes:
            proc.join()