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
from veritas.synth import OctVolSynth, VascularNetworkDataset
# Eti's superstition
#torch.no_grad()

class Unet(object):
    """
    Base class for UNet.
    """
    def __init__(
            self,
            version_n:int,
            model_dir:str='models',
            synth_params:str='complex',
            synth_dtype:torch.dtype=torch.float32,
            learning_rate:float=1e-3,
            device:str='cuda',
            
        ):
        """
        Parameters
        ----------
        version_n : int
            Version of model.
        model_dir : str
            Subdirectory in which to store training versions.
        synth_params : {'complex', 'simple'}
            How to synthesize.
        device : {'cuda', 'cpu'}
            Device to load UNet onto.
        """
        self.version_n=version_n
        self.model_dir = model_dir
        self.device=device
        self.synth_params=synth_params
        self.output_path="output"
        self.version_path=f"{self.output_path}/{model_dir}/version_{version_n}"
        self.json_path=f"{self.version_path}/json_params.json"
        self.checkpoint_dir = f"{self.version_path}/checkpoints"
        self.losses={0: losses.DiceLoss(labels=[1], activation='Sigmoid')}
        self.metrics = torch.nn.ModuleDict({'dice': self.losses[0]})
        self.synth_dtype=synth_dtype
        self.learning_rate=learning_rate


    def load(self, backbone_dict=None, type='last', mode='train'):
        """
        Load a unet from checkpoint

        Parameters
        ----------
        type : {'last', 'best'}
            Which checkpoint to load from version directory.
        mode : {'train', 'test'}
            Whether model will be used for training or testing purposes.
        """
        # Loading the backbone of the model from json file
        if backbone_dict is None:
            print(f'Loading backbone params from json...')
            self.backbone_dict = JsonTools(self.json_path).read()
        else:
            self.backbone_dict=backbone_dict
        # Instantiating segmentation network
        self.segnet = networks.SegNet(
            3, 1, 1, 3,
            backbone='UNet', activation=None,
            kwargs_backbone=self.backbone_dict
            )
        #self.segnet = torch.compile(self.segnet)

        if mode == 'train':
            trainee = train.SupervisedTrainee(
                network=self.segnet,
                loss=self.losses[0],
                metrics=self.metrics,
                augmentation=OctVolSynth(
                    self.synth_params,
                    dtype=self.synth_dtype),
                lr=self.learning_rate
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
        elif mode == 'test':
            trainee = train.SupervisedTrainee(
                network=self.segnet,
                loss=self.losses[0],
                augmentation=None,
                metrics=self.metrics,
                lr=self.learning_rate
            )
            trainee = train.FineTunedTrainee.load_from_checkpoint(
                checkpoint_path=Checkpoint(self.checkpoint_dir).last(),
                trainee=trainee,
                losses=self.losses
                )
        self.trainee = trainee.to(self.device)
        return trainee
    

    def new(self, nb_levels=4, nb_features=[32,64,128,256], dropout=0.05, nb_conv=2,
            kernel_size=3, activation='ReLU', norm='batch'):
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
    

    def train_it(self, data_experiment_number, train_to_val:float=0.95, batch_size:int=1,
                 epochs=1000, check_val_every_n_epoch:int=1,
                 accumulate_gradient_n_batches:int=1, subset=None,
                 texturize_vessels:bool=True, z_decay:bool=True):
        """
        Train unet after defining or loading model.
        Trainer
        Parameters
        ----------
        data_experiment_number : int
            Dataset that will be used for training model.
        train_to_val : float
            Ratio of training data to validation data for training loop.
        batch_size : int
            Number of volumes per batch.
        epochs : int
            Number of epochs in entire trainig loop.
        check_val_every_n_epoch : int
            Number of times validation dice score is calculated (expressed in epochs).
        accumulate_gradient_n_batches : int
            Number of batches to compute before stepping optimizer.
        subset : int
            Number of samples to use for combined training and validation.
        texturize_vessels : bool
            Whether to use vessel texture in synthesis of intensity volumes.
        z_decay : bool
            Whether to implement z-decay artifact in synthesis of intensity volume.
        i_max : float
            Maximum intensity for synthesized volumes. 
        """
        self.data_experiment_number = data_experiment_number
        self.train_to_val = train_to_val
        self.batch_size = batch_size
        self.epochs = epochs
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.accumulate_gradient_n_batches = accumulate_gradient_n_batches
        self.subset = subset
        self.texturize_vessels = texturize_vessels
        self.z_decay = z_decay
        self.gpus=int(torch.cuda.device_count())
        label_paths = glob(f'output/synthetic_data/exp{data_experiment_number:04d}/*label*')
        # Datasets
        dataset = VascularNetworkDataset(
            inputs=label_paths,
            subset=subset,
            )
        # Splitting up train and val sets
        train_split = train_to_val
        val_split = 1 - train_split
        seed = torch.Generator().manual_seed(42)
        self.train_set, self.val_set = random_split(dataset, [train_split, val_split], seed)
        # Logger and checkpoint stuff
        self.logger = TensorBoardLogger(self.output_path, self.model_dir, self.version_n)
        self.checkpoint_callback = ModelCheckpoint(
            monitor="val_metric_dice", mode="min", every_n_epochs=5,
            save_last=True, filename='{epoch}-{val_loss:.5f}')
        # Sries vs parallel training
        if self.gpus == 1:
            self.sequential_train()
        elif self.gpus >= 2:
            self.distributed_train()

    def sequential_train(self):
        """
        Train on single GPU.
        """
        print('Training on 1 gpu.')
        torch.multiprocessing.set_start_method('spawn')
        # Setting up trainer
        trainer_ = Trainer(
            accelerator='gpu',
            check_val_every_n_epoch=self.check_val_every_n_epoch,
            accumulate_grad_batches=self.accumulate_gradient_n_batches,
            devices=1,
            logger=self.logger,
            callbacks=[self.checkpoint_callback],
            max_epochs=self.epochs,
            #gradient_clip_val=0.5,
            #gradient_clip_algorithm='value'
            )
        # Begin training
        trainer_.fit(
            self.trainee,
            DataLoader(self.train_set, self.batch_size, shuffle=True, num_workers=16, persistent_workers=True),
            DataLoader(self.val_set, self.batch_size, shuffle=False)
            )

    def distributed_train(self):
        """
        Train on multiple GPU devices.
        """
        print(f"Training on {self.gpus} gpus in cluster.")
        mp.set_start_method('spawn', force=True)
        trainer_ = Trainer(
            accelerator='gpu',
            accumulate_grad_batches=self.accumulate_gradient_n_batches,
            check_val_every_n_epoch=self.check_val_every_n_epoch,
            devices=self.gpus,
            strategy='ddp',
            num_nodes=1,
            callbacks=[self.checkpoint_callback],
            logger=self.logger,
            max_epochs=self.epochs,
            use_distributed_sampler=True,
            #gradient_clip_val=0.5,
            #gradient_clip_algorithm='value'
        )
        n_processes = self.gpus
        self.segnet.share_memory()
        processes = []
        for rank in range(n_processes):
            process = mp.Process(target=trainer_.fit(
                self.trainee,
                DataLoader(self.train_set, self.batch_size, shuffle=True),
                DataLoader(self.val_set, self.batch_size, shuffle=False)),
                args=(self.segnet,))
            process.start()
            processes.append(process)
        for proc in processes:
            proc.join()

    #def get_dataloader(self):
    #    self.train_loader = DataLoader(self.train_set, self.batch_size, shuffle=True)